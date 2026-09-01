// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MOONCAKE_TRANSFER_ENGINE_PXN_REGISTRY_H_
#define MOONCAKE_TRANSFER_ENGINE_PXN_REGISTRY_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "common/base/status.h"
#include "transport/pxn_transport/pxn_core.h"

namespace mooncake {
namespace pxn {

inline constexpr uint64_t kRegistryMagic = 0x4D4350584E524547ULL;
inline constexpr uint32_t kRegistryAbiVersion = 2;
inline constexpr size_t kMaxRailNameLength = 256;
inline constexpr size_t kCudaIpcHandleSize = 64;

enum class RegistryState : uint32_t {
    kInitializing = 0,
    kReady = 1,
    kStopping = 2,
};

enum class LaneState : uint32_t {
    kFree = 0,
    kClaiming = 1,
    kReady = 2,
};

struct ProcessIdentity {
    uint32_t uid;
    int32_t pid;
    uint64_t start_ticks;

    bool operator==(const ProcessIdentity&) const = default;
};

struct CudaIpcHandle {
    uint8_t bytes[kCudaIpcHandleSize];

    bool operator==(const CudaIpcHandle&) const = default;
};

struct alignas(64) RegistryHeader {
    uint64_t magic;
    uint32_t abi_version;
    uint32_t header_bytes;
    uint64_t mapping_bytes;
    uint32_t state;
    uint32_t rail_bytes;
    ProcessIdentity owner;
    uint64_t epoch;
    uint64_t heartbeat;
    uint32_t lane_count;
    uint32_t slots_per_lane;
    uint64_t slot_size;
    uint64_t arena_size;
    uint32_t ipc_handle_bytes;
    uint32_t rail_count;
    char rail[kMaxRailNameLength];
    CudaIpcHandle arena_handle;
    uint8_t padding[32];
};

struct alignas(64) LaneHeader {
    uint32_t state;
    uint32_t arena_attached;
    ProcessIdentity sender;
    uint64_t sender_epoch;
    uint64_t head;
    uint64_t doorbell;
    uint64_t completed;
    uint64_t padding;
};

struct alignas(64) LaneControl {
    LaneHeader header;
    Descriptor descriptors[kMaxSlotsPerLane];
    Completion completions[kMaxSlotsPerLane];
};

struct alignas(64) ControlBlock {
    RegistryHeader header;
    LaneControl lanes[kMaxLaneCount];
};

static_assert(sizeof(ProcessIdentity) == 16);
static_assert(alignof(ProcessIdentity) == 8);
static_assert(sizeof(CudaIpcHandle) == kCudaIpcHandleSize);
static_assert(alignof(RegistryHeader) == 64);
static_assert(offsetof(RegistryHeader, state) == 24);
static_assert(offsetof(RegistryHeader, owner) == 32);
static_assert(offsetof(RegistryHeader, epoch) == 48);
static_assert(offsetof(RegistryHeader, lane_count) == 64);
static_assert(offsetof(RegistryHeader, rail) == 96);
static_assert(offsetof(RegistryHeader, arena_handle) == 352);
static_assert(sizeof(RegistryHeader) == 448);
static_assert(alignof(LaneHeader) == 64);
static_assert(offsetof(LaneHeader, sender) == 8);
static_assert(offsetof(LaneHeader, sender_epoch) == 24);
static_assert(sizeof(LaneHeader) == 64);
static_assert(alignof(LaneControl) == 64);
static_assert(sizeof(LaneControl) == 1186624);
static_assert(alignof(ControlBlock) == 64);
static_assert(sizeof(ControlBlock) == 8306816);
static_assert(std::is_standard_layout_v<RegistryHeader>);
static_assert(std::is_trivially_copyable_v<RegistryHeader>);
static_assert(std::is_standard_layout_v<LaneHeader>);
static_assert(std::is_trivially_copyable_v<LaneHeader>);
static_assert(std::is_standard_layout_v<ControlBlock>);
static_assert(std::is_trivially_copyable_v<ControlBlock>);

using ProcessProbe = std::function<std::optional<bool>(const ProcessIdentity&)>;

struct RegistryOptions {
    std::string root_directory = "/dev/shm";
    std::string group_id;
    std::optional<ProcessIdentity> identity;
    uint64_t epoch = 0;
    ProcessProbe process_probe;
};

struct RegistryEntry {
    ProcessIdentity identity;
    uint64_t epoch;
    std::vector<std::string> rails;
    CudaIpcHandle arena_handle;
    std::string file_name;
};

class RegistryRegistration {
   public:
    ~RegistryRegistration();

    RegistryRegistration(const RegistryRegistration&) = delete;
    RegistryRegistration& operator=(const RegistryRegistration&) = delete;

    Status publish();
    Status unpublish();
    Status activeAttachments(size_t& count);
    ControlBlock* control() const { return control_; }
    bool published() const { return published_; }

   private:
    friend class Registry;
    RegistryRegistration(int directory_fd, int lock_fd, int fd,
                         std::string temporary_name, std::string final_name,
                         ControlBlock* control, ProcessIdentity identity,
                         ProcessProbe process_probe,
                         std::shared_ptr<std::mutex> group_mutex);

    int directory_fd_ = -1;
    int lock_fd_ = -1;
    int fd_ = -1;
    std::string temporary_name_;
    std::string final_name_;
    ControlBlock* control_ = nullptr;
    ProcessIdentity identity_{};
    ProcessProbe process_probe_;
    std::shared_ptr<std::mutex> group_mutex_;
    std::mutex entry_mutex_;
    bool published_ = false;
};

class PeerMapping {
   public:
    ~PeerMapping();

    PeerMapping(const PeerMapping&) = delete;
    PeerMapping& operator=(const PeerMapping&) = delete;

    Status claimLane(const ProcessIdentity& sender, uint64_t sender_epoch,
                     size_t& lane_index);
    Status releaseLane(const ProcessIdentity& sender, uint64_t sender_epoch,
                       size_t lane_index);
    Status attachArena();
    Status detachArena();
    const RegistryEntry& entry() const { return entry_; }
    ControlBlock* control() const { return control_; }

   private:
    friend class Registry;
    PeerMapping(int fd, RegistryEntry entry, ControlBlock* control,
                ProcessProbe process_probe);

    int fd_ = -1;
    RegistryEntry entry_;
    ControlBlock* control_ = nullptr;
    ProcessProbe process_probe_;
    std::mutex mutex_;
    ProcessIdentity sender_{};
    uint64_t sender_epoch_ = 0;
    size_t lane_index_ = kMaxLaneCount;
    bool arena_attached_ = false;
};

class Registry {
   public:
    ~Registry();

    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;

    static Status Open(RegistryOptions options,
                       std::unique_ptr<Registry>& registry);

    Status createLocal(std::span<const std::string> rails,
                       const CudaIpcHandle& arena_handle,
                       std::unique_ptr<RegistryRegistration>& registration);
    Status discover(std::vector<RegistryEntry>& entries);
    Status mapPeer(const RegistryEntry& entry,
                   std::unique_ptr<PeerMapping>& mapping);

    const ProcessIdentity& identity() const { return identity_; }
    uint64_t epoch() const { return epoch_; }

   private:
    Registry(int root_fd, int directory_fd, int lock_fd,
             ProcessIdentity identity, uint64_t epoch,
             ProcessProbe process_probe,
             std::shared_ptr<std::mutex> group_mutex);

    int root_fd_ = -1;
    int directory_fd_ = -1;
    int lock_fd_ = -1;
    ProcessIdentity identity_{};
    uint64_t epoch_ = 0;
    ProcessProbe process_probe_;
    std::shared_ptr<std::mutex> group_mutex_;
};

bool isValidGroupId(std::string_view group_id);
Status parseProcessStat(std::string_view contents, uint64_t& start_ticks,
                        char& state);
Status readProcessIdentity(int32_t pid, ProcessIdentity& identity);
uint32_t loadRegistryState(const RegistryHeader& header);
uint32_t loadLaneState(const LaneHeader& header);

}  // namespace pxn
}  // namespace mooncake

#endif  // MOONCAKE_TRANSFER_ENGINE_PXN_REGISTRY_H_

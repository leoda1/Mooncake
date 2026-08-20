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

#ifndef MOONCAKE_TRANSFER_ENGINE_PXN_STAGING_H_
#define MOONCAKE_TRANSFER_ENGINE_PXN_STAGING_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/base/status.h"
#include "transport/pxn/pxn_registry.h"

namespace mooncake {
class RdmaTransport;

namespace pxn {

class StagingBackend {
   public:
    virtual ~StagingBackend() = default;

    virtual Status allocateArena(size_t size, uintptr_t& address) = 0;
    virtual Status freeArena(uintptr_t address) = 0;
    virtual Status registerArena(uintptr_t address, size_t size) = 0;
    virtual Status unregisterArena(uintptr_t address) = 0;
    virtual Status exportArena(uintptr_t address, CudaIpcHandle& handle) = 0;
    virtual Status importArena(const CudaIpcHandle& handle,
                               uintptr_t& address) = 0;
    virtual Status closeImportedArena(uintptr_t address) = 0;
};

class PeerResources {
   public:
    ~PeerResources();

    PeerResources(const PeerResources&) = delete;
    PeerResources& operator=(const PeerResources&) = delete;

    Status shutdown();
    const RegistryEntry& entry() const { return entry_; }
    size_t laneIndex() const { return lane_index_; }
    uintptr_t address() const { return address_; }
    std::optional<uintptr_t> slotAddress(size_t slot_index) const;
    LaneControl* laneControl() const;

   private:
    friend class LocalResources;
    PeerResources(StagingBackend& backend, std::unique_ptr<PeerMapping> mapping,
                  RegistryEntry entry, ProcessIdentity sender,
                  uint64_t sender_epoch, size_t lane_index);

    Status importArena(const CudaIpcHandle& handle);
    void quarantine();

    StagingBackend* backend_;
    std::unique_ptr<PeerMapping> mapping_;
    RegistryEntry entry_;
    ProcessIdentity sender_;
    uint64_t sender_epoch_;
    size_t lane_index_;
    uintptr_t address_ = 0;
    bool imported_ = false;
    bool shutdown_ = false;
    bool quarantined_ = false;
};

class StagingArena {
   public:
    StagingArena() = default;
    ~StagingArena();

    StagingArena(const StagingArena&) = delete;
    StagingArena& operator=(const StagingArena&) = delete;

    Status initialize(StagingBackend& backend);
    Status shutdown();
    void abandon();

    uintptr_t address() const { return address_; }
    const CudaIpcHandle& handle() const { return handle_; }
    std::optional<uintptr_t> laneAddress(size_t lane_index) const;
    std::optional<uintptr_t> slotAddress(size_t lane_index,
                                         size_t slot_index) const;
    bool ready() const { return ready_; }
    bool quarantined() const { return quarantined_; }

   private:
    StagingBackend* backend_ = nullptr;
    uintptr_t address_ = 0;
    CudaIpcHandle handle_{};
    bool allocated_ = false;
    bool registered_ = false;
    bool ready_ = false;
    bool quarantined_ = false;
};

class LocalResources {
   public:
    ~LocalResources();

    LocalResources(const LocalResources&) = delete;
    LocalResources& operator=(const LocalResources&) = delete;

    static Status Create(
        RegistryOptions registry_options,
        std::span<const std::string> local_hcas,
        const std::unordered_map<std::string, std::string>& rail_map,
        std::unique_ptr<StagingBackend> backend,
        std::unique_ptr<LocalResources>& resources);

    Status shutdown();
    Status mapPeer(const RegistryEntry& entry, PeerResources*& peer);
    const std::string& localRail() const { return local_rail_; }
    Registry& registry() const { return *registry_; }
    RegistryRegistration& registration() const { return *registration_; }
    StagingArena& arena() { return arena_; }

   private:
    LocalResources(std::unique_ptr<StagingBackend> backend,
                   std::unique_ptr<Registry> registry, std::string local_rail);

    void quarantine();

    std::unique_ptr<StagingBackend> backend_;
    std::unique_ptr<Registry> registry_;
    std::unique_ptr<RegistryRegistration> registration_;
    StagingArena arena_;
    std::vector<std::unique_ptr<PeerResources>> peers_;
    std::string local_rail_;
    bool shutdown_ = false;
    bool quarantined_ = false;
};

std::unique_ptr<StagingBackend> makeCudaRdmaStagingBackend(
    RdmaTransport& transport, int device_id);

}  // namespace pxn
}  // namespace mooncake

#endif  // MOONCAKE_TRANSFER_ENGINE_PXN_STAGING_H_

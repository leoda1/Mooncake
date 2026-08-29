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

#ifndef MOONCAKE_TRANSFER_ENGINE_PXN_TRANSPORT_H_
#define MOONCAKE_TRANSFER_ENGINE_PXN_TRANSPORT_H_

#include <atomic>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/base/status.h"
#include "transport/pxn_transport/pxn_core.h"
#include "transport/pxn_transport/pxn_registry.h"

namespace mooncake {
class RdmaTransport;

namespace pxn {

class QuarantineLatch {
   public:
    bool quarantined() const { return quarantined_; }
    bool trip() {
        if (quarantined_) return false;
        quarantined_ = true;
        return true;
    }

   private:
    bool quarantined_ = false;
};

// Same as QuarantineLatch, but safe to trip/query from multiple threads.
class AtomicQuarantineLatch {
   public:
    bool quarantined() const { return quarantined_.load(); }
    bool trip() {
        bool expected = false;
        return quarantined_.compare_exchange_strong(expected, true);
    }

   private:
    std::atomic<bool> quarantined_{false};
};

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
    ControlBlock* controlBlock() const {
        return mapping_ == nullptr ? nullptr : mapping_->control();
    }

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
    QuarantineLatch quarantine_;
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
    bool quarantined() const { return quarantine_.quarantined(); }

   private:
    StagingBackend* backend_ = nullptr;
    uintptr_t address_ = 0;
    CudaIpcHandle handle_{};
    bool allocated_ = false;
    bool registered_ = false;
    bool ready_ = false;
    QuarantineLatch quarantine_;
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
    const std::vector<std::string>& localRails() const { return local_rails_; }
    bool ownsRail(std::string_view rail) const;
    Registry& registry() const { return *registry_; }
    RegistryRegistration& registration() const { return *registration_; }
    StagingArena& arena() { return arena_; }

   private:
    LocalResources(std::unique_ptr<StagingBackend> backend,
                   std::unique_ptr<Registry> registry,
                   std::vector<std::string> local_rails);

    void quarantine();

    std::unique_ptr<StagingBackend> backend_;
    std::unique_ptr<Registry> registry_;
    std::unique_ptr<RegistryRegistration> registration_;
    StagingArena arena_;
    std::vector<std::unique_ptr<PeerResources>> peers_;
    std::vector<std::string> local_rails_;
    bool shutdown_ = false;
    QuarantineLatch quarantine_;
};

struct LogicalSlice {
    void* context = nullptr;
    size_t length = 0;
    void (*mark_posted)(void*) = nullptr;
    void (*mark_success)(void*) = nullptr;
    void (*mark_failed)(void*) = nullptr;
};

struct SenderSubmission {
    Piece piece;
    std::string session;
    uint64_t target_id = 0;
    uint32_t rail_index = 0;
    std::vector<LogicalSlice> slices;
};

struct SenderCopy {
    uintptr_t source;
    uintptr_t destination;
    size_t length;
};

struct SenderLaneEndpoint {
    uint64_t epoch = 0;
    uintptr_t arena_address = 0;
    size_t lane_index = kLaneCount;
    ControlBlock* control = nullptr;
};

class SenderReadyFence {
   public:
    virtual ~SenderReadyFence() = default;
};

class SenderLaneHandle {
   public:
    virtual ~SenderLaneHandle() = default;
};

enum class SenderPublishState {
    kPublished,
    kPendingCpuDoorbell,
};

class SenderBackend {
   public:
    virtual ~SenderBackend() = default;

    virtual Status createLane(const SenderLaneEndpoint& endpoint,
                              std::unique_ptr<SenderLaneHandle>& handle) = 0;
    virtual Status recordReady(std::unique_ptr<SenderReadyFence>& fence) = 0;
    virtual Status queryReady(const SenderReadyFence& fence, bool& ready) = 0;
    virtual Status publish(SenderLaneHandle& handle,
                           const SenderReadyFence& fence,
                           const std::vector<SenderCopy>& copies,
                           uint64_t sequence, SenderPublishState& state) = 0;
    virtual Status pollCpuDoorbell(SenderLaneHandle& handle, bool& ready) = 0;
};

class FallbackTransfer {
   public:
    virtual ~FallbackTransfer() = default;

    virtual Status poll(bool& completed) = 0;
    virtual void abandon() = 0;
};

class SenderFallback {
   public:
    virtual ~SenderFallback() = default;

    virtual Status submit(const SenderSubmission& submission,
                          std::unique_ptr<FallbackTransfer>& transfer) = 0;
};

class SenderLane {
   public:
    ~SenderLane();

    SenderLane(const SenderLane&) = delete;
    SenderLane& operator=(const SenderLane&) = delete;

    static Status Create(SenderLaneEndpoint endpoint, SenderBackend& backend,
                         SenderFallback& fallback,
                         std::chrono::milliseconds credit_timeout,
                         std::unique_ptr<SenderLane>& lane);

    Status enqueue(SenderSubmission submission);
    Status reap(bool& made_progress);
    Status progress(bool& made_progress);
    Status shutdown();

    uint64_t nextSequence() const { return next_sequence_; }
    uint64_t reapedSequence() const { return reaped_sequence_; }
    size_t inflightCount() const { return inflight_.size(); }
    bool quarantined() const { return quarantine_.quarantined(); }
    uint64_t fallbackBytes() const {
        return fallback_bytes_.load(std::memory_order_relaxed);
    }

   private:
    struct QueuedSubmission {
        SenderSubmission submission;
        std::unique_ptr<SenderReadyFence> fence;
        Status ready_status;
        std::chrono::steady_clock::time_point enqueue_time;
    };

    struct PublishedSubmission {
        uint64_t sequence;
        std::vector<LogicalSlice> slices;
    };

    SenderLane(SenderLaneEndpoint endpoint, SenderBackend& backend,
               SenderFallback& fallback,
               std::chrono::milliseconds credit_timeout,
               std::unique_ptr<SenderLaneHandle> handle);

    Status deferOrStartFallback(QueuedSubmission submission);
    Status startFallback(QueuedSubmission submission);
    Status progressFallbackQueue(bool& made_progress);
    Status pollFallbacks(bool& made_progress);
    Status publish(QueuedSubmission submission, uint64_t sequence,
                   SenderPublishState& state, bool& quarantine_lane);
    void finishPublication(QueuedSubmission submission, uint64_t sequence,
                           bool cpu_doorbell);
    void failSubmission(const SenderSubmission& submission);

    SenderLaneEndpoint endpoint_;
    SenderBackend& backend_;
    SenderFallback& fallback_;
    std::chrono::milliseconds credit_timeout_;
    std::unique_ptr<SenderLaneHandle> handle_;
    std::mutex queue_mutex_;
    std::deque<QueuedSubmission> queue_;
    std::deque<QueuedSubmission> fallback_queue_;
    std::deque<PublishedSubmission> inflight_;
    std::vector<std::unique_ptr<FallbackTransfer>> fallback_transfers_;
    std::unique_ptr<QueuedSubmission> pending_publication_;
    uint64_t pending_sequence_ = 0;
    uint64_t next_sequence_ = 1;
    uint64_t reaped_sequence_ = 0;
    std::atomic<bool> accepting_{true};
    std::atomic<uint64_t> fallback_bytes_{0};
    AtomicQuarantineLatch quarantine_;
    bool shutdown_ = false;
};

class SenderPipeline {
   public:
    SenderPipeline(std::unique_ptr<SenderBackend> backend,
                   std::unique_ptr<SenderFallback> fallback,
                   std::chrono::milliseconds credit_timeout);
    ~SenderPipeline();

    SenderPipeline(const SenderPipeline&) = delete;
    SenderPipeline& operator=(const SenderPipeline&) = delete;

    Status addPeer(PeerResources& peer, SenderLane*& lane);
    Status reapOutbound(bool& made_progress);
    Status progressOutbound(bool& made_progress);
    Status shutdown();

    // Sum of every lane's fallback byte counter.
    uint64_t fallbackBytes();

   private:
    struct PeerLaneEntry {
        ProcessIdentity identity;
        uint64_t epoch;
        SenderLane* lane;
    };

    std::unique_ptr<SenderBackend> backend_;
    std::unique_ptr<SenderFallback> fallback_;
    std::chrono::milliseconds credit_timeout_;
    std::mutex lanes_mutex_;
    std::vector<std::unique_ptr<SenderLane>> lanes_;
    std::vector<PeerLaneEntry> peer_lanes_;
    size_t next_lane_ = 0;
    bool shutdown_ = false;
};

struct RelaySubmission {
    uintptr_t source = 0;
    std::string session;
    uint32_t rail_index = 0;
    std::vector<PlanEntry> plans;
};

class RelayTransfer {
   public:
    virtual ~RelayTransfer() = default;
    virtual Status poll(bool& completed, int32_t& completion_status) = 0;
    virtual void abandon() = 0;
};

class RelayBackend {
   public:
    virtual ~RelayBackend() = default;
    virtual Status submit(const RelaySubmission& submission,
                          std::unique_ptr<RelayTransfer>& transfer) = 0;
};

class RelayPipeline {
   public:
    RelayPipeline(ControlBlock* control, uintptr_t arena_address,
                  uint64_t epoch, size_t max_inflight,
                  std::unique_ptr<RelayBackend> backend);
    ~RelayPipeline();

    Status reapInbound(bool& made_progress);
    Status progressInbound(bool& made_progress);
    void shutdown();

    // Bytes successfully forwarded by the relay's second hop.
    uint64_t relayedBytes() const {
        return relayed_bytes_.load(std::memory_order_relaxed);
    }

   private:
    struct Inflight {
        uint64_t sequence;
        std::unique_ptr<RelayTransfer> transfer;
    };

    void complete(size_t lane_index, uint64_t sequence, int32_t status);

    ControlBlock* control_;
    uintptr_t arena_address_;
    uint64_t epoch_;
    size_t max_inflight_;
    std::unique_ptr<RelayBackend> backend_;
    std::array<uint64_t, kLaneCount> next_sequence_{};
    std::array<uint64_t, kLaneCount> lane_sender_epoch_{};
    std::array<std::deque<Inflight>, kLaneCount> inflight_;
    std::atomic<uint64_t> relayed_bytes_{0};
    size_t inflight_count_ = 0;
    size_t next_lane_ = 0;
};

class PxnPump {
   public:
    PxnPump(SenderPipeline& sender, RelayPipeline& relay);
    ~PxnPump();

    void wake();
    void shutdown();

   private:
    void run();

    SenderPipeline& sender_;
    RelayPipeline& relay_;
    std::atomic<bool> running_{true};
    std::mutex mutex_;
    std::condition_variable condition_;
    std::thread thread_;
};

std::unique_ptr<StagingBackend> makeCudaRdmaStagingBackend(
    RdmaTransport& transport, int device_id);
std::unique_ptr<SenderBackend> makeCudaSenderBackend(int device_id);
std::unique_ptr<SenderFallback> makeRdmaSenderFallback(
    RdmaTransport& transport);
Status makeRdmaRelayBackend(RdmaTransport& transport, uintptr_t arena_address,
                            std::span<const std::string> source_device_names,
                            std::span<const std::string> source_rails,
                            size_t max_inflight,
                            std::unique_ptr<RelayBackend>& backend);

}  // namespace pxn
}  // namespace mooncake

#endif  // MOONCAKE_TRANSFER_ENGINE_PXN_TRANSPORT_H_

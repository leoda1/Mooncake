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

#include "transport/pxn/pxn_transport.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "common.h"
#include "config.h"
#include "error.h"
#include "transport/rdma_transport/rdma_context.h"
#include "transport/rdma_transport/rdma_transport.h"

namespace mooncake {
namespace pxn {
namespace {

class CompletedRelayTransfer final : public RelayTransfer {
   public:
    explicit CompletedRelayTransfer(int32_t status) : status_(status) {}

    Status poll(bool& completed, int32_t& completion_status) override {
        completed = true;
        completion_status = status_;
        return Status::OK();
    }

    void abandon() override {}

   private:
    int32_t status_;
};

class RelaySlicePool {
   public:
    explicit RelaySlicePool(size_t capacity)
        : storage_(std::make_unique<Transport::Slice[]>(capacity)) {
        free_.reserve(capacity);
        for (size_t index = 0; index < capacity; ++index) {
            free_.push_back(&storage_[index]);
        }
    }

    bool acquire(size_t count, std::vector<Transport::Slice*>& slices) {
        if (count > free_.size()) return false;
        slices.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            slices.push_back(free_.back());
            free_.pop_back();
        }
        return true;
    }

    void release(std::vector<Transport::Slice*>& slices) {
        for (auto* slice : slices) {
            slice->peer_nic_path.clear();
            slice->source_location.clear();
            slice->dest_rkeys.clear();
            slice->cleanup_callback = nullptr;
            slice->completion_callback = nullptr;
            slice->completion_context = nullptr;
            slice->task = nullptr;
            free_.push_back(slice);
        }
        slices.clear();
    }

   private:
    std::unique_ptr<Transport::Slice[]> storage_;
    std::vector<Transport::Slice*> free_;
};

class RdmaRelayTransfer final : public RelayTransfer {
   public:
    RdmaRelayTransfer(std::shared_ptr<RelaySlicePool> pool,
                      std::vector<Transport::Slice*> slices)
        : pool_(std::move(pool)),
          slices_(std::move(slices)),
          remaining_(slices_.size()) {}

    ~RdmaRelayTransfer() override {
        if (!submitted_ && !slices_.empty()) pool_->release(slices_);
    }

    std::vector<Transport::Slice*>& slices() { return slices_; }

    void submitted() { submitted_ = true; }

    static void completeSlice(Transport::Slice* slice, bool success) {
        auto* transfer =
            static_cast<RdmaRelayTransfer*>(slice->completion_context);
        if (!success) transfer->failed_.store(true, std::memory_order_relaxed);
        transfer->remaining_.fetch_sub(1, std::memory_order_release);
    }

    Status poll(bool& completed, int32_t& completion_status) override {
        completed = false;
        if (remaining_.load(std::memory_order_acquire) != 0) {
            return Status::OK();
        }
        completion_status =
            failed_.load(std::memory_order_relaxed) ? ERR_CONTEXT : 0;
        pool_->release(slices_);
        completed = true;
        return Status::OK();
    }

    void abandon() override {}

   private:
    std::shared_ptr<RelaySlicePool> pool_;
    std::vector<Transport::Slice*> slices_;
    std::atomic<size_t> remaining_;
    std::atomic<bool> failed_{false};
    bool submitted_ = false;
};

class RdmaRelayBackend final : public RelayBackend {
   public:
    static Status Create(RdmaTransport& transport, uintptr_t arena_address,
                         size_t max_inflight,
                         std::unique_ptr<RelayBackend>& backend) {
        auto candidate = std::unique_ptr<RdmaRelayBackend>(
            new RdmaRelayBackend(transport, arena_address));
        auto status = candidate->initialize(max_inflight);
        if (!status.ok()) return status;
        backend = std::move(candidate);
        return Status::OK();
    }

    Status submit(const RelaySubmission& submission,
                  std::unique_ptr<RelayTransfer>& transfer) override {
        size_t slice_count = 0;
        auto status = forEachSlice(
            submission, [&](uintptr_t, uint64_t, size_t, const TargetRoute&,
                            Transport::SegmentID) { ++slice_count; });
        if (!status.ok()) return status;

        std::vector<Transport::Slice*> slices;
        if (!pool_->acquire(slice_count, slices)) {
            return Status::BatchBusy("PXN relay slice pool is exhausted");
        }
        auto candidate =
            std::make_unique<RdmaRelayTransfer>(pool_, std::move(slices));
        size_t slice_index = 0;
        status = forEachSlice(
            submission,
            [&](uintptr_t source, uint64_t destination, size_t length,
                const TargetRoute& route, Transport::SegmentID target_id) {
                auto* slice = candidate->slices()[slice_index++];
                slice->source_addr = reinterpret_cast<void*>(source);
                slice->length = length;
                slice->opcode = Transport::TransferRequest::WRITE;
                slice->target_id = target_id;
                slice->peer_nic_path = route.peer_nic_path;
                slice->source_location.clear();
                slice->status = Transport::Slice::PENDING;
                slice->task = nullptr;
                slice->dest_rkeys.clear();
                slice->cleanup_callback = nullptr;
                slice->completion_callback = &RdmaRelayTransfer::completeSlice;
                slice->completion_context = candidate.get();
                slice->rdma.dest_addr = destination;
                slice->rdma.source_lkey = source_lkey_;
                slice->rdma.dest_rkey = route.rkey;
                slice->rdma.lkey_index = source_device_id_;
                slice->rdma.rkey_index = route.device_id;
                slice->rdma.qp_depth = nullptr;
                slice->rdma.retry_cnt = 0;
                slice->rdma.max_retry_cnt = globalConfig().retry_cnt;
                slice->rdma.endpoint = nullptr;
                slice->ts = 0;
            });
        if (!status.ok()) return status;

        const int result =
            source_context_->submitPreparedPostSend(candidate->slices());
        if (result != 0) {
            return Status::Context("PXN relay RDMA submission failed");
        }
        candidate->submitted();
        transfer = std::move(candidate);
        return Status::OK();
    }

   private:
    using MrKey = Transport::Slice::mr_key_t;

    struct TargetRoute {
        uint64_t begin;
        uint64_t end;
        MrKey rkey;
        int device_id;
        std::string peer_nic_path;
    };

    struct SessionRoutes {
        Transport::SegmentID target_id;
        std::shared_ptr<RdmaTransport::SegmentDesc> descriptor;
        std::vector<TargetRoute> routes;
    };

    RdmaRelayBackend(RdmaTransport& transport, uintptr_t arena_address)
        : transport_(transport), arena_address_(arena_address) {}

    Status initialize(size_t max_inflight) {
        auto local = transport_.meta()->getSegmentDescByID(LOCAL_SEGMENT_ID);
        int buffer_id = -1;
        if (RdmaTransport::selectDevice(local.get(), arena_address_,
                                        kRequiredArenaSize, buffer_id,
                                        source_device_id_) != 0) {
            return Status::AddressNotRegistered(
                "PXN staging arena is not registered by one RDMA context");
        }
        const auto& contexts = transport_.getContextList();
        if (source_device_id_ < 0 ||
            static_cast<size_t>(source_device_id_) >= contexts.size() ||
            contexts[source_device_id_] == nullptr ||
            !contexts[source_device_id_]->active()) {
            return Status::DeviceNotFound(
                "PXN staging arena RDMA context is unavailable");
        }
        const auto& buffer = local->buffers[buffer_id];
        if (static_cast<size_t>(source_device_id_) >= buffer.lkey.size()) {
            return Status::AddressNotRegistered(
                "PXN staging arena lkey is unavailable");
        }
        source_context_ = contexts[source_device_id_].get();
        source_device_name_ = source_context_->deviceName();
        source_lkey_ = buffer.lkey[source_device_id_];
        slice_size_ = globalConfig().slice_size;
        if (slice_size_ == 0) {
            return Status::InvalidArgument("RDMA slice size must be nonzero");
        }
        const size_t slices_per_piece =
            kMaxPlanCount + (kSlotSize + slice_size_ - 1) / slice_size_;
        pool_ =
            std::make_shared<RelaySlicePool>(max_inflight * slices_per_piece);
        return Status::OK();
    }

    Status resolveRoute(std::string_view session, uint64_t destination,
                        SessionRoutes*& session_routes,
                        const TargetRoute*& route) {
        auto it = sessions_.find(std::string(session));
        if (it == sessions_.end()) {
            const auto target_id =
                transport_.getSegmentID(std::string(session));
            if (target_id == static_cast<Transport::SegmentID>(-1)) {
                return Status::Metadata("PXN target session was not found");
            }
            auto descriptor = transport_.meta()->getSegmentDescByID(target_id);
            if (descriptor == nullptr) {
                return Status::Metadata(
                    "PXN target segment description was not found");
            }
            it = sessions_
                     .emplace(
                         std::string(session),
                         SessionRoutes{target_id, std::move(descriptor), {}})
                     .first;
        }
        session_routes = &it->second;
        for (const auto& cached : session_routes->routes) {
            if (destination >= cached.begin && destination < cached.end) {
                route = &cached;
                return Status::OK();
            }
        }

        int buffer_id = -1;
        int device_id = -1;
        int result = 0;
        if (globalConfig().enable_hca_peer_affinity) {
            result = RdmaTransport::selectDeviceByLocalHca(
                session_routes->descriptor.get(), destination, 1,
                source_device_name_, buffer_id, device_id);
        } else {
            const auto hint = globalConfig().enable_dest_device_affinity
                                  ? std::string_view(source_device_name_)
                                  : std::string_view();
            result = RdmaTransport::selectDevice(
                session_routes->descriptor.get(), destination, 1, hint,
                buffer_id, device_id);
        }
        if (result != 0) {
            return Status::AddressNotRegistered(
                "PXN target address is not registered");
        }
        const auto& descriptor = *session_routes->descriptor;
        const auto& buffer = descriptor.buffers[buffer_id];
        if (device_id < 0 ||
            static_cast<size_t>(device_id) >= descriptor.devices.size() ||
            static_cast<size_t>(device_id) >= buffer.rkey.size()) {
            return Status::AddressNotRegistered(
                "PXN target RDMA route is incomplete");
        }
        session_routes->routes.push_back(
            {buffer.addr, buffer.addr + buffer.length, buffer.rkey[device_id],
             device_id,
             MakeNicPath(descriptor.nicPathServerName(),
                         descriptor.devices[device_id].name)});
        route = &session_routes->routes.back();
        return Status::OK();
    }

    template <typename Callback>
    Status forEachSlice(const RelaySubmission& submission,
                        Callback&& callback) {
        uintptr_t source = submission.source;
        for (const auto& plan : submission.plans) {
            uint64_t destination = plan.final_destination;
            uint64_t remaining = plan.length;
            while (remaining != 0) {
                SessionRoutes* session_routes = nullptr;
                const TargetRoute* route = nullptr;
                auto status = resolveRoute(submission.session, destination,
                                           session_routes, route);
                if (!status.ok()) return status;
                const size_t length = static_cast<size_t>(std::min<uint64_t>(
                    {remaining, slice_size_, route->end - destination}));
                callback(source, destination, length, *route,
                         session_routes->target_id);
                source += length;
                destination += length;
                remaining -= length;
            }
        }
        return Status::OK();
    }

    RdmaTransport& transport_;
    uintptr_t arena_address_;
    RdmaContext* source_context_ = nullptr;
    std::string source_device_name_;
    MrKey source_lkey_ = 0;
    int source_device_id_ = -1;
    size_t slice_size_ = 0;
    std::shared_ptr<RelaySlicePool> pool_;
    std::unordered_map<std::string, SessionRoutes> sessions_;
};

}  // namespace

RelayPipeline::RelayPipeline(ControlBlock* control, uintptr_t arena_address,
                             uint64_t epoch, size_t max_inflight,
                             std::unique_ptr<RelayBackend> backend)
    : control_(control),
      arena_address_(arena_address),
      epoch_(epoch),
      max_inflight_(max_inflight),
      backend_(std::move(backend)) {
    next_sequence_.fill(1);
}

RelayPipeline::~RelayPipeline() { shutdown(); }

Status RelayPipeline::reapInbound(bool& made_progress) {
    made_progress = false;
    Status first_error;
    for (size_t lane_index = 0; lane_index < kLaneCount; ++lane_index) {
        auto& queue = inflight_[lane_index];
        if (queue.empty()) continue;

        bool completed = false;
        int32_t completion_status = 0;
        auto status =
            queue.front().transfer->poll(completed, completion_status);
        if (!status.ok()) {
            queue.front().transfer->abandon();
            (void)queue.front().transfer.release();
            completed = true;
            completion_status = ERR_CONTEXT;
            first_error = status;
        }
        if (!completed) continue;

        complete(lane_index, queue.front().sequence, completion_status);
        queue.pop_front();
        --inflight_count_;
        made_progress = true;
    }
    return first_error;
}

Status RelayPipeline::progressInbound(bool& made_progress) {
    made_progress = false;
    if (inflight_count_ >= max_inflight_) return Status::OK();

    for (size_t offset = 0; offset < kLaneCount; ++offset) {
        const size_t lane_index = (next_lane_ + offset) % kLaneCount;
        auto& lane = control_->lanes[lane_index];
        if (loadLaneState(lane.header) !=
            static_cast<uint32_t>(LaneState::kReady)) {
            continue;
        }

        const uint64_t sequence = next_sequence_[lane_index];
        const uint64_t doorbell =
            __atomic_load_n(&lane.header.doorbell, __ATOMIC_ACQUIRE);
        if (sequence == 0 || doorbell < sequence) continue;

        const size_t slot = *slotIndex(sequence);
        const auto& descriptor = lane.descriptors[slot];
        const auto descriptor_status =
            validateDescriptor(descriptor, sequence, epoch_);
        if (descriptor_status != DescriptorError::kOk) {
            inflight_[lane_index].push_back(
                {sequence, std::make_unique<CompletedRelayTransfer>(
                               ERR_INVALID_ARGUMENT)});
            ++inflight_count_;
            (void)advanceSequence(sequence, next_sequence_[lane_index]);
            next_lane_ = (lane_index + 1) % kLaneCount;
            made_progress = true;
            return Status::OK();
        }

        RelaySubmission submission;
        submission.source =
            arena_address_ + (lane_index * kSlotsPerLane + slot) * kSlotSize;
        submission.session.assign(descriptor.session,
                                  descriptor.session_length);
        submission.plans.assign(descriptor.plans,
                                descriptor.plans + descriptor.plan_count);

        std::unique_ptr<RelayTransfer> transfer;
        auto status = backend_->submit(submission, transfer);
        if (!status.ok()) {
            transfer = std::make_unique<CompletedRelayTransfer>(ERR_CONTEXT);
        }
        inflight_[lane_index].push_back({sequence, std::move(transfer)});
        ++inflight_count_;
        (void)advanceSequence(sequence, next_sequence_[lane_index]);
        next_lane_ = (lane_index + 1) % kLaneCount;
        made_progress = true;
        return status;
    }
    return Status::OK();
}

void RelayPipeline::shutdown() {
    for (auto& queue : inflight_) {
        for (auto& item : queue) {
            item.transfer->abandon();
            (void)item.transfer.release();
        }
        queue.clear();
    }
    inflight_count_ = 0;
}

void RelayPipeline::complete(size_t lane_index, uint64_t sequence,
                             int32_t status) {
    auto& lane = control_->lanes[lane_index];
    const size_t slot = *slotIndex(sequence);
    (void)commitCompletion(lane.completions[slot], sequence, status);
    __atomic_store_n(&lane.header.completed, sequence, __ATOMIC_RELEASE);
}

PxnPump::PxnPump(SenderPipeline& sender, RelayPipeline& relay)
    : sender_(sender), relay_(relay), thread_(&PxnPump::run, this) {}

PxnPump::~PxnPump() { shutdown(); }

void PxnPump::wake() { condition_.notify_one(); }

void PxnPump::shutdown() {
    if (!running_.exchange(false)) return;
    condition_.notify_one();
    if (thread_.joinable()) thread_.join();
}

void PxnPump::run() {
    while (running_.load(std::memory_order_acquire)) {
        bool made_progress = false;
        bool progress = false;
        (void)sender_.reapOutbound(progress);
        made_progress = made_progress || progress;
        (void)relay_.reapInbound(progress);
        made_progress = made_progress || progress;
        (void)relay_.progressInbound(progress);
        made_progress = made_progress || progress;
        (void)sender_.progressOutbound(progress);
        made_progress = made_progress || progress;

        if (!made_progress) {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait_for(lock, std::chrono::microseconds(50));
        }
    }
}

Status makeRdmaRelayBackend(RdmaTransport& transport, uintptr_t arena_address,
                            size_t max_inflight,
                            std::unique_ptr<RelayBackend>& backend) {
    return RdmaRelayBackend::Create(transport, arena_address, max_inflight,
                                    backend);
}

}  // namespace pxn
}  // namespace mooncake

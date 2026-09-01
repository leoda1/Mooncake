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

#include "transport/pxn_transport/pxn_transport.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "common.h"
#include "config.h"
#include "error.h"
#include "transport/pxn_transport/pxn_nvtx.h"
#include "transport/rdma_transport/rdma_context.h"
#include "transport/rdma_transport/rdma_transport.h"

namespace mooncake {
namespace pxn {

Status RelayBackend::submitBatch(std::span<RelayBatchItem> items) {
    Status first_error;
    for (auto& item : items) {
        item.status = submit(item.submission, item.transfer);
        if (first_error.ok() && !item.status.ok()) first_error = item.status;
    }
    return first_error;
}

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
                         std::span<const std::string> source_device_names,
                         std::span<const std::string> source_rails,
                         size_t max_inflight, PxnGeometry geometry,
                         std::unique_ptr<RelayBackend>& backend) {
        auto candidate = std::unique_ptr<RdmaRelayBackend>(
            new RdmaRelayBackend(transport, arena_address, geometry));
        auto status = candidate->initialize(source_device_names, source_rails,
                                            max_inflight);
        if (!status.ok()) return status;
        backend = std::move(candidate);
        return Status::OK();
    }

    Status submit(const RelaySubmission& submission,
                  std::unique_ptr<RelayTransfer>& transfer) override {
        RelayBatchItem item;
        item.submission = submission;
        auto status = submitBatch(std::span<RelayBatchItem>(&item, 1));
        transfer = std::move(item.transfer);
        return item.status.ok() ? status : item.status;
    }

    Status submitBatch(std::span<RelayBatchItem> items) override {
        PXN_NVTX_RELAY("pxn::relay::submit");
        Status first_error;
        for (auto& transfers : prepared_) transfers.clear();
        for (auto& slices : prepared_slices_) slices.clear();

        for (auto& item : items) {
            item.status = prepare(item.submission, item.transfer);
            if (!item.status.ok()) {
                if (first_error.ok()) first_error = item.status;
                continue;
            }
            prepared_[item.submission.rail_index].push_back(
                static_cast<RdmaRelayTransfer*>(item.transfer.get()));
        }

        for (size_t rail_index = 0; rail_index < prepared_.size();
             ++rail_index) {
            auto& transfers = prepared_[rail_index];
            if (transfers.empty()) continue;
            auto& slices = prepared_slices_[rail_index];
            size_t slice_count = 0;
            for (auto* transfer : transfers) {
                slice_count += transfer->slices().size();
            }
            slices.reserve(slice_count);
            for (auto* transfer : transfers) {
                slices.insert(slices.end(), transfer->slices().begin(),
                              transfer->slices().end());
            }
            const int result =
                source_rails_[rail_index].context->submitPreparedPostSend(
                    slices);
            if (result == 0) {
                for (auto* transfer : transfers) transfer->submitted();
                continue;
            }
            auto failure =
                Status::Context("PXN relay RDMA batch submission failed");
            if (first_error.ok()) first_error = failure;
            for (auto& item : items) {
                if (item.status.ok() &&
                    item.submission.rail_index == rail_index) {
                    item.status = failure;
                    item.transfer.reset();
                }
            }
        }
        return first_error;
    }

   private:
    using MrKey = Transport::Slice::mr_key_t;

    struct SourceRail {
        std::string device_name;
        std::string rail;
        RdmaContext* context;
        MrKey lkey;
        int device_id;
    };

    struct TargetRoute {
        uint32_t source_rail_index;
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

    RdmaRelayBackend(RdmaTransport& transport, uintptr_t arena_address,
                     PxnGeometry geometry)
        : transport_(transport),
          arena_address_(arena_address),
          geometry_(geometry),
          rail_resolver_(globalConfig().pxn_rail_map) {}

    Status prepare(const RelaySubmission& submission,
                   std::unique_ptr<RelayTransfer>& transfer) {
        if (submission.rail_index >= source_rails_.size()) {
            return Status::InvalidArgument("PXN relay rail index is invalid");
        }
        const auto& source_rail = source_rails_[submission.rail_index];
        VLOG(1) << "PXN relay uses rail[" << submission.rail_index << "] "
                << source_rail.device_name << " for " << submission.session;
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
                slice->rdma.source_lkey = source_rail.lkey;
                slice->rdma.dest_rkey = route.rkey;
                slice->rdma.lkey_index = source_rail.device_id;
                slice->rdma.rkey_index = route.device_id;
                slice->rdma.qp_depth = nullptr;
                slice->rdma.retry_cnt = 0;
                slice->rdma.max_retry_cnt = globalConfig().retry_cnt;
                slice->rdma.endpoint = nullptr;
                slice->ts = 0;
            });
        if (!status.ok()) return status;
        transfer = std::move(candidate);
        return Status::OK();
    }

    Status initialize(std::span<const std::string> source_device_names,
                      std::span<const std::string> source_rails,
                      size_t max_inflight) {
        if (source_device_names.empty() ||
            source_device_names.size() != source_rails.size() ||
            source_device_names.size() > kMaxRailsPerRank ||
            !geometry_.valid()) {
            return Status::InvalidArgument(
                "PXN relay source rails are invalid");
        }
        auto local = transport_.meta()->getSegmentDescByID(LOCAL_SEGMENT_ID);
        const size_t arena_size = geometry_.arenaSize();
        const auto& contexts = transport_.getContextList();
        for (size_t index = 0; index < source_device_names.size(); ++index) {
            const auto& source_device_name = source_device_names[index];
            if (source_device_name.empty() || source_rails[index].empty()) {
                return Status::InvalidArgument(
                    "PXN relay source rail must not be empty");
            }
            int buffer_id = -1;
            int device_id = -1;
            if (RdmaTransport::selectDevice(local.get(), arena_address_,
                                            arena_size, source_device_name,
                                            buffer_id, device_id) != 0) {
                return Status::AddressNotRegistered(
                    "PXN staging arena is not registered by " +
                    source_device_name);
            }
            if (device_id < 0 ||
                static_cast<size_t>(device_id) >= contexts.size() ||
                contexts[device_id] == nullptr ||
                !contexts[device_id]->active() ||
                contexts[device_id]->deviceName() != source_device_name) {
                return Status::DeviceNotFound(
                    "PXN staging arena RDMA context mismatch for " +
                    source_device_name);
            }
            const auto& buffer = local->buffers[buffer_id];
            if (static_cast<size_t>(device_id) >= buffer.lkey.size()) {
                return Status::AddressNotRegistered(
                    "PXN staging arena lkey is unavailable for " +
                    source_device_name);
            }
            source_rails_.push_back({source_device_name, source_rails[index],
                                     contexts[device_id].get(),
                                     buffer.lkey[device_id], device_id});
            LOG(INFO) << "PXN relay source rail[" << index << "] is "
                      << source_device_name << " (canonical "
                      << source_rails[index] << ")";
        }
        slice_size_ = globalConfig().slice_size;
        if (slice_size_ == 0) {
            return Status::InvalidArgument("RDMA slice size must be nonzero");
        }
        const size_t slot_size = geometry_.slot_size;
        const size_t slot_slices =
            slot_size / slice_size_ + (slot_size % slice_size_ != 0);
        if (slot_slices > std::numeric_limits<size_t>::max() - kMaxPlanCount ||
            max_inflight > std::numeric_limits<size_t>::max() /
                               (kMaxPlanCount + slot_slices)) {
            return Status::InvalidArgument(
                "PXN relay slice pool capacity overflows");
        }
        const size_t slices_per_piece = kMaxPlanCount + slot_slices;
        pool_ =
            std::make_shared<RelaySlicePool>(max_inflight * slices_per_piece);
        prepared_.resize(source_rails_.size());
        prepared_slices_.resize(source_rails_.size());
        return Status::OK();
    }

    Status resolveRoute(uint32_t source_rail_index, std::string_view session,
                        uint64_t destination, SessionRoutes*& session_routes,
                        const TargetRoute*& route) {
        if (source_rail_index >= source_rails_.size()) {
            return Status::InvalidArgument("PXN relay rail index is invalid");
        }
        const auto& source_rail = source_rails_[source_rail_index];
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
            if (cached.source_rail_index == source_rail_index &&
                destination >= cached.begin && destination < cached.end) {
                route = &cached;
                return Status::OK();
            }
        }

        int buffer_id = -1;
        int device_id = -1;
        auto matches_source_rail = [&]() {
            const auto& descriptor = *session_routes->descriptor;
            return device_id >= 0 &&
                   static_cast<size_t>(device_id) < descriptor.devices.size() &&
                   rail_resolver_.canonicalize(
                       descriptor.devices[device_id].name) == source_rail.rail;
        };
        int result = RdmaTransport::selectDevice(
            session_routes->descriptor.get(), destination, 1,
            source_rail.device_name, buffer_id, device_id);
        if (result != 0 || !matches_source_rail()) {
            result = -1;
            const size_t retry_count =
                session_routes->descriptor->devices.size() + 1;
            for (size_t retry = 0; retry < retry_count; ++retry) {
                if (RdmaTransport::selectDevice(
                        session_routes->descriptor.get(), destination, 1,
                        buffer_id, device_id, static_cast<int>(retry)) == 0 &&
                    matches_source_rail()) {
                    result = 0;
                    break;
                }
            }
        }
        if (result != 0 || !matches_source_rail()) {
            return Status::AddressNotRegistered(
                "PXN target address is not registered on rail " +
                source_rail.rail);
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
            {source_rail_index, buffer.addr, buffer.addr + buffer.length,
             buffer.rkey[device_id], device_id,
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
                auto status =
                    resolveRoute(submission.rail_index, submission.session,
                                 destination, session_routes, route);
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
    PxnGeometry geometry_;
    RailResolver rail_resolver_;
    std::vector<SourceRail> source_rails_;
    size_t slice_size_ = 0;
    std::shared_ptr<RelaySlicePool> pool_;
    std::vector<std::vector<RdmaRelayTransfer*>> prepared_;
    std::vector<std::vector<Transport::Slice*>> prepared_slices_;
    std::unordered_map<std::string, SessionRoutes> sessions_;
};

}  // namespace

RelayPipeline::RelayPipeline(ControlBlock* control, uintptr_t arena_address,
                             uint64_t epoch, size_t max_inflight,
                             std::unique_ptr<RelayBackend> backend,
                             PxnGeometry geometry)
    : control_(control),
      arena_address_(arena_address),
      epoch_(epoch),
      max_inflight_(max_inflight),
      geometry_(geometry),
      backend_(std::move(backend)) {
    next_sequence_.fill(1);
}

RelayPipeline::~RelayPipeline() { shutdown(); }

Status RelayPipeline::reapInbound(bool& made_progress) {
    made_progress = false;
    Status first_error;
    for (size_t lane_index = 0; lane_index < geometry_.lane_count;
         ++lane_index) {
        auto& queue = inflight_[lane_index];
        while (!queue.empty()) {
            bool completed = false;
            int32_t completion_status = 0;
            auto status =
                queue.front().transfer->poll(completed, completion_status);
            if (!status.ok()) {
                queue.front().transfer->abandon();
                (void)queue.front().transfer.release();
                completed = true;
                completion_status = ERR_CONTEXT;
                if (first_error.ok()) first_error = status;
            }
            if (!completed) break;

            complete(lane_index, queue.front().sequence, completion_status);
            queue.pop_front();
            --inflight_count_;
            made_progress = true;
        }
    }
    return first_error;
}

Status RelayPipeline::progressInbound(bool& made_progress) {
    made_progress = false;
    Status first_error;
    pending_batch_.clear();
    pending_batch_.reserve(max_inflight_ -
                           std::min(max_inflight_, inflight_count_));
    while (inflight_count_ + pending_batch_.size() < max_inflight_) {
        bool progress = false;
        auto status = collectInboundOne(pending_batch_, progress);
        if (first_error.ok() && !status.ok()) first_error = status;
        made_progress = made_progress || progress;
        if (!status.ok() || !progress) break;
    }
    if (inflight_count_ + pending_batch_.size() >= max_inflight_) {
        for (size_t lane_index = 0; lane_index < geometry_.lane_count;
             ++lane_index) {
            const auto& lane = control_->lanes[lane_index];
            if (loadLaneState(lane.header) !=
                static_cast<uint32_t>(LaneState::kReady)) {
                continue;
            }
            const uint64_t sequence = next_sequence_[lane_index];
            const uint64_t doorbell =
                __atomic_load_n(&lane.header.doorbell, __ATOMIC_ACQUIRE);
            if (sequence == 0 || doorbell < sequence) continue;
            if (last_limit_blocked_sequence_[lane_index] == sequence) continue;
            last_limit_blocked_sequence_[lane_index] = sequence;
            ++limit_blocked_pieces_;
        }
    }

    if (pending_batch_.empty()) return first_error;

    backend_batch_.clear();
    backend_batch_.reserve(pending_batch_.size());
    for (auto& entry : pending_batch_) {
        backend_batch_.push_back(std::move(entry.item));
    }
    ++submit_batches_;
    submitted_batch_slots_ += backend_batch_.size();
    submit_batch_high_watermark_ =
        std::max(submit_batch_high_watermark_, backend_batch_.size());
    auto status = backend_->submitBatch(backend_batch_);
    if (first_error.ok() && !status.ok()) first_error = status;

    for (size_t index = 0; index < pending_batch_.size(); ++index) {
        auto& entry = pending_batch_[index];
        auto& item = backend_batch_[index];
        if (item.status.ok() && item.transfer == nullptr) {
            item.status =
                Status::Context("PXN relay backend returned no transfer");
        }
        if (!item.status.ok() || item.transfer == nullptr) {
            if (first_error.ok() && !item.status.ok()) {
                first_error = item.status;
            }
            item.transfer =
                std::make_unique<CompletedRelayTransfer>(ERR_CONTEXT);
        } else {
            submitted_bytes_.fetch_add(entry.piece_length,
                                       std::memory_order_relaxed);
            submitted_pieces_.fetch_add(1, std::memory_order_relaxed);
        }
        enqueueInflight(entry.lane_index, entry.sequence,
                        std::move(item.transfer));
    }
    return first_error;
}

Status RelayPipeline::collectInboundOne(std::vector<PendingInbound>& pending,
                                        bool& made_progress) {
    made_progress = false;

    for (size_t offset = 0; offset < geometry_.lane_count; ++offset) {
        const size_t lane_index = (next_lane_ + offset) % geometry_.lane_count;
        auto& lane = control_->lanes[lane_index];
        if (loadLaneState(lane.header) !=
            static_cast<uint32_t>(LaneState::kReady)) {
            continue;
        }

        const uint64_t sender_epoch =
            __atomic_load_n(&lane.header.sender_epoch, __ATOMIC_ACQUIRE);
        if (lane_sender_epoch_[lane_index] != sender_epoch) {
            const bool lane_has_pending =
                std::any_of(pending.begin(), pending.end(),
                            [lane_index](const PendingInbound& entry) {
                                return entry.lane_index == lane_index;
                            });
            if (!inflight_[lane_index].empty() || lane_has_pending) {
                continue;
            }
            lane_sender_epoch_[lane_index] = sender_epoch;
            next_sequence_[lane_index] = 1;
        }

        const uint64_t sequence = next_sequence_[lane_index];
        const uint64_t doorbell =
            __atomic_load_n(&lane.header.doorbell, __ATOMIC_ACQUIRE);
        if (sequence == 0 || doorbell < sequence) continue;

        const size_t slot = *slotIndex(sequence, geometry_.slots_per_lane);
        const auto& descriptor = lane.descriptors[slot];
        const auto descriptor_status = validateDescriptor(
            descriptor, sequence, epoch_, geometry_.slot_size);
        if (descriptor_status != DescriptorError::kOk) {
            enqueueInflight(
                lane_index, sequence,
                std::make_unique<CompletedRelayTransfer>(ERR_INVALID_ARGUMENT));
            (void)advanceSequence(sequence, next_sequence_[lane_index]);
            next_lane_ = (lane_index + 1) % geometry_.lane_count;
            made_progress = true;
            return Status::OK();
        }

        RelaySubmission submission;
        submission.source =
            arena_address_ + (lane_index * geometry_.slots_per_lane + slot) *
                                 geometry_.slot_size;
        submission.session.assign(descriptor.session,
                                  descriptor.session_length);
        submission.rail_index = descriptor.rail_index;
        submission.plans.assign(descriptor.plans,
                                descriptor.plans + descriptor.plan_count);
        ready_bytes_.fetch_add(descriptor.piece_length,
                               std::memory_order_relaxed);
        ready_pieces_.fetch_add(1, std::memory_order_relaxed);

        RelayBatchItem item;
        item.submission = std::move(submission);
        pending.push_back(
            {lane_index, sequence, descriptor.piece_length, std::move(item)});
        (void)advanceSequence(sequence, next_sequence_[lane_index]);
        next_lane_ = (lane_index + 1) % geometry_.lane_count;
        made_progress = true;
        return Status::OK();
    }
    return Status::OK();
}

void RelayPipeline::enqueueInflight(size_t lane_index, uint64_t sequence,
                                    std::unique_ptr<RelayTransfer> transfer) {
    inflight_[lane_index].push_back({sequence, std::move(transfer)});
    ++inflight_count_;
    inflight_high_watermark_ =
        std::max(inflight_high_watermark_, inflight_count_);
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

RelayPipeline::InflightStats RelayPipeline::inflightStats() const {
    return {inflight_count_,
            max_inflight_,
            std::max(inflight_high_watermark_, inflight_count_),
            limit_blocked_pieces_,
            submit_batches_,
            submitted_batch_slots_,
            submit_batch_high_watermark_};
}

std::array<RelayPipeline::LaneSlotStats, kMaxLaneCount>
RelayPipeline::laneSlotStats() const {
    std::array<LaneSlotStats, kMaxLaneCount> result;
    for (size_t index = 0; index < geometry_.lane_count; ++index) {
        const auto& header = control_->lanes[index].header;
        auto& stats = result[index];
        stats.active =
            loadLaneState(header) == static_cast<uint32_t>(LaneState::kReady);
        if (!stats.active) continue;

        const uint64_t completed =
            __atomic_load_n(&header.completed, __ATOMIC_ACQUIRE);
        uint64_t doorbell = __atomic_load_n(&header.doorbell, __ATOMIC_ACQUIRE);
        uint64_t head = __atomic_load_n(&header.head, __ATOMIC_ACQUIRE);
        // Clamp against torn/lagging reads so the subtraction stays ordered:
        // completed <= doorbell <= head.
        doorbell = std::max(doorbell, completed);
        head = std::max(head, doorbell);

        const uint64_t occupied =
            std::min<uint64_t>(head - completed, geometry_.slots_per_lane);
        stats.empty = geometry_.slots_per_lane - occupied;
        stats.cuda_pending = static_cast<size_t>(head - doorbell);
        stats.rdma_pending = static_cast<size_t>(doorbell - completed);
    }
    return result;
}

void RelayPipeline::complete(size_t lane_index, uint64_t sequence,
                             int32_t status) {
    auto& lane = control_->lanes[lane_index];
    const size_t slot = *slotIndex(sequence, geometry_.slots_per_lane);
    if (status == 0) {
        relayed_bytes_.fetch_add(lane.descriptors[slot].piece_length,
                                 std::memory_order_relaxed);
        relayed_pieces_.fetch_add(1, std::memory_order_relaxed);
    }
    (void)commitCompletion(lane.completions[slot], sequence, status);
    __atomic_store_n(&lane.header.completed, sequence, __ATOMIC_RELEASE);
}

PxnPump::PxnPump(SenderPipeline& sender, RelayPipeline& relay)
    : sender_(sender), relay_(relay) {
    sender_thread_ = std::thread(&PxnPump::runSender, this);
    relay_thread_ = std::thread(&PxnPump::runRelay, this);
}

PxnPump::~PxnPump() { shutdown(); }

void PxnPump::wake() { sender_condition_.notify_one(); }

void PxnPump::shutdown() {
    if (!running_.exchange(false)) return;
    sender_condition_.notify_one();
    relay_condition_.notify_one();
    if (sender_thread_.joinable()) sender_thread_.join();
    if (relay_thread_.joinable()) relay_thread_.join();
}

void PxnPump::runSender() {
    PXN_NVTX_NAME_THREAD("pxn-sender");
    auto bind_status = sender_.bindThread();
    if (!bind_status.ok()) {
        LOG(ERROR) << "Failed to bind PXN sender thread: "
                   << bind_status.ToString();
    }
    while (running_.load(std::memory_order_acquire)) {
        bool made_progress = false;
        bool progress = false;
        {
            PXN_NVTX_SENDER("pxn::sender::reap");
            (void)sender_.reapOutbound(progress);
        }
        made_progress = made_progress || progress;
        {
            PXN_NVTX_SENDER("pxn::sender::progress");
            (void)sender_.progressOutbound(progress);
        }
        made_progress = made_progress || progress;

        if (!made_progress && !sender_.hasInflight()) {
            std::unique_lock<std::mutex> lock(sender_mutex_);
            sender_condition_.wait_for(lock, std::chrono::microseconds(50));
        }
    }
}

void PxnPump::runRelay() {
    PXN_NVTX_NAME_THREAD("pxn-relay");
    while (running_.load(std::memory_order_acquire)) {
        bool made_progress = false;
        bool progress = false;
        {
            PXN_NVTX_RELAY("pxn::relay::reap");
            (void)relay_.reapInbound(progress);
        }
        made_progress = made_progress || progress;
        {
            PXN_NVTX_RELAY("pxn::relay::progress");
            (void)relay_.progressInbound(progress);
        }
        made_progress = made_progress || progress;

        if (!made_progress && !relay_.hasInflight()) {
            std::unique_lock<std::mutex> lock(relay_mutex_);
            relay_condition_.wait_for(lock, std::chrono::microseconds(50));
        }
    }
}

Status makeRdmaRelayBackend(RdmaTransport& transport, uintptr_t arena_address,
                            std::span<const std::string> source_device_names,
                            std::span<const std::string> source_rails,
                            size_t max_inflight, PxnGeometry geometry,
                            std::unique_ptr<RelayBackend>& backend) {
    return RdmaRelayBackend::Create(transport, arena_address,
                                    source_device_names, source_rails,
                                    max_inflight, geometry, backend);
}

}  // namespace pxn
}  // namespace mooncake

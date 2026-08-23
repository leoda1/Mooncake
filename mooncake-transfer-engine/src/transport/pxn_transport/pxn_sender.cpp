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
#include <limits>
#include <utility>

#include "transport/rdma_transport/rdma_transport.h"

namespace mooncake {
namespace pxn {
namespace {

void markPosted(const std::vector<LogicalSlice>& slices) {
    for (const auto& slice : slices) slice.mark_posted(slice.context);
}

void markSuccess(const std::vector<LogicalSlice>& slices) {
    for (const auto& slice : slices) slice.mark_success(slice.context);
}

void markFailed(const std::vector<LogicalSlice>& slices) {
    for (const auto& slice : slices) slice.mark_failed(slice.context);
}

Status validateSubmission(const SenderSubmission& submission) {
    if (submission.slices.size() != submission.piece.spans.size()) {
        return Status::InvalidArgument("PXN logical slice count mismatch");
    }
    for (size_t index = 0; index < submission.slices.size(); ++index) {
        const auto& slice = submission.slices[index];
        if (slice.context == nullptr || slice.length == 0 ||
            slice.length != submission.piece.spans[index].length ||
            slice.mark_posted == nullptr || slice.mark_success == nullptr ||
            slice.mark_failed == nullptr) {
            return Status::InvalidArgument("invalid PXN logical slice");
        }
    }
    return Status::OK();
}

}  // namespace

SenderLane::SenderLane(SenderLaneEndpoint endpoint, SenderBackend& backend,
                       SenderFallback& fallback,
                       std::chrono::milliseconds credit_timeout,
                       std::unique_ptr<SenderLaneHandle> handle)
    : endpoint_(endpoint),
      backend_(backend),
      fallback_(fallback),
      credit_timeout_(credit_timeout),
      handle_(std::move(handle)) {}

SenderLane::~SenderLane() { (void)shutdown(); }

Status SenderLane::Create(SenderLaneEndpoint endpoint, SenderBackend& backend,
                          SenderFallback& fallback,
                          std::chrono::milliseconds credit_timeout,
                          std::unique_ptr<SenderLane>& lane) {
    if (endpoint.epoch == 0 || endpoint.arena_address == 0 ||
        endpoint.lane_index >= kLaneCount || endpoint.control == nullptr ||
        credit_timeout.count() <= 0) {
        return Status::InvalidArgument("invalid PXN sender lane endpoint");
    }
    std::unique_ptr<SenderLaneHandle> handle;
    auto status = backend.createLane(endpoint, handle);
    if (!status.ok()) return status;
    if (handle == nullptr) {
        return Status::InvalidArgument("missing PXN sender lane handle");
    }
    lane.reset(new SenderLane(endpoint, backend, fallback, credit_timeout,
                              std::move(handle)));
    return Status::OK();
}

Status SenderLane::enqueue(SenderSubmission submission) {
    auto status = validateSubmission(submission);
    if (!status.ok()) {
        failSubmission(submission);
        return status;
    }

    std::unique_ptr<SenderReadyFence> fence;
    auto ready_status = backend_.recordReady(fence);
    QueuedSubmission queued{std::move(submission), std::move(fence),
                            std::move(ready_status),
                            std::chrono::steady_clock::now()};
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (!accepting_.load()) {
        failSubmission(queued.submission);
        return Status::InvalidArgument("PXN sender lane is shutting down");
    }
    queue_.push_back(std::move(queued));
    return Status::OK();
}

Status SenderLane::reap(bool& made_progress) {
    made_progress = false;
    auto& lane = endpoint_.control->lanes[endpoint_.lane_index];
    while (!inflight_.empty()) {
        auto& published = inflight_.front();
        auto slot = slotIndex(published.sequence);
        if (!slot.has_value()) {
            return Status::InvalidArgument("invalid PXN published sequence");
        }
        const auto& completion = lane.completions[*slot];
        const uint64_t completion_sequence = loadCompletionSequence(completion);
        if (completion_sequence != published.sequence) break;
        int32_t completion_status = 0;
        if (!tryLoadCompletion(completion, published.sequence,
                               completion_status)) {
            completion_status = -1;
            quarantine_.trip();
        }
        if (completion_status == 0) {
            markSuccess(published.slices);
        } else {
            markFailed(published.slices);
        }
        reaped_sequence_ = published.sequence;
        inflight_.pop_front();
        made_progress = true;
    }

    bool fallback_progress = false;
    auto status = pollFallbacks(fallback_progress);
    made_progress = made_progress || fallback_progress;
    return status;
}

Status SenderLane::progress(bool& made_progress) {
    made_progress = false;
    Status first_error;
    bool fallback_progress = false;
    auto status = pollFallbacks(fallback_progress);
    if (!status.ok()) first_error = status;
    made_progress = fallback_progress;

    bool fallback_queue_progress = false;
    status = progressFallbackQueue(fallback_queue_progress);
    if (first_error.ok() && !status.ok()) first_error = status;
    made_progress = made_progress || fallback_queue_progress;
    if (!fallback_queue_.empty()) return first_error;

    if (pending_publication_ != nullptr) {
        bool ready = false;
        status = backend_.pollCpuDoorbell(*handle_, ready);
        if (!status.ok()) {
            quarantine_.trip();
            auto pending = std::move(*pending_publication_);
            pending_publication_.reset();
            pending_sequence_ = 0;
            auto fallback_status = deferOrStartFallback(std::move(pending));
            if (first_error.ok()) first_error = status;
            if (first_error.ok() && !fallback_status.ok()) {
                first_error = fallback_status;
            }
            made_progress = true;
            return first_error;
        }
        if (!ready) return first_error;
        auto pending = std::move(*pending_publication_);
        const uint64_t sequence = pending_sequence_;
        pending_publication_.reset();
        pending_sequence_ = 0;
        finishPublication(std::move(pending), sequence, true);
        made_progress = true;
        return first_error;
    }

    QueuedSubmission queued;
    bool have_submission = false;
    bool use_fallback = quarantine_.quarantined() || next_sequence_ == 0;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (queue_.empty()) return first_error;
        if (!use_fallback && !hasRingCredit(next_sequence_, reaped_sequence_)) {
            const auto elapsed =
                std::chrono::steady_clock::now() - queue_.front().enqueue_time;
            if (elapsed < credit_timeout_) return first_error;
            use_fallback = true;
        }
        queued = std::move(queue_.front());
        queue_.pop_front();
        have_submission = true;
    }
    if (!have_submission) return first_error;

    if (use_fallback || !queued.ready_status.ok() || queued.fence == nullptr) {
        status = deferOrStartFallback(std::move(queued));
        made_progress = true;
        return first_error.ok() ? status : first_error;
    }

    SenderPublishState publish_state = SenderPublishState::kPublished;
    bool quarantine_lane = false;
    status = publish(std::move(queued), next_sequence_, publish_state,
                     quarantine_lane);
    if (!status.ok()) {
        if (quarantine_lane) quarantine_.trip();
        made_progress = true;
        return first_error.ok() ? status : first_error;
    }
    made_progress = true;
    return first_error;
}

Status SenderLane::shutdown() {
    if (shutdown_) return Status::OK();
    accepting_.store(false);
    Status first_error;

    if (pending_publication_ != nullptr) {
        auto status = deferOrStartFallback(std::move(*pending_publication_));
        if (!status.ok()) first_error = status;
        pending_publication_.reset();
        pending_sequence_ = 0;
    }

    std::deque<QueuedSubmission> queued;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queued.swap(queue_);
    }
    while (!queued.empty()) {
        auto status = deferOrStartFallback(std::move(queued.front()));
        if (first_error.ok() && !status.ok()) first_error = status;
        queued.pop_front();
    }

    bool fallback_progress = false;
    auto status = pollFallbacks(fallback_progress);
    if (first_error.ok() && !status.ok()) first_error = status;
    bool fallback_queue_progress = false;
    status = progressFallbackQueue(fallback_queue_progress);
    if (first_error.ok() && !status.ok()) first_error = status;
    while (!fallback_queue_.empty()) {
        failSubmission(fallback_queue_.front().submission);
        fallback_queue_.pop_front();
    }
    for (auto& transfer : fallback_transfers_) {
        transfer->abandon();
        // The child RDMA batch can still reference the ticket-owned requests.
        (void)transfer.release();
    }
    fallback_transfers_.clear();

    for (const auto& published : inflight_) markFailed(published.slices);
    inflight_.clear();
    handle_.reset();
    shutdown_ = true;
    return first_error;
}

Status SenderLane::deferOrStartFallback(QueuedSubmission queued) {
    if (!queued.ready_status.ok() || queued.fence == nullptr) {
        failSubmission(queued.submission);
        return queued.ready_status.ok()
                   ? Status::InvalidArgument("missing PXN ready fence")
                   : queued.ready_status;
    }
    bool ready = false;
    auto status = backend_.queryReady(*queued.fence, ready);
    if (!status.ok()) {
        failSubmission(queued.submission);
        return status;
    }
    if (!ready) {
        fallback_queue_.push_back(std::move(queued));
        return Status::OK();
    }
    return startFallback(std::move(queued));
}

Status SenderLane::startFallback(QueuedSubmission queued) {
    std::unique_ptr<FallbackTransfer> transfer;
    auto status = fallback_.submit(queued.submission, transfer);
    if (!status.ok()) {
        failSubmission(queued.submission);
        return status;
    }
    fallback_bytes_.fetch_add(queued.submission.piece.length,
                              std::memory_order_relaxed);
    if (transfer != nullptr) {
        fallback_transfers_.push_back(std::move(transfer));
    }
    return Status::OK();
}

Status SenderLane::progressFallbackQueue(bool& made_progress) {
    made_progress = false;
    if (fallback_queue_.empty()) return Status::OK();
    auto& queued = fallback_queue_.front();
    bool ready = false;
    auto status = backend_.queryReady(*queued.fence, ready);
    if (!status.ok()) {
        failSubmission(queued.submission);
        fallback_queue_.pop_front();
        made_progress = true;
        return status;
    }
    if (!ready) return Status::OK();
    auto submission = std::move(queued);
    fallback_queue_.pop_front();
    made_progress = true;
    return startFallback(std::move(submission));
}

Status SenderLane::pollFallbacks(bool& made_progress) {
    made_progress = false;
    Status first_error;
    for (auto iterator = fallback_transfers_.begin();
         iterator != fallback_transfers_.end();) {
        bool completed = false;
        auto status = (*iterator)->poll(completed);
        if (!status.ok()) {
            (*iterator)->abandon();
            // The child RDMA batch can still reference the ticket-owned
            // requests.
            (void)iterator->release();
            iterator = fallback_transfers_.erase(iterator);
            if (first_error.ok()) first_error = status;
            made_progress = true;
            continue;
        }
        if (!completed) {
            ++iterator;
            continue;
        }
        iterator = fallback_transfers_.erase(iterator);
        made_progress = true;
    }
    return first_error;
}

Status SenderLane::publish(QueuedSubmission submission, uint64_t sequence,
                           SenderPublishState& state, bool& quarantine_lane) {
    quarantine_lane = false;
    auto slot = slotIndex(sequence);
    if (!slot.has_value()) {
        quarantine_lane = true;
        auto status = deferOrStartFallback(std::move(submission));
        return status.ok()
                   ? Status::InvalidArgument("invalid PXN sender sequence")
                   : status;
    }

    auto& lane = endpoint_.control->lanes[endpoint_.lane_index];
    auto status = prepareDescriptor(submission.submission.piece,
                                    submission.submission.session,
                                    endpoint_.epoch, lane.descriptors[*slot]);
    if (!status.ok()) {
        auto fallback_status = deferOrStartFallback(std::move(submission));
        return fallback_status.ok() ? status : fallback_status;
    }

    const size_t slot_offset =
        (endpoint_.lane_index * kSlotsPerLane + *slot) * kSlotSize;
    if (endpoint_.arena_address >
        std::numeric_limits<uintptr_t>::max() - slot_offset) {
        auto fallback_status = deferOrStartFallback(std::move(submission));
        return fallback_status.ok()
                   ? Status::InvalidArgument("PXN sender slot address overflow")
                   : fallback_status;
    }

    std::vector<SenderCopy> copies;
    copies.reserve(submission.submission.piece.spans.size());
    uintptr_t destination = endpoint_.arena_address + slot_offset;
    for (const auto& span : submission.submission.piece.spans) {
        if (destination > std::numeric_limits<uintptr_t>::max() -
                              static_cast<size_t>(span.length)) {
            auto fallback_status = deferOrStartFallback(std::move(submission));
            return fallback_status.ok()
                       ? Status::InvalidArgument("PXN sender copy overflow")
                       : fallback_status;
        }
        copies.push_back({static_cast<uintptr_t>(span.source), destination,
                          static_cast<size_t>(span.length)});
        destination += static_cast<size_t>(span.length);
    }
    if (!commitDescriptor(lane.descriptors[*slot], sequence)) {
        quarantine_lane = true;
        auto fallback_status = deferOrStartFallback(std::move(submission));
        return fallback_status.ok()
                   ? Status::InvalidArgument("invalid PXN descriptor sequence")
                   : fallback_status;
    }

    status =
        backend_.publish(*handle_, *submission.fence, copies, sequence, state);
    if (!status.ok()) {
        quarantine_lane = true;
        auto fallback_status = deferOrStartFallback(std::move(submission));
        return fallback_status.ok() ? status : fallback_status;
    }
    if (state == SenderPublishState::kPendingCpuDoorbell) {
        pending_sequence_ = sequence;
        pending_publication_ =
            std::make_unique<QueuedSubmission>(std::move(submission));
        return Status::OK();
    }
    finishPublication(std::move(submission), sequence, false);
    return Status::OK();
}

void SenderLane::finishPublication(QueuedSubmission submission,
                                   uint64_t sequence, bool cpu_doorbell) {
    auto& header = endpoint_.control->lanes[endpoint_.lane_index].header;
    __atomic_store_n(&header.head, sequence, __ATOMIC_RELEASE);
    if (cpu_doorbell) {
        __atomic_store_n(&header.doorbell, sequence, __ATOMIC_RELEASE);
    }
    markPosted(submission.submission.slices);
    inflight_.push_back({sequence, std::move(submission.submission.slices)});
    (void)advanceSequence(sequence, next_sequence_);
}

void SenderLane::failSubmission(const SenderSubmission& submission) {
    for (const auto& slice : submission.slices) {
        if (slice.mark_failed != nullptr) slice.mark_failed(slice.context);
    }
}

SenderPipeline::SenderPipeline(std::unique_ptr<SenderBackend> backend,
                               std::unique_ptr<SenderFallback> fallback,
                               std::chrono::milliseconds credit_timeout)
    : backend_(std::move(backend)),
      fallback_(std::move(fallback)),
      credit_timeout_(credit_timeout) {}

SenderPipeline::~SenderPipeline() { (void)shutdown(); }

Status SenderPipeline::addPeer(PeerResources& peer, SenderLane*& lane) {
    lane = nullptr;
    std::lock_guard<std::mutex> lock(lanes_mutex_);
    if (shutdown_ || backend_ == nullptr || fallback_ == nullptr) {
        return Status::InvalidArgument("PXN sender pipeline is not active");
    }
    for (const auto& entry : peer_lanes_) {
        if (entry.identity != peer.entry().identity) continue;
        if (entry.epoch != peer.entry().epoch) {
            return Status::InvalidArgument("PXN sender peer epoch changed");
        }
        lane = entry.lane;
        return Status::OK();
    }

    SenderLaneEndpoint endpoint{peer.entry().epoch, peer.address(),
                                peer.laneIndex(), peer.controlBlock()};
    std::unique_ptr<SenderLane> candidate;
    auto status = SenderLane::Create(endpoint, *backend_, *fallback_,
                                     credit_timeout_, candidate);
    if (!status.ok()) return status;
    lane = candidate.get();
    peer_lanes_.push_back({peer.entry().identity, peer.entry().epoch, lane});
    lanes_.push_back(std::move(candidate));
    return Status::OK();
}

Status SenderPipeline::reapOutbound(bool& made_progress) {
    made_progress = false;
    Status first_error;
    std::vector<SenderLane*> lanes;
    {
        std::lock_guard<std::mutex> lock(lanes_mutex_);
        lanes.reserve(lanes_.size());
        for (const auto& lane : lanes_) lanes.push_back(lane.get());
    }
    for (auto* lane : lanes) {
        bool lane_progress = false;
        auto status = lane->reap(lane_progress);
        if (first_error.ok() && !status.ok()) first_error = status;
        made_progress = made_progress || lane_progress;
    }
    return first_error;
}

Status SenderPipeline::progressOutbound(bool& made_progress) {
    made_progress = false;
    SenderLane* lane = nullptr;
    {
        std::lock_guard<std::mutex> lock(lanes_mutex_);
        if (lanes_.empty()) return Status::OK();
        if (next_lane_ >= lanes_.size()) next_lane_ = 0;
        lane = lanes_[next_lane_].get();
        next_lane_ = (next_lane_ + 1) % lanes_.size();
    }
    return lane->progress(made_progress);
}

uint64_t SenderPipeline::fallbackBytes() {
    std::lock_guard<std::mutex> lock(lanes_mutex_);
    uint64_t total = 0;
    for (const auto& lane : lanes_) total += lane->fallbackBytes();
    return total;
}

Status SenderPipeline::shutdown() {
    std::vector<std::unique_ptr<SenderLane>> lanes;
    {
        std::lock_guard<std::mutex> lock(lanes_mutex_);
        if (shutdown_) return Status::OK();
        shutdown_ = true;
        peer_lanes_.clear();
        lanes.swap(lanes_);
    }
    Status first_error;
    for (auto& lane : lanes) {
        auto status = lane->shutdown();
        if (first_error.ok() && !status.ok()) first_error = status;
    }
    return first_error;
}

namespace {

class RdmaFallbackTransfer final : public FallbackTransfer {
   public:
    RdmaFallbackTransfer(RdmaTransport& transport, Transport::BatchID batch_id,
                         std::vector<LogicalSlice> slices,
                         std::vector<Transport::TransferRequest> requests)
        : transport_(transport),
          batch_id_(batch_id),
          slices_(std::move(slices)),
          completed_(slices_.size(), false),
          requests_(std::move(requests)) {}

    std::vector<Transport::TransferRequest>& requests() { return requests_; }

    Status poll(bool& completed) override {
        completed = false;
        for (size_t index = 0; index < completed_.size(); ++index) {
            if (completed_[index]) continue;
            Transport::TransferStatus status;
            auto result = transport_.RdmaTransport::getTransferStatus(
                batch_id_, index, status);
            if (!result.ok()) return result;
            if (status.s == Transport::WAITING ||
                status.s == Transport::PENDING) {
                continue;
            }
            if (status.s == Transport::COMPLETED) {
                slices_[index].mark_success(slices_[index].context);
            } else {
                slices_[index].mark_failed(slices_[index].context);
            }
            completed_[index] = true;
        }

        if (!std::all_of(completed_.begin(), completed_.end(),
                         [](bool value) { return value; })) {
            return Status::OK();
        }
        auto result = transport_.Transport::freeBatchID(batch_id_);
        if (!result.ok()) return result;
        batch_id_ = 0;
        completed = true;
        return Status::OK();
    }

    void abandon() override {
        for (size_t index = 0; index < completed_.size(); ++index) {
            if (completed_[index]) continue;
            slices_[index].mark_failed(slices_[index].context);
            completed_[index] = true;
        }
    }

   private:
    RdmaTransport& transport_;
    Transport::BatchID batch_id_;
    std::vector<LogicalSlice> slices_;
    std::vector<bool> completed_;
    std::vector<Transport::TransferRequest> requests_;
};

class RdmaSenderFallback final : public SenderFallback {
   public:
    explicit RdmaSenderFallback(RdmaTransport& transport)
        : transport_(transport) {}

    Status submit(const SenderSubmission& submission,
                  std::unique_ptr<FallbackTransfer>& transfer) override {
        if (submission.piece.spans.empty() ||
            submission.piece.spans.size() != submission.slices.size()) {
            return Status::InvalidArgument("invalid PXN fallback submission");
        }

        std::vector<Transport::TransferRequest> requests;
        requests.reserve(submission.piece.spans.size());
        for (const auto& span : submission.piece.spans) {
            Transport::TransferRequest request{};
            request.opcode = Transport::TransferRequest::WRITE;
            request.source = reinterpret_cast<void*>(span.source);
            request.target_id = submission.target_id;
            request.target_offset = span.final_destination;
            request.length = static_cast<size_t>(span.length);
            requests.push_back(request);
        }

        const auto batch_id =
            transport_.Transport::allocateBatchID(requests.size());
        auto candidate = std::make_unique<RdmaFallbackTransfer>(
            transport_, batch_id, submission.slices, std::move(requests));
        auto& batch = Transport::toBatchDesc(batch_id);
        batch.task_list.resize(candidate->requests().size());
        std::vector<Transport::TransferTask*> tasks;
        tasks.reserve(candidate->requests().size());
        for (size_t index = 0; index < candidate->requests().size(); ++index) {
            auto& task = batch.task_list[index];
            task.batch_id = batch_id;
            task.request = &candidate->requests()[index];
            task.transport_ = &transport_;
            tasks.push_back(&task);
        }

        (void)transport_.RdmaTransport::submitTransferTask(tasks);
        transfer = std::move(candidate);
        return Status::OK();
    }

   private:
    RdmaTransport& transport_;
};

}  // namespace

std::unique_ptr<SenderFallback> makeRdmaSenderFallback(
    RdmaTransport& transport) {
    return std::make_unique<RdmaSenderFallback>(transport);
}

}  // namespace pxn
}  // namespace mooncake

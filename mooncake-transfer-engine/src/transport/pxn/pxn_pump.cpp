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

#include <utility>

#include "error.h"
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

class RdmaRelayTransfer final : public RelayTransfer {
   public:
    RdmaRelayTransfer(RdmaTransport& transport, Transport::BatchID batch_id,
                      std::vector<Transport::TransferRequest> requests)
        : transport_(transport),
          batch_id_(batch_id),
          requests_(std::move(requests)) {}

    std::vector<Transport::TransferRequest>& requests() { return requests_; }

    Status poll(bool& completed, int32_t& completion_status) override {
        completed = false;
        bool failed = false;
        for (size_t index = 0; index < requests_.size(); ++index) {
            Transport::TransferStatus status;
            auto result = transport_.RdmaTransport::getTransferStatus(
                batch_id_, index, status);
            if (!result.ok()) return result;
            if (status.s == Transport::WAITING ||
                status.s == Transport::PENDING) {
                return Status::OK();
            }
            failed = failed || status.s != Transport::COMPLETED;
        }

        auto result = transport_.Transport::freeBatchID(batch_id_);
        if (!result.ok()) return result;
        batch_id_ = 0;
        completion_status = failed ? ERR_CONTEXT : 0;
        completed = true;
        return Status::OK();
    }

    void abandon() override {}

   private:
    RdmaTransport& transport_;
    Transport::BatchID batch_id_;
    std::vector<Transport::TransferRequest> requests_;
};

class RdmaRelayBackend final : public RelayBackend {
   public:
    explicit RdmaRelayBackend(RdmaTransport& transport)
        : transport_(transport) {}

    Status submit(const RelaySubmission& submission,
                  std::unique_ptr<RelayTransfer>& transfer) override {
        const auto target_id = transport_.getSegmentID(submission.session);
        if (target_id == static_cast<Transport::SegmentID>(-1)) {
            return Status::Metadata("PXN target session was not found");
        }

        std::vector<Transport::TransferRequest> requests;
        requests.reserve(submission.plans.size());
        uintptr_t source = submission.source;
        for (const auto& plan : submission.plans) {
            Transport::TransferRequest request{};
            request.opcode = Transport::TransferRequest::WRITE;
            request.source = reinterpret_cast<void*>(source);
            request.target_id = target_id;
            request.target_offset = plan.final_destination;
            request.length = static_cast<size_t>(plan.length);
            requests.push_back(request);
            source += plan.length;
        }

        const auto batch_id =
            transport_.Transport::allocateBatchID(requests.size());
        auto candidate = std::make_unique<RdmaRelayTransfer>(
            transport_, batch_id, std::move(requests));
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

std::unique_ptr<RelayBackend> makeRdmaRelayBackend(RdmaTransport& transport) {
    return std::make_unique<RdmaRelayBackend>(transport);
}

}  // namespace pxn
}  // namespace mooncake

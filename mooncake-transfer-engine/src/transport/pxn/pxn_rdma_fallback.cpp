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
#include <utility>

#include "transport/rdma_transport/rdma_transport.h"

namespace mooncake {
namespace pxn {
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
                  std::unique_ptr<FallbackTransfer>& transfer) override {  // 提交一条回退传输
        if (submission.piece.spans.empty() ||
            submission.piece.spans.size() != submission.slices.size()) {  // 校验：span 非空且数量与 slice 一致
            return Status::InvalidArgument("invalid PXN fallback submission");  // 不合法则拒绝
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

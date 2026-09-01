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

#include <gtest/gtest.h>

#include <atomic>
#include <memory>

#include "transport/pxn_transport/pxn_transport.h"
#include "transport/transport.h"

namespace mooncake {
namespace pxn {
namespace {

class ImmediateTransfer final : public RelayTransfer {
   public:
    Status poll(bool& completed, int32_t& completion_status) override {
        completed = true;
        completion_status = 0;
        return Status::OK();
    }

    void abandon() override {}
};

class FakeRelayBackend final : public RelayBackend {
   public:
    Status submit(const RelaySubmission& submission,
                  std::unique_ptr<RelayTransfer>& transfer) override {
        last_submission = submission;
        transfer = std::make_unique<ImmediateTransfer>();
        return Status::OK();
    }

    RelaySubmission last_submission;
};

class PendingTransfer final : public RelayTransfer {
   public:
    explicit PendingTransfer(const bool& ready) : ready_(ready) {}

    Status poll(bool& completed, int32_t& completion_status) override {
        completed = ready_;
        completion_status = 0;
        return Status::OK();
    }

    void abandon() override {}

   private:
    const bool& ready_;
};

class CountingPendingRelayBackend final : public RelayBackend {
   public:
    Status submit(const RelaySubmission&,
                  std::unique_ptr<RelayTransfer>& transfer) override {
        ++submit_count;
        transfer = std::make_unique<PendingTransfer>(complete);
        return Status::OK();
    }

    Status submitBatch(std::span<RelayBatchItem> items) override {
        ++batch_count;
        last_batch_size = items.size();
        return RelayBackend::submitBatch(items);
    }

    size_t submit_count = 0;
    size_t batch_count = 0;
    size_t last_batch_size = 0;
    bool complete = false;
};

class ControlledTransfer final : public RelayTransfer {
   public:
    explicit ControlledTransfer(std::shared_ptr<std::atomic<bool>> completed)
        : completed_(std::move(completed)) {}

    Status poll(bool& completed, int32_t& completion_status) override {
        completed = completed_->load();
        completion_status = 0;
        return Status::OK();
    }

    void abandon() override {}

   private:
    std::shared_ptr<std::atomic<bool>> completed_;
};

class ControlledRelayBackend final : public RelayBackend {
   public:
    Status submit(const RelaySubmission&,
                  std::unique_ptr<RelayTransfer>& transfer) override {
        auto completed = std::make_shared<std::atomic<bool>>(false);
        completions.push_back(completed);
        transfer = std::make_unique<ControlledTransfer>(std::move(completed));
        return Status::OK();
    }

    std::vector<std::shared_ptr<std::atomic<bool>>> completions;
};

class PartialFailureRelayBackend final : public RelayBackend {
   public:
    Status submit(const RelaySubmission&,
                  std::unique_ptr<RelayTransfer>& transfer) override {
        transfer = std::make_unique<ImmediateTransfer>();
        return Status::OK();
    }

    Status submitBatch(std::span<RelayBatchItem> items) override {
        Status first_error;
        for (auto& item : items) {
            if (item.submission.rail_index == 1) {
                item.status = Status::Context("injected rail failure");
                if (first_error.ok()) first_error = item.status;
            } else {
                item.transfer = std::make_unique<ImmediateTransfer>();
            }
        }
        return first_error;
    }
};

TEST(PxnPumpTest, RelaysDoorbellAndPublishesCompletion) {
    auto control = std::make_unique<ControlBlock>();
    auto& lane = control->lanes[2];
    __atomic_store_n(&lane.header.state,
                     static_cast<uint32_t>(LaneState::kReady),
                     __ATOMIC_RELEASE);

    Piece piece;
    piece.length = 64;
    piece.spans.push_back({0, 0x20000000, 64, 0});
    ASSERT_TRUE(
        prepareDescriptor(piece, "decode", 7, lane.descriptors[0]).ok());
    ASSERT_TRUE(commitDescriptor(lane.descriptors[0], 1));
    __atomic_store_n(&lane.header.doorbell, uint64_t{1}, __ATOMIC_RELEASE);

    auto backend = std::make_unique<FakeRelayBackend>();
    auto* backend_ptr = backend.get();
    RelayPipeline relay(control.get(), 0x400000000ULL, 7, 4,
                        std::move(backend));

    bool progressed = false;
    ASSERT_TRUE(relay.progressInbound(progressed).ok());
    ASSERT_TRUE(progressed);
    EXPECT_EQ(backend_ptr->last_submission.source,
              0x400000000ULL + 2 * kSlotsPerLane * kSlotSize);
    EXPECT_EQ(backend_ptr->last_submission.session, "decode");
    ASSERT_EQ(backend_ptr->last_submission.plans.size(), 1u);
    EXPECT_EQ(backend_ptr->last_submission.plans[0].final_destination,
              0x20000000u);

    ASSERT_TRUE(relay.reapInbound(progressed).ok());
    ASSERT_TRUE(progressed);
    int32_t completion_status = -1;
    EXPECT_TRUE(tryLoadCompletion(lane.completions[0], 1, completion_status));
    EXPECT_EQ(completion_status, 0);
    EXPECT_EQ(lane.header.completed, 1u);
}

TEST(PxnPumpTest, SubmitsReadySlotsBeforeEarlierRdmaCompletes) {
    constexpr size_t kReadySlots = 6;
    constexpr size_t kInflightLimit = 4;
    auto control = std::make_unique<ControlBlock>();
    auto& lane = control->lanes[0];
    __atomic_store_n(&lane.header.state,
                     static_cast<uint32_t>(LaneState::kReady),
                     __ATOMIC_RELEASE);

    for (uint64_t sequence = 1; sequence <= kReadySlots; ++sequence) {
        Piece piece;
        piece.length = kSlotSize;
        piece.spans.push_back(
            {0, 0x20000000 + (sequence - 1) * kSlotSize, kSlotSize, 0});
        const size_t slot = *slotIndex(sequence);
        ASSERT_TRUE(prepareDescriptor(piece, "decode-overlap", 7,
                                      lane.descriptors[slot])
                        .ok());
        ASSERT_TRUE(commitDescriptor(lane.descriptors[slot], sequence));
    }
    __atomic_store_n(&lane.header.doorbell, static_cast<uint64_t>(kReadySlots),
                     __ATOMIC_RELEASE);

    auto backend = std::make_unique<CountingPendingRelayBackend>();
    auto* backend_ptr = backend.get();
    RelayPipeline relay(control.get(), 0x400000000ULL, 7, kInflightLimit,
                        std::move(backend));

    bool progressed = false;
    ASSERT_TRUE(relay.progressInbound(progressed).ok());
    EXPECT_TRUE(progressed);
    EXPECT_EQ(backend_ptr->submit_count, kInflightLimit);
    EXPECT_EQ(backend_ptr->batch_count, 1u);
    EXPECT_EQ(backend_ptr->last_batch_size, kInflightLimit);
    EXPECT_EQ(relay.inflightStats().current, kInflightLimit);
    EXPECT_EQ(relay.inflightStats().high_watermark, kInflightLimit);
    EXPECT_EQ(relay.inflightStats().submit_batches, 1u);
    EXPECT_EQ(relay.inflightStats().submitted_batch_slots, kInflightLimit);

    backend_ptr->complete = true;
    ASSERT_TRUE(relay.reapInbound(progressed).ok());
    EXPECT_TRUE(progressed);
    EXPECT_EQ(relay.inflightStats().current, 0u);
}

TEST(PxnPumpTest, BatchesReadySlotsAcrossActiveLanes) {
    auto control = std::make_unique<ControlBlock>();
    constexpr PxnGeometry geometry{3, 4, 256};
    for (size_t lane_index = 0; lane_index < geometry.lane_count;
         ++lane_index) {
        auto& lane = control->lanes[lane_index];
        __atomic_store_n(&lane.header.state,
                         static_cast<uint32_t>(LaneState::kReady),
                         __ATOMIC_RELEASE);
        lane.header.sender_epoch = lane_index + 1;
        Piece piece;
        piece.length = geometry.slot_size;
        piece.spans.push_back({0, 0x30000000 + lane_index * geometry.slot_size,
                               geometry.slot_size, 0});
        ASSERT_TRUE(prepareDescriptor(piece, "decode-batch", 9, 0,
                                      lane.descriptors[0], geometry.slot_size)
                        .ok());
        ASSERT_TRUE(commitDescriptor(lane.descriptors[0], 1));
        __atomic_store_n(&lane.header.doorbell, uint64_t{1}, __ATOMIC_RELEASE);
    }

    auto backend = std::make_unique<CountingPendingRelayBackend>();
    auto* backend_ptr = backend.get();
    RelayPipeline relay(control.get(), 0x500000000ULL, 9, 8, std::move(backend),
                        geometry);

    bool progressed = false;
    ASSERT_TRUE(relay.progressInbound(progressed).ok());
    EXPECT_TRUE(progressed);
    EXPECT_EQ(backend_ptr->batch_count, 1u);
    EXPECT_EQ(backend_ptr->last_batch_size, geometry.lane_count);
    EXPECT_EQ(backend_ptr->submit_count, geometry.lane_count);
    EXPECT_EQ(relay.inflightStats().current, geometry.lane_count);
}

TEST(PxnPumpTest, SameLaneCompletionWaitsForContiguousSequence) {
    auto control = std::make_unique<ControlBlock>();
    auto& lane = control->lanes[0];
    __atomic_store_n(&lane.header.state,
                     static_cast<uint32_t>(LaneState::kReady),
                     __ATOMIC_RELEASE);
    for (uint64_t sequence = 1; sequence <= 2; ++sequence) {
        Piece piece;
        piece.length = 64;
        piece.spans.push_back({0, 0x60000000 + (sequence - 1) * 64, 64, 0});
        const size_t slot = *slotIndex(sequence);
        ASSERT_TRUE(
            prepareDescriptor(piece, "decode-order", 11, lane.descriptors[slot])
                .ok());
        ASSERT_TRUE(commitDescriptor(lane.descriptors[slot], sequence));
    }
    __atomic_store_n(&lane.header.doorbell, uint64_t{2}, __ATOMIC_RELEASE);

    auto backend = std::make_unique<ControlledRelayBackend>();
    auto* backend_ptr = backend.get();
    RelayPipeline relay(control.get(), 0x600000000ULL, 11, 4,
                        std::move(backend));
    bool progressed = false;
    ASSERT_TRUE(relay.progressInbound(progressed).ok());
    ASSERT_EQ(backend_ptr->completions.size(), 2u);

    backend_ptr->completions[1]->store(true);
    ASSERT_TRUE(relay.reapInbound(progressed).ok());
    EXPECT_FALSE(progressed);
    EXPECT_EQ(lane.header.completed, 0u);

    backend_ptr->completions[0]->store(true);
    ASSERT_TRUE(relay.reapInbound(progressed).ok());
    EXPECT_TRUE(progressed);
    EXPECT_EQ(lane.header.completed, 2u);
}

TEST(PxnPumpTest, BatchFailureIsScopedToAffectedRail) {
    auto control = std::make_unique<ControlBlock>();
    for (size_t lane_index = 0; lane_index < 2; ++lane_index) {
        auto& lane = control->lanes[lane_index];
        __atomic_store_n(&lane.header.state,
                         static_cast<uint32_t>(LaneState::kReady),
                         __ATOMIC_RELEASE);
        Piece piece;
        piece.length = 64;
        piece.spans.push_back({0, 0x70000000 + lane_index * 64, 64, 0});
        ASSERT_TRUE(prepareDescriptor(piece, "decode-failure", 13,
                                      static_cast<uint32_t>(lane_index),
                                      lane.descriptors[0])
                        .ok());
        ASSERT_TRUE(commitDescriptor(lane.descriptors[0], 1));
        __atomic_store_n(&lane.header.doorbell, uint64_t{1}, __ATOMIC_RELEASE);
    }

    RelayPipeline relay(control.get(), 0x700000000ULL, 13, 4,
                        std::make_unique<PartialFailureRelayBackend>());
    bool progressed = false;
    EXPECT_FALSE(relay.progressInbound(progressed).ok());
    EXPECT_TRUE(progressed);
    ASSERT_TRUE(relay.reapInbound(progressed).ok());
    EXPECT_TRUE(progressed);

    int32_t status = -1;
    ASSERT_TRUE(tryLoadCompletion(control->lanes[0].completions[0], 1, status));
    EXPECT_EQ(status, 0);
    ASSERT_TRUE(tryLoadCompletion(control->lanes[1].completions[0], 1, status));
    EXPECT_NE(status, 0);
}

TEST(PxnPumpTest, CompletesSliceWithoutTransferTask) {
    struct Completion {
        size_t completed = 0;
        bool failed = false;
    } completion;
    auto callback = [](Transport::Slice* slice, bool success) {
        auto* state = static_cast<Completion*>(slice->completion_context);
        ++state->completed;
        state->failed = state->failed || !success;
    };

    Transport::Slice slice{};
    slice.length = 64;
    slice.completion_callback = callback;
    slice.completion_context = &completion;
    slice.markSuccess();

    Transport::Slice failed_slice{};
    failed_slice.completion_callback = callback;
    failed_slice.completion_context = &completion;
    failed_slice.markFailed();

    EXPECT_EQ(completion.completed, 2u);
    EXPECT_TRUE(completion.failed);
    EXPECT_EQ(slice.status, Transport::Slice::SUCCESS);
    EXPECT_EQ(failed_slice.status, Transport::Slice::FAILED);
}

}  // namespace
}  // namespace pxn
}  // namespace mooncake

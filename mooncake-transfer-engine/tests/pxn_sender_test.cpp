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

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "transport/pxn_transport/pxn_transport.h"

namespace mooncake {
namespace pxn {
namespace {

struct SliceState {
    std::atomic<int> posted{0};
    std::atomic<int> succeeded{0};
    std::atomic<int> failed{0};
};

void markPosted(void* context) {
    static_cast<SliceState*>(context)->posted.fetch_add(1);
}

void markSucceeded(void* context) {
    static_cast<SliceState*>(context)->succeeded.fetch_add(1);
}

void markFailed(void* context) {
    static_cast<SliceState*>(context)->failed.fetch_add(1);
}

SenderSubmission makeSubmission(size_t index, SliceState& state) {
    SenderSubmission submission;
    submission.piece.length = 64;
    submission.piece.spans.push_back(
        {0x10000000ULL + index * 64, 0x20000000ULL + index * 64, 64, index});
    submission.session = "decode";
    submission.target_id = 7;
    submission.slices.push_back(
        {&state, 64, markPosted, markSucceeded, markFailed});
    return submission;
}

class FakeFence final : public SenderReadyFence {};

class FakeLaneHandle final : public SenderLaneHandle {
   public:
    explicit FakeLaneHandle(SenderLaneEndpoint endpoint) : endpoint(endpoint) {}

    SenderLaneEndpoint endpoint;
};

class FakeSenderBackend final : public SenderBackend {
   public:
    Status createLane(const SenderLaneEndpoint& endpoint,
                      std::unique_ptr<SenderLaneHandle>& handle) override {
        handle = std::make_unique<FakeLaneHandle>(endpoint);
        return Status::OK();
    }

    Status recordReady(std::unique_ptr<SenderReadyFence>& fence) override {
        record_count.fetch_add(1);
        fence = std::make_unique<FakeFence>();
        return Status::OK();
    }

    Status queryReady(const SenderReadyFence&, bool& ready) override {
        ready = source_ready.load();
        return Status::OK();
    }

    Status publish(SenderLaneHandle& handle, const SenderReadyFence&,
                   const std::vector<SenderCopy>& copies, uint64_t sequence,
                   SenderPublishState& state) override {
        auto& lane_handle = static_cast<FakeLaneHandle&>(handle);
        auto& lane = lane_handle.endpoint.control
                         ->lanes[lane_handle.endpoint.lane_index];
        const auto slot = slotIndex(sequence);
        if (!slot.has_value() ||
            loadDescriptorSequence(lane.descriptors[*slot]) != sequence ||
            copies.empty()) {
            return Status::InvalidArgument("descriptor was not committed");
        }
        publish_count.fetch_add(1);
        if (fail_publish.load()) {
            return Status::Memory("publish failed");
        }
        if (pending_cpu_doorbell.load()) {
            state = SenderPublishState::kPendingCpuDoorbell;
            return Status::OK();
        }
        __atomic_store_n(&lane.header.doorbell, sequence, __ATOMIC_RELEASE);
        state = SenderPublishState::kPublished;
        return Status::OK();
    }

    Status pollCpuDoorbell(SenderLaneHandle&, bool& ready) override {
        ready = cpu_doorbell_ready.load();
        return Status::OK();
    }

    std::atomic<size_t> record_count{0};
    std::atomic<size_t> publish_count{0};
    std::atomic<bool> source_ready{true};
    std::atomic<bool> pending_cpu_doorbell{false};
    std::atomic<bool> cpu_doorbell_ready{false};
    std::atomic<bool> fail_publish{false};
};

class SynchronousFallback final : public SenderFallback {
   public:
    Status submit(const SenderSubmission& submission,
                  std::unique_ptr<FallbackTransfer>& transfer) override {
        submit_count.fetch_add(1);
        for (const auto& slice : submission.slices) {
            slice.mark_success(slice.context);
        }
        transfer.reset();
        return Status::OK();
    }

    std::atomic<size_t> submit_count{0};
};

TEST(PxnSenderTest, MpscQueueRespectsEighteenSlotCredit) {
    auto control = std::make_unique<ControlBlock>();
    FakeSenderBackend backend;
    SynchronousFallback fallback;
    std::unique_ptr<SenderLane> lane;
    ASSERT_TRUE(SenderLane::Create({9, 0x400000000ULL, 0, control.get()},
                                   backend, fallback, std::chrono::seconds(1),
                                   lane)
                    .ok());

    std::array<SliceState, 19> states;
    std::atomic<bool> enqueue_ok{true};
    auto producer = [&](size_t begin, size_t end) {
        for (size_t index = begin; index < end; ++index) {
            if (!lane->enqueue(makeSubmission(index, states[index])).ok()) {
                enqueue_ok.store(false);
            }
        }
    };
    std::thread first(producer, 0, 9);
    std::thread second(producer, 9, 18);
    first.join();
    second.join();
    ASSERT_TRUE(enqueue_ok.load());

    for (size_t index = 0; index < 18; ++index) {
        bool progressed = false;
        ASSERT_TRUE(lane->progress(progressed).ok());
        ASSERT_TRUE(progressed);
    }
    EXPECT_EQ(lane->inflightCount(), 18u);
    EXPECT_EQ(lane->nextSequence(), 19u);
    EXPECT_EQ(control->lanes[0].header.doorbell, 18u);

    ASSERT_TRUE(lane->enqueue(makeSubmission(18, states[18])).ok());
    bool progressed = true;
    ASSERT_TRUE(lane->progress(progressed).ok());
    EXPECT_FALSE(progressed);
    EXPECT_EQ(backend.publish_count.load(), 18u);

    ASSERT_TRUE(commitCompletion(control->lanes[0].completions[0], 1, 0));
    ASSERT_TRUE(lane->reap(progressed).ok());
    EXPECT_TRUE(progressed);
    ASSERT_TRUE(lane->progress(progressed).ok());
    EXPECT_TRUE(progressed);
    EXPECT_EQ(loadDescriptorSequence(control->lanes[0].descriptors[0]), 19u);
    EXPECT_EQ(control->lanes[0].header.doorbell, 19u);
    EXPECT_EQ(backend.record_count.load(), 19u);
    EXPECT_EQ(fallback.submit_count.load(), 0u);
}

TEST(PxnSenderTest, PublishesAfterCopyAndFallsBackBeforeDoorbell) {
    auto control = std::make_unique<ControlBlock>();
    FakeSenderBackend backend;
    backend.pending_cpu_doorbell.store(true);
    SynchronousFallback fallback;
    std::unique_ptr<SenderLane> lane;
    ASSERT_TRUE(SenderLane::Create({11, 0x500000000ULL, 1, control.get()},
                                   backend, fallback, std::chrono::seconds(1),
                                   lane)
                    .ok());

    SliceState first;
    ASSERT_TRUE(lane->enqueue(makeSubmission(0, first)).ok());
    bool progressed = false;
    ASSERT_TRUE(lane->progress(progressed).ok());
    EXPECT_TRUE(progressed);
    EXPECT_EQ(first.posted.load(), 0);
    EXPECT_EQ(control->lanes[1].header.doorbell, 0u);

    backend.cpu_doorbell_ready.store(true);
    ASSERT_TRUE(lane->progress(progressed).ok());
    EXPECT_TRUE(progressed);
    EXPECT_EQ(first.posted.load(), 1);
    EXPECT_EQ(control->lanes[1].header.doorbell, 1u);

    backend.pending_cpu_doorbell.store(false);
    backend.fail_publish.store(true);
    backend.source_ready.store(false);
    SliceState second;
    ASSERT_TRUE(lane->enqueue(makeSubmission(1, second)).ok());
    EXPECT_FALSE(lane->progress(progressed).ok());
    EXPECT_TRUE(progressed);
    EXPECT_TRUE(lane->quarantined());
    EXPECT_EQ(control->lanes[1].header.doorbell, 1u);
    EXPECT_EQ(second.succeeded.load(), 0);
    EXPECT_EQ(fallback.submit_count.load(), 0u);

    backend.source_ready.store(true);
    ASSERT_TRUE(lane->progress(progressed).ok());
    EXPECT_TRUE(progressed);
    EXPECT_EQ(second.succeeded.load(), 1);
    EXPECT_EQ(fallback.submit_count.load(), 1u);
}

}  // namespace
}  // namespace pxn
}  // namespace mooncake

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

#include <memory>

#include "transport/pxn/pxn_transport.h"
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

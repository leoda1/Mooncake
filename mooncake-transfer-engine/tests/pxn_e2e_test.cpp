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
//
// PXN health-path end-to-end coverage (Commit 6). Every case wires a real
// SenderLane to a real RelayPipeline through one shared ControlBlock and drives
// both sides with the same four-step order PxnPump::run() uses, but without a
// thread so the interleaving stays deterministic.

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "transport/pxn_transport/pxn_transport.h"

namespace mooncake {
namespace pxn {
namespace {

constexpr uint64_t kEpoch = 42;

// ---------------------------------------------------------------------------
// Sender-side doubles
// ---------------------------------------------------------------------------

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

class StubFence final : public SenderReadyFence {};

class StubLaneHandle final : public SenderLaneHandle {
   public:
    explicit StubLaneHandle(SenderLaneEndpoint endpoint) : endpoint(endpoint) {}
    SenderLaneEndpoint endpoint;
};

// Stands in for makeCudaSenderBackend(). publish() copies the payload with the
// CPU and rings the doorbell itself, which is what cudaMemcpyBatchAsync plus
// cuStreamWriteValue64 do on a real GPU.
class HostSenderBackend final : public SenderBackend {
   public:
    Status createLane(const SenderLaneEndpoint& endpoint,
                      std::unique_ptr<SenderLaneHandle>& handle) override {
        handle = std::make_unique<StubLaneHandle>(endpoint);
        return Status::OK();
    }

    Status recordReady(std::unique_ptr<SenderReadyFence>& fence) override {
        fence = std::make_unique<StubFence>();
        return Status::OK();
    }

    Status queryReady(const SenderReadyFence&, bool& ready) override {
        ready = true;
        return Status::OK();
    }

    Status publish(SenderLaneHandle& handle, const SenderReadyFence&,
                   const std::vector<SenderCopy>& copies, uint64_t sequence,
                   SenderPublishState& state) override {
        auto& lane_handle = static_cast<StubLaneHandle&>(handle);
        auto& lane = lane_handle.endpoint.control
                         ->lanes[lane_handle.endpoint.lane_index];
        const auto slot = slotIndex(sequence);
        if (!slot.has_value() ||
            loadDescriptorSequence(lane.descriptors[*slot]) != sequence ||
            copies.empty()) {
            return Status::InvalidArgument("descriptor was not committed");
        }
        for (const auto& copy : copies) {
            std::memcpy(reinterpret_cast<void*>(copy.destination),
                        reinterpret_cast<const void*>(copy.source),
                        copy.length);
            copied_bytes.fetch_add(copy.length);
        }
        publish_count.fetch_add(1);
        batched_copies.fetch_add(copies.size());
        __atomic_store_n(&lane.header.doorbell, sequence, __ATOMIC_RELEASE);
        state = SenderPublishState::kPublished;
        return Status::OK();
    }

    Status pollCpuDoorbell(SenderLaneHandle&, bool& ready) override {
        ready = true;
        return Status::OK();
    }

    std::atomic<size_t> publish_count{0};
    std::atomic<size_t> batched_copies{0};
    std::atomic<uint64_t> copied_bytes{0};
};

// The health path must never touch this.
class CountingFallback final : public SenderFallback {
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

// ---------------------------------------------------------------------------
// Relay-side double: performs the second hop as a host copy so payload bytes
// can be verified end to end.
// ---------------------------------------------------------------------------

class HostRelayTransfer final : public RelayTransfer {
   public:
    Status poll(bool& completed, int32_t& completion_status) override {
        completed = true;
        completion_status = 0;
        return Status::OK();
    }
    void abandon() override {}
};

class HostRelayBackend final : public RelayBackend {
   public:
    Status submit(const RelaySubmission& submission,
                  std::unique_ptr<RelayTransfer>& transfer) override {
        uintptr_t source = submission.source;
        for (const auto& plan : submission.plans) {
            std::memcpy(reinterpret_cast<void*>(plan.final_destination),
                        reinterpret_cast<const void*>(source), plan.length);
            source += plan.length;
            forwarded_bytes.fetch_add(plan.length);
        }
        submit_count.fetch_add(1);
        last_session = submission.session;
        last_source = submission.source;
        transfer = std::make_unique<HostRelayTransfer>();
        return Status::OK();
    }

    std::atomic<size_t> submit_count{0};
    std::atomic<uint64_t> forwarded_bytes{0};
    std::string last_session;
    uintptr_t last_source = 0;
};

// ---------------------------------------------------------------------------
// Test fixture: one ControlBlock shared by sender and relay, plus a real
// heap-backed arena so payload bytes actually move.
// ---------------------------------------------------------------------------

class PxnE2E {
   public:
    explicit PxnE2E(std::chrono::milliseconds credit_timeout =
                        std::chrono::milliseconds(60000),
                    size_t max_inflight = 4)
        : control_(std::make_unique<ControlBlock>()),
          arena_(kRequiredArenaSize) {
        auto relay_backend = std::make_unique<HostRelayBackend>();
        relay_backend_ = relay_backend.get();
        // The arena the relay reads from is the same memory the sender wrote,
        // so both sides are handed the same base address.
        relay_ = std::make_unique<RelayPipeline>(
            control_.get(), reinterpret_cast<uintptr_t>(arena_.data()), kEpoch,
            max_inflight, std::move(relay_backend));
        credit_timeout_ = credit_timeout;
    }

    // Brings one lane up the way Registry::claimLane would.
    Status openLane(size_t lane_index, std::unique_ptr<SenderLane>& lane) {
        auto& lane_control = control_->lanes[lane_index];
        __atomic_store_n(&lane_control.header.state,
                         static_cast<uint32_t>(LaneState::kReady),
                         __ATOMIC_RELEASE);
        SenderLaneEndpoint endpoint;
        endpoint.epoch = kEpoch;
        endpoint.arena_address = reinterpret_cast<uintptr_t>(arena_.data());
        endpoint.lane_index = lane_index;
        endpoint.control = control_.get();
        return SenderLane::Create(endpoint, *sender_backend_, *fallback_,
                                  credit_timeout_, lane);
    }

    // One iteration of PxnPump::run()'s fixed four-step order.
    void pump(SenderLane& lane) {
        bool progressed = false;
        (void)lane.reap(progressed);
        (void)relay_->reapInbound(progressed);
        (void)relay_->progressInbound(progressed);
        (void)lane.progress(progressed);
    }

    // Drives until every slice reports a terminal state or the budget runs out.
    bool drain(SenderLane& lane, SliceState& state, int expected_slices,
               int max_iterations = 400000) {
        for (int i = 0; i < max_iterations; ++i) {
            if (state.succeeded.load() + state.failed.load() >=
                expected_slices) {
                return true;
            }
            pump(lane);
        }
        return false;
    }

    ControlBlock* control() { return control_.get(); }
    RelayPipeline& relay() { return *relay_; }
    HostRelayBackend& relayBackend() { return *relay_backend_; }
    HostSenderBackend& senderBackend() { return *sender_backend_; }
    CountingFallback& fallback() { return *fallback_; }
    uint8_t* arena() { return arena_.data(); }

   private:
    std::unique_ptr<ControlBlock> control_;
    std::vector<uint8_t> arena_;
    std::unique_ptr<HostSenderBackend> sender_backend_ =
        std::make_unique<HostSenderBackend>();
    std::unique_ptr<CountingFallback> fallback_ =
        std::make_unique<CountingFallback>();
    HostRelayBackend* relay_backend_ = nullptr;
    std::unique_ptr<RelayPipeline> relay_;
    std::chrono::milliseconds credit_timeout_{60000};
};

// Builds a submission whose source is `source` and whose final destination is
// `destination`, split into Pieces exactly the way the transport does.
std::vector<SenderSubmission> makeSubmissions(uint8_t* source,
                                              uint8_t* destination,
                                              size_t length,
                                              const std::string& session,
                                              SliceState& state) {
    std::vector<TransferSpan> spans;
    spans.push_back({reinterpret_cast<uint64_t>(source),
                     reinterpret_cast<uint64_t>(destination), length, 0});
    std::vector<Piece> pieces;
    EXPECT_TRUE(buildPieces(spans, pieces).ok());

    std::vector<SenderSubmission> submissions;
    submissions.reserve(pieces.size());
    for (auto& piece : pieces) {
        SenderSubmission submission;
        submission.session = session;
        submission.target_id = 1;
        submission.slices.reserve(piece.spans.size());
        for (const auto& span : piece.spans) {
            submission.slices.push_back({&state, span.length, markPosted,
                                         markSucceeded, markFailed});
        }
        submission.piece = std::move(piece);
        submissions.push_back(std::move(submission));
    }
    return submissions;
}

size_t totalSlices(const std::vector<SenderSubmission>& submissions) {
    size_t count = 0;
    for (const auto& submission : submissions) {
        count += submission.slices.size();
    }
    return count;
}

void fillPattern(uint8_t* buffer, size_t length, uint8_t seed) {
    for (size_t i = 0; i < length; ++i) {
        buffer[i] = static_cast<uint8_t>((i * 31 + seed) & 0xff);
    }
}

// ---------------------------------------------------------------------------
// 1. Cross-rail 10 KiB through the PXN relay.
// ---------------------------------------------------------------------------

TEST(PxnE2ETest, CrossRail10KiBRelays) {
    constexpr size_t kLength = 10 * 1024;
    PxnE2E env;
    std::unique_ptr<SenderLane> lane;
    ASSERT_TRUE(env.openLane(0, lane).ok());

    std::vector<uint8_t> source(kLength);
    std::vector<uint8_t> destination(kLength, 0);
    fillPattern(source.data(), kLength, 7);

    SliceState state;
    auto submissions = makeSubmissions(source.data(), destination.data(),
                                       kLength, "decode-a", state);
    ASSERT_EQ(submissions.size(), 1u) << "10KiB must fit one Piece";
    const auto slices = totalSlices(submissions);
    for (auto& submission : submissions) {
        ASSERT_TRUE(lane->enqueue(std::move(submission)).ok());
    }

    ASSERT_TRUE(env.drain(*lane, state, static_cast<int>(slices)));
    EXPECT_EQ(state.succeeded.load(), static_cast<int>(slices));
    EXPECT_EQ(state.failed.load(), 0);
    EXPECT_EQ(destination, source) << "payload must arrive byte-identical";

    // Health path: everything went through the relay, nothing fell back.
    EXPECT_GT(env.relay().relayedBytes(), 0u);
    EXPECT_EQ(lane->fallbackBytes(), 0u);
    EXPECT_EQ(env.fallback().submit_count.load(), 0u);
    EXPECT_EQ(env.relay().relayedBytes(), kLength);
    EXPECT_EQ(env.relayBackend().last_session, "decode-a");
}

// ---------------------------------------------------------------------------
// 2. 8 MiB exactly, and the two boundaries either side of a full slot.
// ---------------------------------------------------------------------------

TEST(PxnE2ETest, SlotSizeBoundaries) {
    struct Case {
        const char* name;
        size_t length;
        size_t expected_pieces;
    };
    const Case cases[] = {
        {"8MiB-1", kSlotSize - 1, 1},
        {"8MiB", kSlotSize, 1},
        {"8MiB+1", kSlotSize + 1, 2},
    };

    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        PxnE2E env;
        std::unique_ptr<SenderLane> lane;
        ASSERT_TRUE(env.openLane(0, lane).ok());

        std::vector<uint8_t> source(test_case.length);
        std::vector<uint8_t> destination(test_case.length, 0);
        fillPattern(source.data(), test_case.length, 11);

        SliceState state;
        auto submissions = makeSubmissions(source.data(), destination.data(),
                                           test_case.length, "decode-b", state);
        EXPECT_EQ(submissions.size(), test_case.expected_pieces);
        const auto slices = totalSlices(submissions);
        for (auto& submission : submissions) {
            ASSERT_TRUE(lane->enqueue(std::move(submission)).ok());
        }

        ASSERT_TRUE(env.drain(*lane, state, static_cast<int>(slices)));
        EXPECT_EQ(state.failed.load(), 0);
        EXPECT_EQ(destination, source);
        EXPECT_EQ(env.relay().relayedBytes(), test_case.length);
        EXPECT_EQ(lane->fallbackBytes(), 0u);
    }
}

// ---------------------------------------------------------------------------
// 3. 18 Pieces fill the ring exactly; the 19th waits for credit and then
//    reuses slot 0.
// ---------------------------------------------------------------------------

TEST(PxnE2ETest, EighteenPiecesFillRingNineteenthReusesSlotZero) {
    // A long credit timeout keeps the 19th Piece waiting instead of falling
    // back, which is what the health path requires.
    PxnE2E env(std::chrono::milliseconds(60000));
    std::unique_ptr<SenderLane> lane;
    ASSERT_TRUE(env.openLane(3, lane).ok());

    auto& lane_control = env.control()->lanes[3];
    SliceState state;

    // Enqueue 18 single-span Pieces and publish them without reaping, so the
    // ring ends up exactly full.
    std::vector<std::vector<uint8_t>> sources;
    std::vector<std::vector<uint8_t>> destinations;
    for (size_t i = 0; i < kSlotsPerLane; ++i) {
        sources.emplace_back(64, static_cast<uint8_t>(i));
        destinations.emplace_back(64, 0);
    }
    for (size_t i = 0; i < kSlotsPerLane; ++i) {
        SenderSubmission submission;
        submission.session = "decode-c";
        submission.target_id = 1;
        submission.piece.length = 64;
        submission.piece.spans.push_back(
            {reinterpret_cast<uint64_t>(sources[i].data()),
             reinterpret_cast<uint64_t>(destinations[i].data()), 64, 0});
        submission.slices.push_back(
            {&state, 64, markPosted, markSucceeded, markFailed});
        ASSERT_TRUE(lane->enqueue(std::move(submission)).ok());
    }

    // Only drive the sender so nothing is reaped yet.
    for (size_t i = 0; i < kSlotsPerLane; ++i) {
        bool progressed = false;
        ASSERT_TRUE(lane->progress(progressed).ok());
        ASSERT_TRUE(progressed) << "Piece " << i << " should publish";
    }
    EXPECT_EQ(env.senderBackend().publish_count.load(), kSlotsPerLane);
    EXPECT_EQ(lane->nextSequence(), kSlotsPerLane + 1);
    EXPECT_EQ(__atomic_load_n(&lane_control.header.doorbell, __ATOMIC_ACQUIRE),
              kSlotsPerLane);

    // The 19th Piece has no credit: 19 - 0 > 18.
    std::vector<uint8_t> extra_source(64, 0xAB);
    std::vector<uint8_t> extra_destination(64, 0);
    SenderSubmission extra;
    extra.session = "decode-c";
    extra.target_id = 1;
    extra.piece.length = 64;
    extra.piece.spans.push_back(
        {reinterpret_cast<uint64_t>(extra_source.data()),
         reinterpret_cast<uint64_t>(extra_destination.data()), 64, 0});
    extra.slices.push_back(
        {&state, 64, markPosted, markSucceeded, markFailed});
    ASSERT_TRUE(lane->enqueue(std::move(extra)).ok());

    bool progressed = true;
    ASSERT_TRUE(lane->progress(progressed).ok());
    EXPECT_FALSE(progressed) << "19th Piece must wait, not publish";
    EXPECT_EQ(env.senderBackend().publish_count.load(), kSlotsPerLane)
        << "still 18 publishes";
    EXPECT_EQ(env.fallback().submit_count.load(), 0u)
        << "waiting must not become a fallback";

    // Free slot 0 by completing sequence 1, then the 19th Piece can publish
    // into it.
    ASSERT_TRUE(commitCompletion(lane_control.completions[0], 1, 0));
    __atomic_store_n(&lane_control.header.completed, uint64_t{1},
                     __ATOMIC_RELEASE);
    ASSERT_TRUE(lane->reap(progressed).ok());
    EXPECT_EQ(lane->reapedSequence(), 1u);

    ASSERT_TRUE(lane->progress(progressed).ok());
    EXPECT_TRUE(progressed) << "credit freed, 19th Piece publishes";
    EXPECT_EQ(env.senderBackend().publish_count.load(), kSlotsPerLane + 1);
    // (19 - 1) % 18 == 0
    EXPECT_EQ(loadDescriptorSequence(lane_control.descriptors[0]),
              kSlotsPerLane + 1);
    EXPECT_EQ(lane->fallbackBytes(), 0u);
}

// ---------------------------------------------------------------------------
// 4. 5 GiB splits into exactly 640 Pieces, and 640 Pieces really do wrap the
//    18-slot ring many times with every byte verified.
// ---------------------------------------------------------------------------

// buildPieces() never dereferences the span addresses, so the 5 GiB split can
// be checked without allocating 5 GiB.
TEST(PxnE2ETest, FiveGiBSplitsIntoSixHundredFortyPieces) {
    constexpr uint64_t kLength = 5ULL * 1024 * 1024 * 1024;
    std::vector<TransferSpan> spans;
    spans.push_back({0x10000000ULL, 0x20000000ULL, kLength, 0});

    std::vector<Piece> pieces;
    ASSERT_TRUE(buildPieces(spans, pieces).ok());
    EXPECT_EQ(pieces.size(), 640u);
    EXPECT_EQ(kLength / kSlotSize, 640u);

    uint64_t total = 0;
    for (const auto& piece : pieces) {
        EXPECT_LE(piece.length, kSlotSize);
        total += piece.length;
    }
    EXPECT_EQ(total, kLength) << "no bytes lost across the split";
}

// Moves 640 Pieces through the lane for real. Piece size is kept at 64 KiB so
// the test stays memory-light; the 5 GiB Piece-count math is covered above and
// the full 8 MiB slot is covered by SlotSizeBoundaries.
TEST(PxnE2ETest, SixHundredFortyPiecesWrapRingAndVerifyPayload) {
    constexpr size_t kPieces = 640;
    constexpr size_t kPayload = 64 * 1024;

    PxnE2E env;
    std::unique_ptr<SenderLane> lane;
    ASSERT_TRUE(env.openLane(1, lane).ok());

    std::vector<std::vector<uint8_t>> sources;
    std::vector<std::vector<uint8_t>> destinations;
    sources.reserve(kPieces);
    destinations.reserve(kPieces);
    SliceState state;
    for (size_t i = 0; i < kPieces; ++i) {
        sources.emplace_back(kPayload);
        // A per-Piece pattern so a mis-routed Piece cannot pass unnoticed.
        fillPattern(sources.back().data(), kPayload, static_cast<uint8_t>(i));
        destinations.emplace_back(kPayload, 0);
    }
    for (size_t i = 0; i < kPieces; ++i) {
        SenderSubmission submission;
        submission.session = "decode-d";
        submission.target_id = 1;
        submission.piece.length = kPayload;
        submission.piece.spans.push_back(
            {reinterpret_cast<uint64_t>(sources[i].data()),
             reinterpret_cast<uint64_t>(destinations[i].data()), kPayload, 0});
        submission.slices.push_back({&state, kPayload, markPosted,
                                     markSucceeded, markFailed});
        ASSERT_TRUE(lane->enqueue(std::move(submission)).ok());
    }

    ASSERT_TRUE(env.drain(*lane, state, static_cast<int>(kPieces)))
        << "all 640 Pieces should complete";
    EXPECT_EQ(state.succeeded.load(), static_cast<int>(kPieces));
    EXPECT_EQ(state.failed.load(), 0);

    for (size_t i = 0; i < kPieces; ++i) {
        ASSERT_EQ(destinations[i], sources[i]) << "Piece " << i << " corrupted";
    }

    // 640 sequences over an 18-slot ring is ~35 wraps.
    EXPECT_EQ(lane->nextSequence(), kPieces + 1);
    EXPECT_GT(kPieces / kSlotsPerLane, 30u);
    EXPECT_EQ(env.relay().relayedBytes(), kPieces * kPayload);
    EXPECT_EQ(lane->fallbackBytes(), 0u);
    EXPECT_EQ(env.fallback().submit_count.load(), 0u);
}

// ---------------------------------------------------------------------------
// 5. Seven senders write the seven lanes of one relay concurrently.
// ---------------------------------------------------------------------------

TEST(PxnE2ETest, SevenSendersShareOneRelay) {
    PxnE2E env(std::chrono::milliseconds(60000), /*max_inflight=*/4);

    std::vector<std::unique_ptr<SenderLane>> lanes(kLaneCount);
    for (size_t i = 0; i < kLaneCount; ++i) {
        ASSERT_TRUE(env.openLane(i, lanes[i]).ok());
    }

    constexpr size_t kPiecesPerLane = 6;
    constexpr size_t kPayload = 4096;
    std::vector<SliceState> states(kLaneCount);
    std::vector<std::vector<std::vector<uint8_t>>> sources(kLaneCount);
    std::vector<std::vector<std::vector<uint8_t>>> destinations(kLaneCount);

    for (size_t lane_index = 0; lane_index < kLaneCount; ++lane_index) {
        for (size_t piece = 0; piece < kPiecesPerLane; ++piece) {
            sources[lane_index].emplace_back(kPayload);
            fillPattern(sources[lane_index].back().data(), kPayload,
                        static_cast<uint8_t>(lane_index * 16 + piece));
            destinations[lane_index].emplace_back(kPayload, 0);
        }
    }

    // Enqueue from seven threads at once, mirroring seven independent senders.
    // gtest assertions are not thread-safe, so failures are recorded and
    // checked on the main thread.
    std::array<std::atomic<int>, kLaneCount> enqueue_failures{};
    for (auto& counter : enqueue_failures) counter.store(0);
    std::vector<std::thread> producers;
    for (size_t lane_index = 0; lane_index < kLaneCount; ++lane_index) {
        producers.emplace_back([&, lane_index]() {
            for (size_t piece = 0; piece < kPiecesPerLane; ++piece) {
                SenderSubmission submission;
                submission.session = "decode-e";
                submission.target_id = 1;
                submission.piece.length = kPayload;
                submission.piece.spans.push_back(
                    {reinterpret_cast<uint64_t>(
                         sources[lane_index][piece].data()),
                     reinterpret_cast<uint64_t>(
                         destinations[lane_index][piece].data()),
                     kPayload, 0});
                submission.slices.push_back({&states[lane_index], kPayload,
                                             markPosted, markSucceeded,
                                             markFailed});
                if (!lanes[lane_index]->enqueue(std::move(submission)).ok()) {
                    enqueue_failures[lane_index].fetch_add(1);
                }
            }
        });
    }
    for (auto& producer : producers) producer.join();
    for (size_t lane_index = 0; lane_index < kLaneCount; ++lane_index) {
        ASSERT_EQ(enqueue_failures[lane_index].load(), 0)
            << "lane " << lane_index << " had enqueue failures";
    }

    // Round-robin the pump across all seven lanes. max_inflight is 4, below the
    // 7 lanes, so relay progress and reap must interleave for this to finish.
    bool done = false;
    for (int iteration = 0; iteration < 400000 && !done; ++iteration) {
        for (size_t lane_index = 0; lane_index < kLaneCount; ++lane_index) {
            env.pump(*lanes[lane_index]);
        }
        done = true;
        for (size_t lane_index = 0; lane_index < kLaneCount; ++lane_index) {
            const int settled = states[lane_index].succeeded.load() +
                                states[lane_index].failed.load();
            if (settled < static_cast<int>(kPiecesPerLane)) done = false;
        }
    }
    ASSERT_TRUE(done) << "all seven lanes should drain";

    for (size_t lane_index = 0; lane_index < kLaneCount; ++lane_index) {
        EXPECT_EQ(states[lane_index].succeeded.load(),
                  static_cast<int>(kPiecesPerLane))
            << "lane " << lane_index;
        EXPECT_EQ(states[lane_index].failed.load(), 0) << "lane " << lane_index;
        EXPECT_EQ(lanes[lane_index]->fallbackBytes(), 0u)
            << "lane " << lane_index;
        for (size_t piece = 0; piece < kPiecesPerLane; ++piece) {
            EXPECT_EQ(destinations[lane_index][piece],
                      sources[lane_index][piece])
                << "lane " << lane_index << " piece " << piece;
        }
    }
    EXPECT_EQ(env.relay().relayedBytes(),
              kLaneCount * kPiecesPerLane * kPayload);
    EXPECT_EQ(env.fallback().submit_count.load(), 0u);
}

// ---------------------------------------------------------------------------
// 6. The relay forwards on its own, with no sender-side work driving it.
// ---------------------------------------------------------------------------

TEST(PxnE2ETest, RelayForwardsWithoutLocalSenderActivity) {
    PxnE2E env;
    // Publish a descriptor by hand, exactly as a remote sender process would,
    // without creating any local SenderLane.
    auto& lane = env.control()->lanes[5];
    __atomic_store_n(&lane.header.state,
                     static_cast<uint32_t>(LaneState::kReady),
                     __ATOMIC_RELEASE);

    constexpr size_t kPayload = 2048;
    std::vector<uint8_t> destination(kPayload, 0);
    // Stage the payload where the relay expects to read it: lane 5, slot 0.
    uint8_t* slot = env.arena() + (5 * kSlotsPerLane + 0) * kSlotSize;
    fillPattern(slot, kPayload, 77);

    Piece piece;
    piece.length = kPayload;
    piece.spans.push_back(
        {0, reinterpret_cast<uint64_t>(destination.data()), kPayload, 0});
    ASSERT_TRUE(
        prepareDescriptor(piece, "decode-f", kEpoch, lane.descriptors[0]).ok());
    ASSERT_TRUE(commitDescriptor(lane.descriptors[0], 1));
    __atomic_store_n(&lane.header.doorbell, uint64_t{1}, __ATOMIC_RELEASE);

    bool progressed = false;
    ASSERT_TRUE(env.relay().progressInbound(progressed).ok());
    EXPECT_TRUE(progressed) << "relay must pick up the doorbell unprompted";
    ASSERT_TRUE(env.relay().reapInbound(progressed).ok());
    EXPECT_TRUE(progressed);

    int32_t completion_status = -1;
    EXPECT_TRUE(tryLoadCompletion(lane.completions[0], 1, completion_status));
    EXPECT_EQ(completion_status, 0);
    EXPECT_EQ(__atomic_load_n(&lane.header.completed, __ATOMIC_ACQUIRE), 1u);
    EXPECT_EQ(env.relay().relayedBytes(), kPayload);
    EXPECT_EQ(std::memcmp(destination.data(), slot, kPayload), 0);
}

// ---------------------------------------------------------------------------
// 7. Health-path invariant stated as its own case.
// ---------------------------------------------------------------------------

TEST(PxnE2ETest, HealthPathRelaysEverythingAndNeverFallsBack) {
    PxnE2E env;
    std::unique_ptr<SenderLane> lane;
    ASSERT_TRUE(env.openLane(2, lane).ok());

    constexpr size_t kPieces = 40;
    constexpr size_t kPayload = 64 * 1024;
    std::vector<std::vector<uint8_t>> sources;
    std::vector<std::vector<uint8_t>> destinations;
    SliceState state;
    for (size_t i = 0; i < kPieces; ++i) {
        sources.emplace_back(kPayload);
        fillPattern(sources.back().data(), kPayload, static_cast<uint8_t>(i));
        destinations.emplace_back(kPayload, 0);
    }
    for (size_t i = 0; i < kPieces; ++i) {
        SenderSubmission submission;
        submission.session = "decode-g";
        submission.target_id = 1;
        submission.piece.length = kPayload;
        submission.piece.spans.push_back(
            {reinterpret_cast<uint64_t>(sources[i].data()),
             reinterpret_cast<uint64_t>(destinations[i].data()), kPayload, 0});
        submission.slices.push_back({&state, kPayload, markPosted,
                                     markSucceeded, markFailed});
        ASSERT_TRUE(lane->enqueue(std::move(submission)).ok());
    }

    ASSERT_TRUE(env.drain(*lane, state, static_cast<int>(kPieces)));

    // This is the Commit 6 acceptance criterion.
    EXPECT_GT(env.relay().relayedBytes(), 0u);
    EXPECT_EQ(lane->fallbackBytes(), 0u);

    EXPECT_EQ(env.relay().relayedBytes(), kPieces * kPayload);
    EXPECT_EQ(env.fallback().submit_count.load(), 0u);
    EXPECT_EQ(state.succeeded.load(), static_cast<int>(kPieces));
    EXPECT_EQ(state.failed.load(), 0);
    EXPECT_FALSE(lane->quarantined());
    for (size_t i = 0; i < kPieces; ++i) {
        EXPECT_EQ(destinations[i], sources[i]) << "piece " << i;
    }
}

// ---------------------------------------------------------------------------
// 8. The copies handed to the backend are batched, which is what lets the CUDA
//    backend issue one cudaMemcpyBatchAsync per Piece.
// ---------------------------------------------------------------------------

TEST(PxnE2ETest, MultiSpanPieceIsPublishedAsOneBatch) {
    PxnE2E env;
    std::unique_ptr<SenderLane> lane;
    ASSERT_TRUE(env.openLane(0, lane).ok());

    constexpr size_t kSpans = 64;
    constexpr size_t kSpanLength = 1024;
    std::vector<std::vector<uint8_t>> sources;
    std::vector<std::vector<uint8_t>> destinations;
    SliceState state;
    SenderSubmission submission;
    submission.session = "decode-h";
    submission.target_id = 1;
    submission.piece.length = kSpans * kSpanLength;
    for (size_t i = 0; i < kSpans; ++i) {
        sources.emplace_back(kSpanLength);
        fillPattern(sources.back().data(), kSpanLength,
                    static_cast<uint8_t>(i));
        destinations.emplace_back(kSpanLength, 0);
    }
    for (size_t i = 0; i < kSpans; ++i) {
        submission.piece.spans.push_back(
            {reinterpret_cast<uint64_t>(sources[i].data()),
             reinterpret_cast<uint64_t>(destinations[i].data()), kSpanLength,
             i});
        submission.slices.push_back({&state, kSpanLength, markPosted,
                                     markSucceeded, markFailed});
    }
    ASSERT_TRUE(lane->enqueue(std::move(submission)).ok());

    ASSERT_TRUE(env.drain(*lane, state, static_cast<int>(kSpans)));

    // One publish carrying all 64 copies: the CUDA backend turns exactly this
    // vector into a single cudaMemcpyBatchAsync call.
    EXPECT_EQ(env.senderBackend().publish_count.load(), 1u);
    EXPECT_EQ(env.senderBackend().batched_copies.load(), kSpans);
    EXPECT_EQ(env.senderBackend().copied_bytes.load(), kSpans * kSpanLength);

    EXPECT_EQ(state.succeeded.load(), static_cast<int>(kSpans));
    EXPECT_EQ(env.relay().relayedBytes(), kSpans * kSpanLength);
    EXPECT_EQ(lane->fallbackBytes(), 0u);
    for (size_t i = 0; i < kSpans; ++i) {
        EXPECT_EQ(destinations[i], sources[i]) << "span " << i;
    }
}

}  // namespace
}  // namespace pxn
}  // namespace mooncake

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
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.h"
#include "transport/pxn_transport/pxn_transport.h"

namespace mooncake {
namespace pxn {
namespace {

struct BackendState {
    uintptr_t local_address = 0;
    size_t expected_arena_size = kDefaultArenaSize;
    bool fail_register = false;
    bool fail_export = false;
    std::vector<std::string> calls;
};

class FakeBackend final : public StagingBackend {
   public:
    explicit FakeBackend(std::shared_ptr<BackendState> state)
        : state_(std::move(state)) {}

    Status allocateArena(size_t size, uintptr_t& address) override {
        state_->calls.push_back("allocate");
        if (size != state_->expected_arena_size) {
            return Status::InvalidArgument("unexpected arena size");
        }
        address = state_->local_address;
        return Status::OK();
    }

    Status freeArena(uintptr_t) override {
        state_->calls.push_back("free");
        return Status::OK();
    }

    Status registerArena(uintptr_t, size_t) override {
        state_->calls.push_back("register");
        return state_->fail_register ? Status::Memory("register failed")
                                     : Status::OK();
    }

    Status unregisterArena(uintptr_t) override {
        state_->calls.push_back("unregister");
        return Status::OK();
    }

    Status exportArena(uintptr_t address, CudaIpcHandle& handle) override {
        state_->calls.push_back("export");
        if (state_->fail_export) return Status::Memory("export failed");
        std::memcpy(handle.bytes, &address, sizeof(address));
        return Status::OK();
    }

    Status importArena(const CudaIpcHandle& handle,
                       uintptr_t& address) override {
        state_->calls.push_back("import");
        std::memcpy(&address, handle.bytes, sizeof(address));
        return Status::OK();
    }

    Status closeImportedArena(uintptr_t) override {
        state_->calls.push_back("close");
        return Status::OK();
    }

   private:
    std::shared_ptr<BackendState> state_;
};

class TemporaryDirectory {
   public:
    TemporaryDirectory() {
        std::array<char, 64> pattern{};
        std::strcpy(pattern.data(), "/tmp/mooncake-pxn-test.XXXXXX");
        char* path = mkdtemp(pattern.data());
        if (path == nullptr) throw std::runtime_error("mkdtemp failed");
        path_ = path;
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    const std::string& path() const { return path_; }

   private:
    std::string path_;
};

RegistryOptions registryOptions(const TemporaryDirectory& directory,
                                int32_t pid, uint64_t epoch,
                                int32_t local_rank_index = 0) {
    RegistryOptions options;
    options.root_directory = directory.path();
    options.group_id = "job";
    options.identity = ProcessIdentity{static_cast<uint32_t>(geteuid()), pid,
                                       static_cast<uint64_t>(pid)};
    options.epoch = epoch;
    options.local_rank_index = local_rank_index;
    options.process_probe = [](const ProcessIdentity&) {
        return std::optional<bool>(true);
    };
    return options;
}

class ScopedPxnGeometryConfig {
   public:
    explicit ScopedPxnGeometryConfig(PxnGeometry geometry)
        : config_(globalConfig()),
          saved_{config_.pxn_lane_count, config_.pxn_slots_per_lane,
                 config_.pxn_slot_size} {
        config_.pxn_lane_count = geometry.lane_count;
        config_.pxn_slots_per_lane = geometry.slots_per_lane;
        config_.pxn_slot_size = geometry.slot_size;
    }

    ~ScopedPxnGeometryConfig() {
        config_.pxn_lane_count = saved_.lane_count;
        config_.pxn_slots_per_lane = saved_.slots_per_lane;
        config_.pxn_slot_size = saved_.slot_size;
    }

   private:
    GlobalConfig& config_;
    PxnGeometry saved_;
};

TEST(PxnLifecycleTest, MapsPeerLaneAndHoldsArenaLease) {
    TemporaryDirectory directory;
    auto relay_state = std::make_shared<BackendState>();
    relay_state->local_address = 0x100000000ULL;
    auto sender_state = std::make_shared<BackendState>();
    sender_state->local_address = 0x200000000ULL;
    const std::array<std::string, 1> hcas{"mlx5_0"};

    std::unique_ptr<LocalResources> relay;
    ASSERT_TRUE(LocalResources::Create(
                    registryOptions(directory, 1001, 11, 3), hcas, {},
                    std::make_unique<FakeBackend>(relay_state), relay)
                    .ok());
    std::unique_ptr<LocalResources> sender;
    ASSERT_TRUE(LocalResources::Create(
                    registryOptions(directory, 1002, 22, 5), hcas, {},
                    std::make_unique<FakeBackend>(sender_state), sender)
                    .ok());

    std::vector<RegistryEntry> peers;
    ASSERT_TRUE(sender->registry().discover(peers).ok());
    ASSERT_EQ(peers.size(), 1u);
    EXPECT_EQ(peers.front().local_rank_index, 3);
    PeerResources* peer = nullptr;
    ASSERT_TRUE(sender->mapPeer(peers.front(), peer).ok());
    ASSERT_NE(peer, nullptr);
    EXPECT_EQ(peer->laneIndex(), 0u);
    EXPECT_EQ(peer->slotAddress(kSlotsPerLane - 1),
              relay_state->local_address + (kSlotsPerLane - 1) * kSlotSize);

    EXPECT_TRUE(relay->shutdown().IsBatchBusy());
    EXPECT_TRUE(sender->shutdown().ok());
    EXPECT_TRUE(relay->shutdown().ok());
    EXPECT_NE(std::find(sender_state->calls.begin(), sender_state->calls.end(),
                        "import"),
              sender_state->calls.end());
    EXPECT_NE(std::find(sender_state->calls.begin(), sender_state->calls.end(),
                        "close"),
              sender_state->calls.end());
}

TEST(PxnLifecycleTest, RollsBackExportAndQuarantinesRegistrationFailure) {
    auto export_state = std::make_shared<BackendState>();
    export_state->local_address = 0x300000000ULL;
    export_state->fail_export = true;
    FakeBackend export_backend(export_state);
    StagingArena export_arena;
    EXPECT_FALSE(export_arena.initialize(export_backend).ok());
    EXPECT_EQ(export_state->calls,
              (std::vector<std::string>{"allocate", "register", "export",
                                        "unregister", "free"}));

    auto register_state = std::make_shared<BackendState>();
    register_state->local_address = 0x400000000ULL;
    register_state->fail_register = true;
    FakeBackend register_backend(register_state);
    StagingArena register_arena;
    EXPECT_FALSE(register_arena.initialize(register_backend).ok());
    EXPECT_TRUE(register_arena.quarantined());
    EXPECT_EQ(register_state->calls,
              (std::vector<std::string>{"allocate", "register"}));
}

TEST(PxnLifecycleTest, StagingArenaUsesRuntimeGeometry) {
    constexpr PxnGeometry geometry{2, 3, 256 * 1024};
    auto state = std::make_shared<BackendState>();
    state->local_address = 0x500000000ULL;
    state->expected_arena_size = geometry.arenaSize();
    FakeBackend backend(state);
    StagingArena arena;

    ASSERT_TRUE(arena.initialize(backend, geometry).ok());
    EXPECT_EQ(
        arena.laneAddress(1),
        state->local_address + geometry.slots_per_lane * geometry.slot_size);
    EXPECT_EQ(arena.slotAddress(1, 2),
              state->local_address +
                  (geometry.slots_per_lane + 2) * geometry.slot_size);
    EXPECT_FALSE(arena.laneAddress(2).has_value());
    EXPECT_FALSE(arena.slotAddress(1, 3).has_value());
    EXPECT_TRUE(arena.shutdown().ok());
}

TEST(PxnLifecycleTest, RegistryPublishesRuntimeGeometry) {
    constexpr PxnGeometry geometry{2, 3, 256 * 1024};
    ScopedPxnGeometryConfig config_scope(geometry);
    TemporaryDirectory directory;
    auto state = std::make_shared<BackendState>();
    state->local_address = 0x600000000ULL;
    state->expected_arena_size = geometry.arenaSize();
    const std::array<std::string, 1> hcas{"mlx5_0"};

    std::unique_ptr<LocalResources> resources;
    ASSERT_TRUE(LocalResources::Create(
                    registryOptions(directory, 1003, 33, 7), hcas, {},
                    std::make_unique<FakeBackend>(state), resources)
                    .ok());
    const auto& header = resources->registration().control()->header;
    EXPECT_EQ(header.lane_count, geometry.lane_count);
    EXPECT_EQ(header.slots_per_lane, geometry.slots_per_lane);
    EXPECT_EQ(header.slot_size, geometry.slot_size);
    EXPECT_EQ(header.arena_size, geometry.arenaSize());
    EXPECT_EQ(header.local_rank_index, 7);
    EXPECT_TRUE(resources->shutdown().ok());
}

}  // namespace
}  // namespace pxn
}  // namespace mooncake

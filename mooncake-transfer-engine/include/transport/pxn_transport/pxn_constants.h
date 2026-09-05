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

#ifndef MOONCAKE_TRANSFER_ENGINE_PXN_CONSTANTS_H_
#define MOONCAKE_TRANSFER_ENGINE_PXN_CONSTANTS_H_

#include <cstddef>
#include <limits>

namespace mooncake {
namespace pxn {

inline constexpr size_t kMaxLaneCount = 7;
inline constexpr size_t kMaxSlotsPerLane = 18;
// Keep each staging unit small so the relay can submit RDMA for an earlier
// slot while CUDA is still filling later slots on the sender stream.
inline constexpr size_t kDefaultSlotSize = 512ULL * 1024;
// Compatibility aliases for tests and fixed-capacity shared-memory types.
inline constexpr size_t kLaneCount = kMaxLaneCount;
inline constexpr size_t kSlotsPerLane = kMaxSlotsPerLane;
inline constexpr size_t kSlotSize = kDefaultSlotSize;
inline constexpr size_t kDefaultArenaSize =
    kMaxLaneCount * kMaxSlotsPerLane * kDefaultSlotSize;
inline constexpr size_t kRequiredArenaSize = kDefaultArenaSize;
inline constexpr size_t kMaxPlanCount = 4096;
inline constexpr size_t kMaxSessionLength = 256;
inline constexpr size_t kMaxRailsPerRank = 16;

struct PxnGeometry {
    size_t lane_count = kMaxLaneCount;
    size_t slots_per_lane = kMaxSlotsPerLane;
    size_t slot_size = kDefaultSlotSize;

    constexpr bool valid() const {
        return lane_count >= 1 && lane_count <= kMaxLaneCount &&
               slots_per_lane >= 1 && slots_per_lane <= kMaxSlotsPerLane &&
               slot_size >= 1 &&
               slots_per_lane <=
                   std::numeric_limits<size_t>::max() / slot_size &&
               lane_count <= std::numeric_limits<size_t>::max() /
                                 (slots_per_lane * slot_size);
    }

    constexpr size_t arenaSize() const {
        return lane_count * slots_per_lane * slot_size;
    }
};

static_assert(kDefaultArenaSize == 63ULL * 1024 * 1024);

}  // namespace pxn
}  // namespace mooncake

#endif  // MOONCAKE_TRANSFER_ENGINE_PXN_CONSTANTS_H_

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
#include <cstdint>

namespace mooncake {
namespace pxn {

inline constexpr size_t kLaneCount = 7;
inline constexpr size_t kSlotsPerLane = 18;
inline constexpr size_t kSlotSize = 8ULL * 1024 * 1024;
inline constexpr size_t kReadyStepSize = 1ULL * 1024 * 1024;
inline constexpr size_t kRequiredArenaSize =
    kLaneCount * kSlotsPerLane * kSlotSize;
inline constexpr size_t kMaxPlanCount = 4096;
inline constexpr size_t kMaxSessionLength = 256;
inline constexpr size_t kMaxRailsPerRank = 16;

static_assert(kRequiredArenaSize == 1008ULL * 1024 * 1024);
static_assert(kSlotSize % kReadyStepSize == 0);

inline constexpr uint64_t readyStepCount(uint64_t length) {
    return (length + kReadyStepSize - 1) / kReadyStepSize;
}

}  // namespace pxn
}  // namespace mooncake

#endif  // MOONCAKE_TRANSFER_ENGINE_PXN_CONSTANTS_H_

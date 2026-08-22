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

#ifndef MOONCAKE_TRANSFER_ENGINE_PXN_CORE_H_
#define MOONCAKE_TRANSFER_ENGINE_PXN_CORE_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "common/base/status.h"
#include "transport/pxn_transport/pxn_constants.h"

namespace mooncake {
namespace pxn {

struct PlanEntry {
    uint64_t final_destination;
    uint64_t length;
};

struct alignas(64) Descriptor {
    uint64_t commit_sequence;
    uint64_t epoch;
    uint32_t piece_length;
    uint32_t plan_count;
    uint32_t session_length;
    uint32_t reserved;
    char session[kMaxSessionLength];
    PlanEntry plans[kMaxPlanCount];
};

struct alignas(64) Completion {
    uint64_t commit_sequence;
    int32_t status;
    uint32_t reserved;
    uint8_t padding[48];
};

static_assert(sizeof(PlanEntry) == 16);
static_assert(alignof(PlanEntry) == 8);
static_assert(offsetof(PlanEntry, final_destination) == 0);
static_assert(offsetof(PlanEntry, length) == 8);
static_assert(alignof(Descriptor) == 64);
static_assert(offsetof(Descriptor, commit_sequence) == 0);
static_assert(offsetof(Descriptor, epoch) == 8);
static_assert(offsetof(Descriptor, piece_length) == 16);
static_assert(offsetof(Descriptor, plan_count) == 20);
static_assert(offsetof(Descriptor, session_length) == 24);
static_assert(offsetof(Descriptor, reserved) == 28);
static_assert(offsetof(Descriptor, session) == 32);
static_assert(offsetof(Descriptor, plans) == 288);
static_assert(sizeof(Descriptor) == 65856);
static_assert(alignof(Completion) == 64);
static_assert(offsetof(Completion, commit_sequence) == 0);
static_assert(offsetof(Completion, status) == 8);
static_assert(offsetof(Completion, reserved) == 12);
static_assert(sizeof(Completion) == 64);
static_assert(std::is_standard_layout_v<Descriptor>);
static_assert(std::is_trivially_copyable_v<Descriptor>);
static_assert(std::is_standard_layout_v<Completion>);
static_assert(std::is_trivially_copyable_v<Completion>);
static_assert(__atomic_always_lock_free(sizeof(uint64_t), nullptr));

enum class DescriptorError {
    kOk = 0,
    kSequence,
    kEpoch,
    kPieceLength,
    kPlanCount,
    kSessionLength,
    kReserved,
    kPlanLength,
    kDestinationOverflow,
    kLengthMismatch,
};

struct TransferSpan {
    uint64_t source;
    uint64_t final_destination;
    uint64_t length;
    size_t request_index;
};

struct PieceSpan {
    uint64_t source;
    uint64_t final_destination;
    uint64_t length;
    size_t request_index;
};

struct Piece {
    uint64_t length = 0;
    std::vector<PieceSpan> spans;
};

class RailResolver {
   public:
    explicit RailResolver(
        std::unordered_map<std::string, std::string> rail_map);

    std::string canonicalize(std::string_view rail) const;
    std::optional<std::string> resolveUnique(
        std::span<const std::string> rails) const;

   private:
    std::unordered_map<std::string, std::string> rail_map_;
};

std::optional<size_t> slotIndex(uint64_t sequence);
bool hasRingCredit(uint64_t next_sequence, uint64_t reaped_sequence);
bool advanceSequence(uint64_t sequence, uint64_t& next_sequence);

Status buildPieces(std::span<const TransferSpan> spans,
                   std::vector<Piece>& pieces);

Status prepareDescriptor(const Piece& piece, std::string_view session,
                         uint64_t epoch, Descriptor& descriptor);

bool commitDescriptor(Descriptor& descriptor, uint64_t sequence);
uint64_t loadDescriptorSequence(const Descriptor& descriptor);
DescriptorError validateDescriptor(const Descriptor& descriptor,
                                   uint64_t expected_sequence,
                                   uint64_t expected_epoch);

bool commitCompletion(Completion& completion, uint64_t sequence,
                      int32_t status);
uint64_t loadCompletionSequence(const Completion& completion);
bool tryLoadCompletion(const Completion& completion, uint64_t expected_sequence,
                       int32_t& status);

}  // namespace pxn
}  // namespace mooncake

#endif  // MOONCAKE_TRANSFER_ENGINE_PXN_CORE_H_

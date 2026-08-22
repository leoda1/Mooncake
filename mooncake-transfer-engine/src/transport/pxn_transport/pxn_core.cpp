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

#include "transport/pxn/pxn_core.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace mooncake {
namespace pxn {
namespace {

bool rangeOverflows(uint64_t address, uint64_t length) {
    return length == 0 ||
           address > std::numeric_limits<uint64_t>::max() - (length - 1);
}

Status validatePiece(const Piece& piece) {
    if (piece.length == 0 || piece.length > kSlotSize) {
        return Status::InvalidArgument("invalid PXN piece length");
    }
    if (piece.spans.empty() || piece.spans.size() > kMaxPlanCount) {
        return Status::InvalidArgument("invalid PXN plan count");
    }

    uint64_t total = 0;
    for (const auto& span : piece.spans) {
        if (rangeOverflows(span.source, span.length) ||
            rangeOverflows(span.final_destination, span.length) ||
            span.length > piece.length - total) {
            return Status::InvalidArgument("invalid PXN piece span");
        }
        total += span.length;
    }
    if (total != piece.length) {
        return Status::InvalidArgument("PXN piece length mismatch");
    }
    return Status::OK();
}

}  // namespace

RailResolver::RailResolver(
    std::unordered_map<std::string, std::string> rail_map)
    : rail_map_(std::move(rail_map)) {}

std::string RailResolver::canonicalize(std::string_view rail) const {
    auto it = rail_map_.find(std::string(rail));
    return it == rail_map_.end() ? std::string(rail) : it->second;
}

std::optional<std::string> RailResolver::resolveUnique(
    std::span<const std::string> rails) const {
    if (rails.size() != 1 || rails.front().empty()) return std::nullopt;
    return canonicalize(rails.front());
}

std::optional<size_t> slotIndex(uint64_t sequence) {
    if (sequence == 0) return std::nullopt;
    return static_cast<size_t>((sequence - 1) % kSlotsPerLane);
}

bool hasRingCredit(uint64_t next_sequence, uint64_t reaped_sequence) {
    if (next_sequence == 0 || next_sequence <= reaped_sequence) return false;
    return next_sequence - reaped_sequence <= kSlotsPerLane;
}

bool advanceSequence(uint64_t sequence, uint64_t& next_sequence) {
    if (sequence == 0 || sequence == std::numeric_limits<uint64_t>::max()) {
        next_sequence = 0;
        return false;
    }
    next_sequence = sequence + 1;
    return true;
}

Status buildPieces(std::span<const TransferSpan> spans,
                   std::vector<Piece>& pieces) {
    std::vector<Piece> result;

    for (const auto& span : spans) {
        if (rangeOverflows(span.source, span.length) ||
            rangeOverflows(span.final_destination, span.length)) {
            return Status::InvalidArgument("invalid PXN transfer span");
        }

        uint64_t offset = 0;
        while (offset < span.length) {
            if (result.empty() || result.back().length == kSlotSize ||
                result.back().spans.size() == kMaxPlanCount) {
                result.emplace_back();
            }

            auto& piece = result.back();
            const uint64_t available = kSlotSize - piece.length;
            const uint64_t length = std::min(span.length - offset, available);
            piece.spans.push_back({span.source + offset,
                                   span.final_destination + offset, length,
                                   span.request_index});
            piece.length += length;
            offset += length;
        }
    }

    pieces = std::move(result);
    return Status::OK();
}

Status prepareDescriptor(const Piece& piece, std::string_view session,
                         uint64_t epoch, Descriptor& descriptor) {
    auto status = validatePiece(piece);
    if (!status.ok()) return status;
    if (session.empty() || session.size() > kMaxSessionLength) {
        return Status::InvalidArgument("invalid PXN session length");
    }
    if (epoch == 0) return Status::InvalidArgument("invalid PXN epoch");

    constexpr size_t kPayloadOffset = offsetof(Descriptor, epoch);
    std::memset(reinterpret_cast<char*>(&descriptor) + kPayloadOffset, 0,
                sizeof(descriptor) - kPayloadOffset);
    descriptor.epoch = epoch;
    descriptor.piece_length = static_cast<uint32_t>(piece.length);
    descriptor.plan_count = static_cast<uint32_t>(piece.spans.size());
    descriptor.session_length = static_cast<uint32_t>(session.size());
    std::memcpy(descriptor.session, session.data(), session.size());
    for (size_t i = 0; i < piece.spans.size(); ++i) {
        descriptor.plans[i] = {piece.spans[i].final_destination,
                               piece.spans[i].length};
    }
    return Status::OK();
}

bool commitDescriptor(Descriptor& descriptor, uint64_t sequence) {
    if (sequence == 0) return false;
    __atomic_store_n(&descriptor.commit_sequence, sequence, __ATOMIC_RELEASE);
    return true;
}

uint64_t loadDescriptorSequence(const Descriptor& descriptor) {
    return __atomic_load_n(&descriptor.commit_sequence, __ATOMIC_ACQUIRE);
}

DescriptorError validateDescriptor(const Descriptor& descriptor,
                                   uint64_t expected_sequence,
                                   uint64_t expected_epoch) {
    if (expected_sequence == 0 ||
        loadDescriptorSequence(descriptor) != expected_sequence) {
        return DescriptorError::kSequence;
    }
    if (expected_epoch == 0 || descriptor.epoch != expected_epoch) {
        return DescriptorError::kEpoch;
    }
    if (descriptor.piece_length == 0 || descriptor.piece_length > kSlotSize) {
        return DescriptorError::kPieceLength;
    }
    if (descriptor.plan_count == 0 || descriptor.plan_count > kMaxPlanCount) {
        return DescriptorError::kPlanCount;
    }
    if (descriptor.session_length == 0 ||
        descriptor.session_length > kMaxSessionLength) {
        return DescriptorError::kSessionLength;
    }
    if (descriptor.reserved != 0) return DescriptorError::kReserved;

    uint64_t total = 0;
    for (size_t i = 0; i < descriptor.plan_count; ++i) {
        const auto& entry = descriptor.plans[i];
        if (entry.length == 0 ||
            entry.length > descriptor.piece_length - total) {
            return DescriptorError::kPlanLength;
        }
        if (rangeOverflows(entry.final_destination, entry.length)) {
            return DescriptorError::kDestinationOverflow;
        }
        total += entry.length;
    }
    return total == descriptor.piece_length ? DescriptorError::kOk
                                            : DescriptorError::kLengthMismatch;
}

bool commitCompletion(Completion& completion, uint64_t sequence,
                      int32_t status) {
    if (sequence == 0) return false;
    completion.status = status;
    completion.reserved = 0;
    __atomic_store_n(&completion.commit_sequence, sequence, __ATOMIC_RELEASE);
    return true;
}

uint64_t loadCompletionSequence(const Completion& completion) {
    return __atomic_load_n(&completion.commit_sequence, __ATOMIC_ACQUIRE);
}

bool tryLoadCompletion(const Completion& completion, uint64_t expected_sequence,
                       int32_t& status) {
    if (expected_sequence == 0 ||
        loadCompletionSequence(completion) != expected_sequence ||
        completion.reserved != 0) {
        return false;
    }
    status = completion.status;
    return true;
}

}  // namespace pxn
}  // namespace mooncake

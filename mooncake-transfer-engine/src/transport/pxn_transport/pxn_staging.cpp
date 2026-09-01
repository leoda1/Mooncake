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

#include "transport/pxn_transport/pxn_transport.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "config.h"

namespace mooncake {
namespace pxn {

StagingArena::~StagingArena() {
    auto status = shutdown();
    if (!status.ok()) LOG(ERROR) << status.ToString();
}

Status StagingArena::initialize(StagingBackend& backend, PxnGeometry geometry) {
    if (backend_ != nullptr || allocated_ || registered_ || ready_) {
        return Status::InvalidArgument(
            "PXN staging arena is already initialized");
    }
    if (!geometry.valid()) {
        return Status::InvalidArgument("invalid PXN staging geometry");
    }
    geometry_ = geometry;
    backend_ = &backend;

    auto status = backend_->allocateArena(geometry_.arenaSize(), address_);
    if (!status.ok()) {
        backend_ = nullptr;
        address_ = 0;
        return status;
    }
    if (address_ == 0) {
        auto cleanup = backend_->freeArena(address_);
        if (!cleanup.ok()) quarantine_.trip();
        backend_ = nullptr;
        return cleanup.ok()
                   ? Status::Memory("PXN staging backend returned a null arena")
                   : cleanup;
    }
    allocated_ = true;

    status = backend_->registerArena(address_, geometry_.arenaSize());
    if (!status.ok()) {
        quarantine_.trip();
        allocated_ = false;
        backend_ = nullptr;
        return status;
    }
    registered_ = true;

    status = backend_->exportArena(address_, handle_);
    if (!status.ok()) {
        auto cleanup = backend_->unregisterArena(address_);
        if (!cleanup.ok()) {
            quarantine_.trip();
            ready_ = false;
            registered_ = false;
            allocated_ = false;
            backend_ = nullptr;
            return cleanup;
        }
        registered_ = false;
        cleanup = backend_->freeArena(address_);
        allocated_ = false;
        if (!cleanup.ok()) quarantine_.trip();
        if (cleanup.ok()) address_ = 0;
        backend_ = nullptr;
        return cleanup.ok() ? status : cleanup;
    }

    ready_ = true;
    return Status::OK();
}

Status StagingArena::shutdown() {
    if (backend_ == nullptr || quarantine_.quarantined()) return Status::OK();
    ready_ = false;
    if (registered_) {
        auto status = backend_->unregisterArena(address_);
        if (!status.ok()) {
            quarantine_.trip();
            registered_ = false;
            allocated_ = false;
            backend_ = nullptr;
            return status;
        }
        registered_ = false;
    }
    if (allocated_) {
        auto status = backend_->freeArena(address_);
        if (!status.ok()) {
            quarantine_.trip();
            allocated_ = false;
            backend_ = nullptr;
            return status;
        }
        allocated_ = false;
    }
    address_ = 0;
    handle_ = {};
    backend_ = nullptr;
    return Status::OK();
}

void StagingArena::abandon() {
    ready_ = false;
    allocated_ = false;
    registered_ = false;
    quarantine_.trip();
    backend_ = nullptr;
}

std::optional<uintptr_t> StagingArena::laneAddress(size_t lane_index) const {
    if (!ready_ || lane_index >= geometry_.lane_count) return std::nullopt;
    const size_t offset =
        lane_index * geometry_.slots_per_lane * geometry_.slot_size;
    if (address_ > std::numeric_limits<uintptr_t>::max() - offset) {
        return std::nullopt;
    }
    return address_ + offset;
}

std::optional<uintptr_t> StagingArena::slotAddress(size_t lane_index,
                                                   size_t slot_index) const {
    if (!ready_ || lane_index >= geometry_.lane_count ||
        slot_index >= geometry_.slots_per_lane) {
        return std::nullopt;
    }
    const size_t offset = (lane_index * geometry_.slots_per_lane + slot_index) *
                          geometry_.slot_size;
    if (address_ > std::numeric_limits<uintptr_t>::max() - offset) {
        return std::nullopt;
    }
    return address_ + offset;
}

PeerResources::PeerResources(StagingBackend& backend,
                             std::unique_ptr<PeerMapping> mapping,
                             RegistryEntry entry, ProcessIdentity sender,
                             uint64_t sender_epoch, size_t lane_index)
    : backend_(&backend),
      mapping_(std::move(mapping)),
      entry_(std::move(entry)),
      sender_(sender),
      sender_epoch_(sender_epoch),
      lane_index_(lane_index) {}

PeerResources::~PeerResources() {
    auto status = shutdown();
    if (!status.ok()) {
        LOG(ERROR) << status.ToString();
        quarantine();
    }
}

Status PeerResources::shutdown() {
    if (shutdown_ || quarantine_.quarantined()) return Status::OK();
    if (imported_) {
        auto status = backend_->closeImportedArena(address_);
        if (!status.ok()) return status;
        imported_ = false;
        address_ = 0;
    }

    auto status = mapping_->detachArena();
    if (!status.ok()) return status;
    status = mapping_->releaseLane(sender_, sender_epoch_, lane_index_);
    mapping_.reset();
    shutdown_ = true;
    return status;
}

Status PeerResources::importArena(const CudaIpcHandle& handle) {
    auto status = backend_->importArena(handle, address_);
    if (!status.ok()) return status;
    if (address_ == 0) {
        return Status::Memory("PXN staging backend returned a null peer arena");
    }
    imported_ = true;
    return Status::OK();
}

std::optional<uintptr_t> PeerResources::slotAddress(size_t slot_index) const {
    if (!imported_ || mapping_ == nullptr) return std::nullopt;
    const auto& header = mapping_->control()->header;
    if (slot_index >= header.slots_per_lane) return std::nullopt;
    const size_t offset =
        (lane_index_ * header.slots_per_lane + slot_index) * header.slot_size;
    if (address_ > std::numeric_limits<uintptr_t>::max() - offset) {
        return std::nullopt;
    }
    return address_ + offset;
}

LaneControl* PeerResources::laneControl() const {
    if (mapping_ == nullptr ||
        lane_index_ >= mapping_->control()->header.lane_count) {
        return nullptr;
    }
    return &mapping_->control()->lanes[lane_index_];
}

void PeerResources::quarantine() {
    if (!quarantine_.trip()) return;
    (void)mapping_.release();
    backend_ = nullptr;
}

LocalResources::LocalResources(std::unique_ptr<StagingBackend> backend,
                               std::unique_ptr<Registry> registry,
                               std::vector<std::string> local_rails)
    : backend_(std::move(backend)),
      registry_(std::move(registry)),
      local_rails_(std::move(local_rails)) {}

LocalResources::~LocalResources() {
    auto status = shutdown();
    if (!status.ok()) {
        LOG(ERROR) << status.ToString();
        if (!shutdown_) quarantine();
    }
}

Status LocalResources::Create(
    RegistryOptions registry_options, std::span<const std::string> local_hcas,
    const std::unordered_map<std::string, std::string>& rail_map,
    std::unique_ptr<StagingBackend> backend,
    std::unique_ptr<LocalResources>& resources) {
    if (backend == nullptr) {
        return Status::InvalidArgument("missing PXN staging backend");
    }
    const auto& config = globalConfig();
    const PxnGeometry geometry{config.pxn_lane_count, config.pxn_slots_per_lane,
                               config.pxn_slot_size};
    if (!geometry.valid()) {
        return Status::InvalidArgument("invalid PXN staging geometry");
    }

    RailResolver resolver(rail_map);
    std::vector<std::string> local_rails;
    local_rails.reserve(local_hcas.size());
    for (const auto& hca : local_hcas) {
        auto rail = resolver.canonicalize(hca);
        if (rail.empty() || std::find(local_rails.begin(), local_rails.end(),
                                      rail) != local_rails.end()) {
            continue;
        }
        local_rails.push_back(std::move(rail));
    }
    if (local_rails.empty() || local_rails.size() > kMaxRailsPerRank) {
        return Status::InvalidArgument("PXN requires active local rails");
    }

    std::unique_ptr<Registry> registry;
    auto status = Registry::Open(std::move(registry_options), registry);
    if (!status.ok()) return status;

    std::unique_ptr<LocalResources> candidate(new LocalResources(
        std::move(backend), std::move(registry), std::move(local_rails)));
    status = candidate->arena_.initialize(*candidate->backend_, geometry);
    if (!status.ok()) return status;

    status = candidate->registry_->createLocal(candidate->local_rails_,
                                               candidate->arena_.handle(),
                                               candidate->registration_);
    if (!status.ok()) return status;

    status = candidate->registration_->publish();
    if (!status.ok()) return status;

    resources = std::move(candidate);
    return Status::OK();
}

bool LocalResources::ownsRail(std::string_view rail) const {
    return std::find(local_rails_.begin(), local_rails_.end(), rail) !=
           local_rails_.end();
}

Status LocalResources::shutdown() {
    if (shutdown_) return Status::OK();
    Status peer_status;
    for (auto& peer : peers_) {
        auto status = peer->shutdown();
        if (!status.ok()) {
            if (peer_status.ok()) peer_status = status;
            peer->quarantine();
        }
    }
    peers_.clear();

    if (registration_ != nullptr) {
        auto status = registration_->unpublish();
        if (!status.ok()) return status;
        size_t active_attachments = 0;
        status = registration_->activeAttachments(active_attachments);
        if (!status.ok()) return status;
        if (active_attachments != 0) {
            return Status::BatchBusy(
                "PXN staging arena still has peer mappings");
        }
    }

    registration_.reset();

    auto status = arena_.shutdown();
    registry_.reset();
    shutdown_ = true;
    if (!status.ok()) return status;
    return peer_status;
}

Status LocalResources::mapPeer(const RegistryEntry& entry,
                               PeerResources*& peer) {
    peer = nullptr;
    if (shutdown_ || quarantine_.quarantined()) {
        return Status::InvalidArgument("PXN resources are not active");
    }
    for (const auto& existing : peers_) {
        if (existing->entry().identity != entry.identity) continue;
        if (existing->entry().epoch != entry.epoch) {
            return Status::InvalidArgument("PXN peer epoch changed");
        }
        if (existing->address() == 0) {
            return Status::InvalidArgument("PXN peer resources are not active");
        }
        peer = existing.get();
        return Status::OK();
    }

    std::unique_ptr<PeerMapping> mapping;
    auto status = registry_->mapPeer(entry, mapping);
    if (!status.ok()) return status;

    size_t lane_index = 0;
    status = mapping->claimLane(registry_->identity(), registry_->epoch(),
                                lane_index);
    if (!status.ok()) return status;
    status = mapping->attachArena();
    if (!status.ok()) {
        (void)mapping->releaseLane(registry_->identity(), registry_->epoch(),
                                   lane_index);
        return status;
    }

    std::unique_ptr<PeerResources> candidate(new PeerResources(
        *backend_, std::move(mapping), entry, registry_->identity(),
        registry_->epoch(), lane_index));
    status = candidate->importArena(entry.arena_handle);
    if (!status.ok()) return status;
    peer = candidate.get();
    peers_.push_back(std::move(candidate));
    return Status::OK();
}

void LocalResources::quarantine() {
    if (!quarantine_.trip()) return;
    arena_.abandon();
    (void)registration_.release();
    (void)registry_.release();
    (void)backend_.release();
}

}  // namespace pxn
}  // namespace mooncake

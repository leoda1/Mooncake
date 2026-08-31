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

#include "transport/pxn_transport/pxn_rdma_transport.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "config.h"
#include "memory_location.h"
#include "transport/pxn_transport/pxn_nvtx.h"
#include "transport/pxn_transport/pxn_transport.h"
#include "transport/rdma_transport/rdma_context.h"

namespace mooncake {
namespace {

void markPxnPosted(void* context) {
    static_cast<Transport::Slice*>(context)->status = Transport::Slice::POSTED;
}

void markPxnSuccess(void* context) {
    static_cast<Transport::Slice*>(context)->markSuccess();
}

void markPxnFailed(void* context) {
    static_cast<Transport::Slice*>(context)->markFailed();
}

bool covers(const TransferMetadata::BufferDesc& buffer, uint64_t address,
            size_t length) {
    return address >= buffer.addr && length <= buffer.length &&
           address - buffer.addr <= buffer.length - length;
}

bool covers(uint64_t begin, uint64_t end, uint64_t address, size_t length) {
    return begin < end && address >= begin && address < end &&
           length <= end - begin && address - begin <= end - begin - length;
}

std::vector<std::string> canonicalRails(const Topology& topology,
                                        std::string_view location,
                                        const pxn::RailResolver& resolver) {
    const auto matrix = topology.getMatrix();
    const std::vector<std::string>* candidates = nullptr;
    auto entry = matrix.find(std::string(location));
    if (entry != matrix.end()) {
        if (!entry->second.preferred_hca.empty()) {
            candidates = &entry->second.preferred_hca;
        } else if (!entry->second.avail_hca.empty()) {
            candidates = &entry->second.avail_hca;
        }
    }
    const auto& wildcard_candidates = topology.getHcaList();
    if (candidates == nullptr) candidates = &wildcard_candidates;
    std::vector<std::string> rails;
    for (const auto& candidate : *candidates) {
        auto rail = resolver.canonicalize(candidate);
        if (rail.empty() ||
            std::find(rails.begin(), rails.end(), rail) != rails.end()) {
            continue;
        }
        rails.push_back(std::move(rail));
    }
    return rails;
}

}  // namespace

struct PxnRdmaTransport::SelectionContext {
    struct TargetRoute {
        uint64_t begin = 0;
        uint64_t end = 0;
        std::string rail;
        bool same_rail = false;
    };

    struct TargetEntry {
        struct RouteSet {
            uint64_t begin = 0;
            uint64_t end = 0;
            std::vector<TargetRoute> routes;
            size_t next = 0;
        };

        bool loaded = false;
        std::shared_ptr<TransferMetadata::SegmentDesc> descriptor;
        std::vector<RouteSet> route_sets;
    };

    bool local_loaded = false;
    std::shared_ptr<TransferMetadata::SegmentDesc> local_descriptor;
    std::vector<std::pair<uint64_t, uint64_t>> cuda_source_ranges;
    std::unordered_map<SegmentID, TargetEntry> targets;
    std::unordered_map<std::string, RelayLane> relays;

    uint64_t not_write = 0;
    uint64_t not_cuda = 0;
    uint64_t no_target = 0;
    uint64_t no_device = 0;
    uint64_t same_rail = 0;
    uint64_t no_relay = 0;
    uint64_t route_cache_hit = 0;
    uint64_t route_cache_miss = 0;
    std::string first_same_rail;
    std::string first_missing_relay;
    std::string first_registry_rails;
};

PxnRdmaTransport::PxnRdmaTransport() = default;

PxnRdmaTransport::~PxnRdmaTransport() {
    pump_.reset();
    relay_pipeline_.reset();
    sender_pipeline_.reset();
    resources_.reset();
    rail_resolver_.reset();
}

int PxnRdmaTransport::install(std::string& local_server_name,
                              std::shared_ptr<TransferMetadata> metadata,
                              std::shared_ptr<Topology> topology) {
    auto pxn_topology = topology;
    const int result = RdmaTransport::install(
        local_server_name, std::move(metadata), std::move(topology));
    if (result != 0 || !globalConfig().pxn_enable) return result;

    std::vector<std::string> active_hcas;
    for (const auto& context : getContextList()) {
        if (context != nullptr && context->active()) {
            active_hcas.push_back(context->deviceName());
        }
    }

    int device_id = -1;
    auto cuda_error = cudaGetDevice(&device_id);
    if (cuda_error != cudaSuccess) {
        LOG(WARNING) << "PXN is disabled: cudaGetDevice failed: "
                     << cudaGetErrorString(cuda_error);
        return result;
    }

    const std::string location = "cuda:" + std::to_string(device_id);
    bool preferred_resolved = active_hcas.size() == 1;
    if (active_hcas.size() > 1 && pxn_topology != nullptr) {
        const auto matrix = pxn_topology->getMatrix();
        const auto entry = matrix.find(location);
        if (entry != matrix.end() && !entry->second.preferred_hca.empty()) {
            const auto& candidates = entry->second.preferred_hca;
            std::vector<std::string> local_hcas;
            for (const auto& candidate : candidates) {
                if (std::find(active_hcas.begin(), active_hcas.end(),
                              candidate) != active_hcas.end()) {
                    local_hcas.push_back(candidate);
                }
            }
            if (!local_hcas.empty()) {
                active_hcas = std::move(local_hcas);
                preferred_resolved = true;
            }
        }
    }
    if (active_hcas.empty() || !preferred_resolved) {
        LOG(WARNING) << "PXN is disabled: cannot resolve preferred HCAs for "
                     << location;
        return result;
    }

    rail_resolver_ =
        std::make_unique<pxn::RailResolver>(globalConfig().pxn_rail_map);
    std::vector<std::string> source_hcas;
    std::vector<std::string> source_rails;
    for (const auto& hca : active_hcas) {
        auto rail = rail_resolver_->canonicalize(hca);
        if (std::find(source_rails.begin(), source_rails.end(), rail) !=
            source_rails.end()) {
            continue;
        }
        source_hcas.push_back(hca);
        source_rails.push_back(std::move(rail));
    }
    active_hcas = std::move(source_hcas);
    std::string local_rail_log;
    for (size_t index = 0; index < active_hcas.size(); ++index) {
        if (!local_rail_log.empty()) local_rail_log += ", ";
        local_rail_log += active_hcas[index] + "=" + source_rails[index];
    }
    LOG(INFO) << "PXN local rails for " << location << " are ["
              << local_rail_log << "]";
    pxn::RegistryOptions registry_options;
    registry_options.group_id = globalConfig().pxn_group_id;
    auto backend = pxn::makeCudaRdmaStagingBackend(*this, device_id);
    auto status = pxn::LocalResources::Create(
        std::move(registry_options), active_hcas, globalConfig().pxn_rail_map,
        std::move(backend), resources_);
    if (!status.ok()) {
        LOG(WARNING) << "PXN is disabled: " << status.ToString();
        return result;
    }
    std::unique_ptr<pxn::RelayBackend> relay_backend;
    status = pxn::makeRdmaRelayBackend(*this, resources_->arena().address(),
                                       active_hcas, resources_->localRails(),
                                       globalConfig().pxn_inflight_depth,
                                       relay_backend);
    if (!status.ok()) {
        LOG(WARNING) << "PXN is disabled: " << status.ToString();
        resources_.reset();
        return result;
    }
    sender_pipeline_ = std::make_unique<pxn::SenderPipeline>(
        pxn::makeCudaSenderBackend(device_id),
        pxn::makeRdmaSenderFallback(*this),
        std::chrono::milliseconds(globalConfig().pxn_credit_timeout_ms));
    relay_pipeline_ = std::make_unique<pxn::RelayPipeline>(
        resources_->registration().control(), resources_->arena().address(),
        resources_->registry().epoch(), globalConfig().pxn_inflight_depth,
        std::move(relay_backend));
    pump_ = std::make_unique<pxn::PxnPump>(*sender_pipeline_, *relay_pipeline_);
    return result;
}

Status PxnRdmaTransport::submitTransferTask(
    const std::vector<TransferTask*>& task_list) {
    PXN_NVTX_SENDER("pxn::submitTransferTask");
    if (!pxnReady()) return RdmaTransport::submitTransferTask(task_list);

    struct Group {
        pxn::SenderLane* lane;
        uint32_t rail_index;
        SegmentID target_id;
        std::string session;
        std::vector<TransferTask*> tasks;
        std::vector<pxn::TransferSpan> spans;
    };
    struct Pending {
        pxn::SenderLane* lane;
        pxn::SenderSubmission submission;
    };

    std::vector<TransferTask*> direct_tasks;
    std::vector<Group> groups;
    SelectionContext selection;
    for (auto* task : task_list) {
        std::string session;
        pxn::SenderLane* lane = nullptr;
        uint32_t rail_index = 0;
        const auto& request = *task->request;
        if (!selectPxnLane(request, selection, session, lane, rail_index)) {
            direct_tasks.push_back(task);
            continue;
        }

        size_t index = 0;
        while (index < groups.size() &&
               (groups[index].target_id != request.target_id ||
                groups[index].lane != lane ||
                groups[index].rail_index != rail_index)) {
            ++index;
        }
        if (index == groups.size()) {
            groups.push_back({lane,
                              rail_index,
                              request.target_id,
                              std::move(session),
                              {},
                              {}});
        }
        auto& group = groups[index];
        const size_t request_index = group.tasks.size();
        group.tasks.push_back(task);
        group.spans.push_back({reinterpret_cast<uint64_t>(request.source),
                               request.target_offset, request.length,
                               request_index});
    }

    std::vector<Pending> pending;
    for (auto& group : groups) {
        std::vector<pxn::Piece> pieces;
        auto status = pxn::buildPieces(group.spans, pieces);
        if (!status.ok()) {
            direct_tasks.insert(direct_tasks.end(), group.tasks.begin(),
                                group.tasks.end());
            continue;
        }

        for (auto& piece : pieces) {
            pxn::SenderSubmission submission;
            submission.session = group.session;
            submission.target_id = group.target_id;
            submission.rail_index = group.rail_index;
            submission.slices.reserve(piece.spans.size());
            for (const auto& span : piece.spans) {
                auto* task = group.tasks[span.request_index];
                auto* slice = getSliceCache().allocate();
                slice->source_addr = reinterpret_cast<void*>(span.source);
                slice->length = static_cast<size_t>(span.length);
                slice->opcode = TransferRequest::WRITE;
                slice->target_id = group.target_id;
                slice->peer_nic_path.clear();
                slice->source_location.clear();
                slice->status = Slice::PENDING;
                slice->task = task;
                slice->cleanup_callback = nullptr;
                slice->ts = 0;
                task->slice_list.push_back(slice);
                task->total_bytes += slice->length;
                __sync_fetch_and_add(&task->slice_count, 1);
                submission.slices.push_back({slice, slice->length,
                                             markPxnPosted, markPxnSuccess,
                                             markPxnFailed});
            }
            submission.piece = std::move(piece);
            pending.push_back({group.lane, std::move(submission)});
        }
    }

    uint64_t direct_bytes = 0;
    for (const auto* task : direct_tasks) direct_bytes += task->request->length;
    uint64_t pxn_used = 0;
    uint64_t pxn_bytes = 0;
    for (const auto& item : pending) {
        ++pxn_used;
        pxn_bytes += item.submission.piece.length;
    }

    auto add_stat = [](std::atomic<uint64_t>& counter, uint64_t value) {
        if (value != 0) counter.fetch_add(value, std::memory_order_relaxed);
    };
    add_stat(pxn_stats_.not_write, selection.not_write);
    add_stat(pxn_stats_.not_cuda, selection.not_cuda);
    add_stat(pxn_stats_.no_target, selection.no_target);
    add_stat(pxn_stats_.no_device, selection.no_device);
    if (selection.same_rail != 0) {
        const uint64_t previous = pxn_stats_.same_rail.fetch_add(
            selection.same_rail, std::memory_order_relaxed);
        if (previous == 0) {
            LOG(WARNING) << "PXN skipped: target rail "
                         << selection.first_same_rail
                         << " is owned by the sender rank"
                         << " (same rail always uses native RDMA)";
        }
    }
    if (selection.no_relay != 0) {
        const uint64_t previous = pxn_stats_.no_relay.fetch_add(
            selection.no_relay, std::memory_order_relaxed);
        if (previous == 0) {
            LOG(WARNING) << "PXN skipped: no relay on rail "
                         << selection.first_missing_relay << "; registry has ["
                         << selection.first_registry_rails
                         << "] -- a same-machine process must own a NIC on the "
                            "target rail";
        }
    }
    add_stat(pxn_stats_.direct_bytes, direct_bytes);
    add_stat(pxn_stats_.pxn_used, pxn_used);
    add_stat(pxn_stats_.pxn_bytes, pxn_bytes);
    add_stat(pxn_stats_.route_cache_hit, selection.route_cache_hit);
    add_stat(pxn_stats_.route_cache_miss, selection.route_cache_miss);

    Status result;
    if (!direct_tasks.empty()) {
        result = RdmaTransport::submitTransferTask(direct_tasks);
    }
    for (auto& item : pending) {
        const uint64_t bytes = item.submission.piece.length;
        const uint64_t source_spans = item.submission.piece.spans.size();
        auto status = item.lane->enqueue(std::move(item.submission));
        if (result.ok() && !status.ok()) result = status;
        if (!status.ok()) continue;
        add_stat(pxn_stats_.enqueued_bytes, bytes);
        add_stat(pxn_stats_.enqueued_pieces, 1);
        add_stat(pxn_stats_.source_spans, source_spans);
        auto update_max = [](std::atomic<uint64_t>& counter, uint64_t value) {
            uint64_t current = counter.load(std::memory_order_relaxed);
            while (current < value &&
                   !counter.compare_exchange_weak(current, value,
                                                  std::memory_order_relaxed)) {
            }
        };
        update_max(pxn_stats_.max_source_spans, source_spans);
    }
    if (!pending.empty()) pump_->wake();
    reportPxnStats();
    return result;
}

bool PxnRdmaTransport::selectPxnLane(const TransferRequest& request,
                                     SelectionContext& context,
                                     std::string& session,
                                     pxn::SenderLane*& lane,
                                     uint32_t& rail_index) {
    if (request.opcode != TransferRequest::WRITE || request.length == 0) {
        ++context.not_write;
        return false;
    }

    auto& target_entry = context.targets[request.target_id];
    if (!target_entry.loaded) {
        target_entry.descriptor =
            metadata_->getSegmentDescByID(request.target_id);
        target_entry.loaded = true;
    }
    const auto& target = target_entry.descriptor;
    if (target == nullptr || target->name.empty() ||
        target->name.size() > pxn::kMaxSessionLength) {
        ++context.no_target;
        return false;
    }

    SelectionContext::TargetRoute transient_route;
    const SelectionContext::TargetRoute* target_route = nullptr;
    for (auto& cached : target_entry.route_sets) {
        if (covers(cached.begin, cached.end, request.target_offset,
                   request.length) &&
            !cached.routes.empty()) {
            target_route = &cached.routes[cached.next % cached.routes.size()];
            cached.next = (cached.next + 1) % cached.routes.size();
            ++context.route_cache_hit;
            break;
        }
    }

    if (target_route == nullptr) {
        ++context.route_cache_miss;
        int buffer_id = -1;
        int device_id = -1;
        if (RdmaTransport::selectDevice(target.get(), request.target_offset,
                                        request.length, buffer_id,
                                        device_id) != 0) {
            ++context.no_device;
            return false;
        }

        const auto& buffer = target->buffers[buffer_id];
        uint64_t route_begin = request.target_offset;
        uint64_t route_end = request.target_offset;
        if (request.target_offset <=
            std::numeric_limits<uint64_t>::max() - request.length) {
            route_end += request.length;
        }
        SegmentsLocationInfo segments;
        const bool segmented = parseSegmentsLocation(buffer.name, segments);
        const std::string location =
            segmented
                ? resolveSegmentsLocation(segments, buffer.length,
                                          request.target_offset - buffer.addr)
                : buffer.name;
        const std::string target_rail =
            rail_resolver_->canonicalize(target->devices[device_id].name);
        const bool cacheable =
            !segmented &&
            buffer.addr <= std::numeric_limits<uint64_t>::max() - buffer.length;
        if (cacheable) {
            route_begin = buffer.addr;
            route_end = buffer.addr + buffer.length;
        }
        transient_route = {route_begin, route_end, target_rail,
                           resources_->ownsRail(target_rail)};
        if (cacheable) {
            auto rails =
                canonicalRails(target->topology, location, *rail_resolver_);
            auto selected = std::find(rails.begin(), rails.end(), target_rail);
            if (selected != rails.end()) {
                std::rotate(rails.begin(), selected, rails.end());
            } else {
                rails.insert(rails.begin(), target_rail);
            }

            SelectionContext::TargetEntry::RouteSet route_set;
            route_set.begin = route_begin;
            route_set.end = route_end;
            route_set.routes.reserve(rails.size());
            for (auto& rail : rails) {
                route_set.routes.push_back(
                    {route_begin, route_end, rail, resources_->ownsRail(rail)});
            }
            route_set.next = route_set.routes.size() > 1 ? 1 : 0;
            target_entry.route_sets.push_back(std::move(route_set));
            target_route = &target_entry.route_sets.back().routes.front();
        } else {
            target_route = &transient_route;
        }
    }

    if (target_route->same_rail) {
        ++context.same_rail;
        if (context.first_same_rail.empty()) {
            context.first_same_rail = target_route->rail;
        }
        return false;
    }

    if (!context.local_loaded) {
        context.local_descriptor =
            metadata_->getSegmentDescByID(LOCAL_SEGMENT_ID);
        context.local_loaded = true;
    }
    const uint64_t source = reinterpret_cast<uint64_t>(request.source);
    bool cuda_source = false;
    for (const auto& range : context.cuda_source_ranges) {
        if (covers(range.first, range.second, source, request.length)) {
            cuda_source = true;
            break;
        }
    }
    if (!cuda_source && context.local_descriptor != nullptr) {
        for (const auto& buffer : context.local_descriptor->buffers) {
            if (!covers(buffer, source, request.length) ||
                buffer.name.rfind("cuda:", 0) != 0) {
                continue;
            }
            cuda_source = true;
            if (buffer.addr <=
                std::numeric_limits<uint64_t>::max() - buffer.length) {
                context.cuda_source_ranges.push_back(
                    {buffer.addr, buffer.addr + buffer.length});
            }
            break;
        }
    }
    if (!cuda_source) {
        ++context.not_cuda;
        return false;
    }

    auto resolved = context.relays.find(target_route->rail);
    if (resolved != context.relays.end()) {
        if (resolved->second.lane == nullptr) {
            ++context.no_relay;
            return false;
        }
        lane = resolved->second.lane;
        rail_index = resolved->second.rail_index;
        session = target->name;
        return true;
    }

    std::lock_guard<std::mutex> lock(peer_mutex_);
    auto cached = lanes_by_rail_.find(target_route->rail);
    if (cached != lanes_by_rail_.end()) {
        lane = cached->second.lane;
        rail_index = cached->second.rail_index;
        context.relays[target_route->rail] = cached->second;
        session = target->name;
        return true;
    }
    std::vector<pxn::RegistryEntry> entries;
    if (!resources_->registry().discover(entries).ok()) {
        ++context.no_relay;
        context.relays[target_route->rail] = {};
        if (context.first_missing_relay.empty()) {
            context.first_missing_relay = target_route->rail;
        }
        return false;
    }
    for (const auto& entry : entries) {
        const auto rail = std::find(entry.rails.begin(), entry.rails.end(),
                                    target_route->rail);
        if (rail == entry.rails.end()) continue;
        pxn::PeerResources* peer = nullptr;
        if (!resources_->mapPeer(entry, peer).ok()) return false;
        if (!sender_pipeline_->addPeer(*peer, lane).ok()) return false;
        for (size_t index = 0; index < entry.rails.size(); ++index) {
            lanes_by_rail_[entry.rails[index]] = {lane,
                                                  static_cast<uint32_t>(index)};
        }
        rail_index = static_cast<uint32_t>(rail - entry.rails.begin());
        context.relays[target_route->rail] = {lane, rail_index};
        session = target->name;
        return true;
    }
    ++context.no_relay;
    context.relays[target_route->rail] = {};
    if (context.first_missing_relay.empty()) {
        context.first_missing_relay = target_route->rail;
        std::string rails;
        for (const auto& entry : entries) {
            for (const auto& rail : entry.rails) rails += rail + " ";
        }
        context.first_registry_rails = std::move(rails);
    }
    return false;
}

void PxnRdmaTransport::reportPxnStats() {
    const uint64_t n = pxn_stats_.reported.fetch_add(1) + 1;
    if (n % 200 != 0) return;
    const uint64_t enqueued_pieces = pxn_stats_.enqueued_pieces.load();
    const uint64_t source_spans = pxn_stats_.source_spans.load();
    LOG(INFO)
        << "PXN stats: used=" << pxn_stats_.pxn_used.load()
        << " pxn_bytes=" << pxn_stats_.pxn_bytes.load()
        << " direct_bytes=" << pxn_stats_.direct_bytes.load()
        << " | pipeline: tx_enqueued_bytes=" << pxn_stats_.enqueued_bytes.load()
        << " tx_pieces=" << enqueued_pieces
        << " relay_ready_bytes=" << relay_pipeline_->readyBytes()
        << " relay_ready_pieces=" << relay_pipeline_->readyPieces()
        << " relay_submitted_bytes=" << relay_pipeline_->submittedBytes()
        << " relay_submitted_pieces=" << relay_pipeline_->submittedPieces()
        << " relay_completed_bytes=" << relay_pipeline_->relayedBytes()
        << " relay_completed_pieces=" << relay_pipeline_->relayedPieces()
        << " | shape: copy_spans_total=" << source_spans
        << " copy_spans_avg_x100="
        << (enqueued_pieces == 0 ? 0 : source_spans * 100 / enqueued_pieces)
        << " copy_spans_max=" << pxn_stats_.max_source_spans.load()
        << " | skipped: same_rail=" << pxn_stats_.same_rail.load()
        << " no_relay=" << pxn_stats_.no_relay.load()
        << " not_cuda=" << pxn_stats_.not_cuda.load()
        << " no_device=" << pxn_stats_.no_device.load()
        << " no_target=" << pxn_stats_.no_target.load()
        << " not_write=" << pxn_stats_.not_write.load()
        << " | route_cache: hit=" << pxn_stats_.route_cache_hit.load()
        << " miss=" << pxn_stats_.route_cache_miss.load();

    LOG(INFO) << "PXN pipeline: fallback_bytes="
              << sender_pipeline_->fallbackBytes();
}

}  // namespace mooncake

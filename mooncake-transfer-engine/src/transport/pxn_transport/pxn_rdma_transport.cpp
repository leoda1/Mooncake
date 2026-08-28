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

#include <chrono>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "config.h"
#include "memory_location.h"
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

bool hasUniqueCanonicalRail(const Topology& topology, std::string_view location,
                            const pxn::RailResolver& resolver,
                            std::string_view expected_rail) {
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
    if (candidates->empty()) return false;
    for (const auto& candidate : *candidates) {
        if (resolver.canonicalize(candidate) != expected_rail) return false;
    }
    return true;
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
        bool loaded = false;
        std::shared_ptr<TransferMetadata::SegmentDesc> descriptor;
        std::vector<TargetRoute> routes;
    };

    bool local_loaded = false;
    std::shared_ptr<TransferMetadata::SegmentDesc> local_descriptor;
    std::vector<std::pair<uint64_t, uint64_t>> cuda_source_ranges;
    std::unordered_map<SegmentID, TargetEntry> targets;
    std::unordered_map<std::string, pxn::SenderLane*> relays;

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

    if (active_hcas.size() > 1 && pxn_topology != nullptr) {
        const std::string location = "cuda:" + std::to_string(device_id);
        const int preferred = pxn_topology->selectDevice(location);
        if (preferred >= 0 &&
            static_cast<size_t>(preferred) < getContextList().size()) {
            const auto& context = getContextList()[preferred];
            if (context != nullptr && context->active()) {
                LOG(INFO) << "PXN local rail for " << location << " is "
                          << context->deviceName() << " (of "
                          << active_hcas.size() << " active HCAs)";
                active_hcas = {context->deviceName()};
            }
        }
        if (active_hcas.size() > 1) {
            LOG(WARNING) << "PXN is disabled: cannot pick a local rail for "
                         << location << " among " << active_hcas.size()
                         << " active HCAs";
            return result;
        }
    }
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
                                       globalConfig().pxn_inflight_depth,
                                       relay_backend);
    if (!status.ok()) {
        LOG(WARNING) << "PXN is disabled: " << status.ToString();
        resources_.reset();
        return result;
    }
    rail_resolver_ =
        std::make_unique<pxn::RailResolver>(globalConfig().pxn_rail_map);
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
    if (!pxnReady()) return RdmaTransport::submitTransferTask(task_list);

    struct Group {
        pxn::SenderLane* lane;
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
        const auto& request = *task->request;
        if (!selectPxnLane(request, selection, session, lane)) {
            direct_tasks.push_back(task);
            continue;
        }

        size_t index = 0;
        while (index < groups.size() &&
               (groups[index].target_id != request.target_id ||
                groups[index].lane != lane)) {
            ++index;
        }
        if (index == groups.size()) {
            groups.push_back(
                {lane, request.target_id, std::move(session), {}, {}});
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
                         << selection.first_same_rail << " == local rail "
                         << resources_->localRail()
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
    reportPxnStats();

    Status result;
    if (!direct_tasks.empty()) {
        result = RdmaTransport::submitTransferTask(direct_tasks);
    }
    for (auto& item : pending) {
        auto status = item.lane->enqueue(std::move(item.submission));
        if (result.ok() && !status.ok()) result = status;
    }
    if (!pending.empty()) pump_->wake();
    return result;
}

bool PxnRdmaTransport::selectPxnLane(const TransferRequest& request,
                                     SelectionContext& context,
                                     std::string& session,
                                     pxn::SenderLane*& lane) {
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
    for (const auto& cached : target_entry.routes) {
        if (covers(cached.begin, cached.end, request.target_offset,
                   request.length)) {
            target_route = &cached;
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
        // Pin the whole buffer only when doing so preserves selectDevice()'s
        // multi-HCA routing. Segmented or multi-rail buffers stay uncached.
        const bool cacheable =
            !segmented &&
            hasUniqueCanonicalRail(target->topology, location, *rail_resolver_,
                                   target_rail) &&
            buffer.addr <= std::numeric_limits<uint64_t>::max() - buffer.length;
        if (cacheable) {
            route_begin = buffer.addr;
            route_end = buffer.addr + buffer.length;
        }
        transient_route = {route_begin, route_end, target_rail,
                           target_rail == resources_->localRail()};
        if (cacheable) {
            target_entry.routes.push_back(transient_route);
            target_route = &target_entry.routes.back();
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
        if (resolved->second == nullptr) {
            ++context.no_relay;
            return false;
        }
        lane = resolved->second;
        session = target->name;
        return true;
    }

    std::lock_guard<std::mutex> lock(peer_mutex_);
    auto cached = lanes_by_rail_.find(target_route->rail);
    if (cached != lanes_by_rail_.end()) {
        lane = cached->second;
        context.relays[target_route->rail] = lane;
        session = target->name;
        return true;
    }
    std::vector<pxn::RegistryEntry> entries;
    if (!resources_->registry().discover(entries).ok()) {
        ++context.no_relay;
        context.relays[target_route->rail] = nullptr;
        if (context.first_missing_relay.empty()) {
            context.first_missing_relay = target_route->rail;
        }
        return false;
    }
    for (const auto& entry : entries) {
        if (entry.rail != target_route->rail) continue;
        pxn::PeerResources* peer = nullptr;
        if (!resources_->mapPeer(entry, peer).ok()) return false;
        if (!sender_pipeline_->addPeer(*peer, lane).ok()) return false;
        lanes_by_rail_[target_route->rail] = lane;
        context.relays[target_route->rail] = lane;
        session = target->name;
        return true;
    }
    ++context.no_relay;
    context.relays[target_route->rail] = nullptr;
    if (context.first_missing_relay.empty()) {
        context.first_missing_relay = target_route->rail;
        std::string rails;
        for (const auto& e : entries) rails += e.rail + " ";
        context.first_registry_rails = std::move(rails);
    }
    return false;
}

void PxnRdmaTransport::reportPxnStats() {
    const uint64_t n = pxn_stats_.reported.fetch_add(1) + 1;
    if (n % 200 != 0) return;
    LOG(INFO) << "PXN stats: used=" << pxn_stats_.pxn_used.load()
              << " pxn_bytes=" << pxn_stats_.pxn_bytes.load()
              << " direct_bytes=" << pxn_stats_.direct_bytes.load()
              << " | skipped: same_rail=" << pxn_stats_.same_rail.load()
              << " no_relay=" << pxn_stats_.no_relay.load()
              << " not_cuda=" << pxn_stats_.not_cuda.load()
              << " no_device=" << pxn_stats_.no_device.load()
              << " no_target=" << pxn_stats_.no_target.load()
              << " not_write=" << pxn_stats_.not_write.load()
              << " | route_cache: hit=" << pxn_stats_.route_cache_hit.load()
              << " miss=" << pxn_stats_.route_cache_miss.load();
}

}  // namespace mooncake

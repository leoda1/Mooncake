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

#include "transport/pxn/pxn_rdma_transport.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include "config.h"
#include "transport/pxn/pxn_transport.h"
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

}  // namespace

PxnRdmaTransport::~PxnRdmaTransport() {
    pump_.reset();
    relay_pipeline_.reset();
    sender_pipeline_.reset();
    resources_.reset();
}

int PxnRdmaTransport::install(std::string& local_server_name,
                              std::shared_ptr<TransferMetadata> metadata,
                              std::shared_ptr<Topology> topology) {
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
    for (auto* task : task_list) {
        std::string session;
        pxn::SenderLane* lane = nullptr;
        const auto& request = *task->request;
        if (!selectPxnLane(request, session, lane)) {
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
                                     std::string& session,
                                     pxn::SenderLane*& lane) {
    if (request.opcode != TransferRequest::WRITE || request.length == 0) {
        return false;
    }

    auto local = metadata_->getSegmentDescByID(LOCAL_SEGMENT_ID);
    const uint64_t source = reinterpret_cast<uint64_t>(request.source);
    bool cuda_source = false;
    for (const auto& buffer : local->buffers) {
        if (covers(buffer, source, request.length) &&
            buffer.name.rfind("cuda:", 0) == 0) {
            cuda_source = true;
            break;
        }
    }
    if (!cuda_source) return false;

    auto target = metadata_->getSegmentDescByID(request.target_id);
    if (target == nullptr || target->name.empty() ||
        target->name.size() > pxn::kMaxSessionLength) {
        return false;
    }

    int buffer_id = -1;
    int device_id = -1;
    if (RdmaTransport::selectDevice(target.get(), request.target_offset,
                                    request.length, buffer_id,
                                    device_id) != 0) {
        return false;
    }
    pxn::RailResolver resolver(globalConfig().pxn_rail_map);
    const std::string target_rail =
        resolver.canonicalize(target->devices[device_id].name);
    if (target_rail == resources_->localRail()) return false;

    std::lock_guard<std::mutex> lock(peer_mutex_);
    auto cached = lanes_by_rail_.find(target_rail);
    if (cached != lanes_by_rail_.end()) {
        lane = cached->second;
        session = target->name;
        return true;
    }
    std::vector<pxn::RegistryEntry> entries;
    if (!resources_->registry().discover(entries).ok()) return false;
    for (const auto& entry : entries) {
        if (entry.rail != target_rail) continue;
        pxn::PeerResources* peer = nullptr;
        if (!resources_->mapPeer(entry, peer).ok()) return false;
        if (!sender_pipeline_->addPeer(*peer, lane).ok()) return false;
        lanes_by_rail_[target_rail] = lane;
        session = target->name;
        return true;
    }
    return false;
}

}  // namespace mooncake

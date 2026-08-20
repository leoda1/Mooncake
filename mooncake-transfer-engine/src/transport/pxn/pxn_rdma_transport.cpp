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
namespace {  // # 新增:匿名命名空间,存放 PXN 提交路径用到的辅助回调

void markPxnPosted(void* context) {  // # 新增:slice 提交到 lane 时置为 POSTED
    static_cast<Transport::Slice*>(context)->status = Transport::Slice::POSTED;  // # 直接改状态位,不触发回调
}

void markPxnSuccess(void* context) {  // # 新增:slice 成功完成后调用
    static_cast<Transport::Slice*>(context)->markSuccess();  // # 复用 Slice 自带成功处理
}

void markPxnFailed(void* context) {  // # 新增:slice 失败后调用
    static_cast<Transport::Slice*>(context)->markFailed();  // # 复用 Slice 自带失败处理
}

bool covers(const TransferMetadata::BufferDesc& buffer, uint64_t address,
            size_t length) {  // # 新增:判断 [address, address+length) 是否完整落在 buffer 内
    return address >= buffer.addr && length <= buffer.length &&  // # 起点不越界且长度不超
           address - buffer.addr <= buffer.length - length;  // # 等价于 address+length <= addr+length,避免溢出
}

}  // namespace  // # 新增:匿名命名空间结束

PxnRdmaTransport::~PxnRdmaTransport() {
    pump_.reset();  // # 新增:先停 pump 线程,避免再触发 relay/sender
    relay_pipeline_.reset();  // # 新增:释放 relay 流水线(会 abandon 在途传输)
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
    sender_pipeline_ = std::make_unique<pxn::SenderPipeline>(
        pxn::makeCudaSenderBackend(device_id),
        pxn::makeRdmaSenderFallback(*this),
        std::chrono::milliseconds(globalConfig().pxn_credit_timeout_ms));
    relay_pipeline_ = std::make_unique<pxn::RelayPipeline>(  // # 新增:创建 relay 流水线,负责远端转发写入
        resources_->registration().control(),  // # 共享控制块,读取远端 doorbell/描述符
        resources_->arena().address(),  // # 共享 arena 基址,定位接收槽
        resources_->registry().epoch(),  // # 当前 epoch,校验描述符有效性
        globalConfig().pxn_inflight_depth,  // # 最大在途转发数
        pxn::makeRdmaRelayBackend(*this));  // # relay 后端:转发用普通 RDMA 写
    pump_ = std::make_unique<pxn::PxnPump>(*sender_pipeline_, *relay_pipeline_);  // # 新增:后台 pump 线程统一推进收发与 relay
    return result;
}

Status PxnRdmaTransport::submitTransferTask(  // # 新增:override 提交入口,把可走 PXN 的写请求组包,其余回落普通 RDMA
    const std::vector<TransferTask*>& task_list) {
    if (!pxnReady()) return RdmaTransport::submitTransferTask(task_list);  // # PXN 未就绪时全部走普通 RDMA

    struct Group {  // # 新增:按 (目标段, lane) 聚合的一组任务
        pxn::SenderLane* lane;  // # 该组使用的发送 lane
        SegmentID target_id;  // # 目标段 ID
        std::string session;  // # 目标段会话名
        std::vector<TransferTask*> tasks;  // # 组内原始任务
        std::vector<pxn::TransferSpan> spans;  // # 组内 span 列表
    };
    struct Pending {  // # 新增:待入队的提交项
        pxn::SenderLane* lane;  // # 目标 lane
        pxn::SenderSubmission submission;  // # 组装好的提交
    };

    std::vector<TransferTask*> direct_tasks;  // # 不走 PXN 的任务
    std::vector<Group> groups;  // # 分组结果
    for (auto* task : task_list) {  // # 遍历所有任务
        std::string session;
        pxn::SenderLane* lane = nullptr;
        const auto& request = *task->request;
        if (!selectPxnLane(request, session, lane)) {  // # 该请求不满足 PXN 条件
            direct_tasks.push_back(task);  // # 直接交给普通 RDMA
            continue;
        }

        size_t index = 0;  // # 在已有分组中查找同 (target_id, lane)
        while (index < groups.size() &&
               (groups[index].target_id != request.target_id ||
                groups[index].lane != lane)) {
            ++index;
        }
        if (index == groups.size()) {  // # 没找到则新建一个分组
            groups.push_back(
                {lane, request.target_id, std::move(session), {}, {}});
        }
        auto& group = groups[index];
        const size_t request_index = group.tasks.size();  // # 记录任务在组内的下标
        group.tasks.push_back(task);
        group.spans.push_back({reinterpret_cast<uint64_t>(request.source),  // # 源地址
                               request.target_offset,  // # 目标偏移
                               request.length,  // # 传输长度
                               request_index});  // # 对应的任务下标
    }

    std::vector<Pending> pending;
    for (auto& group : groups) {  // # 对每个分组构建 piece
        std::vector<pxn::Piece> pieces;
        auto status = pxn::buildPieces(group.spans, pieces);  // # 按连续区间把 span 切成 piece
        if (!status.ok()) {  // # 切分失败则整组回落普通 RDMA
            direct_tasks.insert(direct_tasks.end(), group.tasks.begin(),
                                group.tasks.end());
            continue;
        }

        for (auto& piece : pieces) {  // # 每个 piece 生成一个 SenderSubmission
            pxn::SenderSubmission submission;
            submission.session = group.session;
            submission.target_id = group.target_id;
            submission.slices.reserve(piece.spans.size());
            for (const auto& span : piece.spans) {  // # 为每个 span 分配一个 slice
                auto* task = group.tasks[span.request_index];
                auto* slice = getSliceCache().allocate();  // # 从 slice 缓存池取对象
                slice->source_addr = reinterpret_cast<void*>(span.source);
                slice->length = static_cast<size_t>(span.length);
                slice->opcode = TransferRequest::WRITE;
                slice->target_id = group.target_id;
                slice->peer_nic_path.clear();
                slice->source_location.clear();
                slice->status = Slice::PENDING;
                slice->task = task;  // # 关联回原始任务
                slice->cleanup_callback = nullptr;
                slice->ts = 0;
                task->slice_list.push_back(slice);  // # 挂到任务的 slice 列表
                task->total_bytes += slice->length;
                __sync_fetch_and_add(&task->slice_count, 1);  // # 原子递增未完成计数
                submission.slices.push_back({slice, slice->length,  // # 注册三个回调
                                             markPxnPosted, markPxnSuccess,
                                             markPxnFailed});
            }
            submission.piece = std::move(piece);
            pending.push_back({group.lane, std::move(submission)});
        }
    }

    Status result;
    if (!direct_tasks.empty()) {  // # 先提交回落的普通 RDMA 任务
        result = RdmaTransport::submitTransferTask(direct_tasks);
    }
    for (auto& item : pending) {  // # 再把 PXN 提交入队
        auto status = item.lane->enqueue(std::move(item.submission));
        if (result.ok() && !status.ok()) result = status;  // # 保留首个错误
    }
    if (!pending.empty()) pump_->wake();  // # 有新提交则唤醒 pump 线程推进
    return result;
}

bool PxnRdmaTransport::selectPxnLane(const TransferRequest& request,  // # 新增:为写请求挑选 PXN lane,不满足条件返回 false 回落
                                     std::string& session,
                                     pxn::SenderLane*& lane) {
    if (request.opcode != TransferRequest::WRITE || request.length == 0) {  // # 只接受非空 WRITE
        return false;
    }

    auto local = metadata_->getSegmentDescByID(LOCAL_SEGMENT_ID);  // # 取本地段描述
    const uint64_t source = reinterpret_cast<uint64_t>(request.source);
    bool cuda_source = false;
    for (const auto& buffer : local->buffers) {  // # 检查源地址是否落在 CUDA 缓冲区
        if (covers(buffer, source, request.length) &&
            buffer.name.rfind("cuda:", 0) == 0) {
            cuda_source = true;
            break;
        }
    }
    if (!cuda_source) return false;  // # 源不在 CUDA 显存则不走 PXN

    auto target = metadata_->getSegmentDescByID(request.target_id);  // # 取目标段描述
    if (target == nullptr || target->name.empty() ||
        target->name.size() > pxn::kMaxSessionLength) {  // # 目标没有合法会话名则回落
        return false;
    }

    int buffer_id = -1;
    int device_id = -1;
    if (RdmaTransport::selectDevice(target.get(), request.target_offset,  // # 解析目标 buffer/设备
                                    request.length, buffer_id,
                                    device_id) != 0) {
        return false;
    }
    pxn::RailResolver resolver(globalConfig().pxn_rail_map);  // # 用 rail map 规范化目标 rail
    const std::string target_rail =
        resolver.canonicalize(target->devices[device_id].name);
    if (target_rail == resources_->localRail()) return false;  // # 同机柜本地 rail 不需要 PXN 转发

    std::lock_guard<std::mutex> lock(peer_mutex_);  // # 保护 lane 缓存
    auto cached = lanes_by_rail_.find(target_rail);  // # 查已建过的 lane
    if (cached != lanes_by_rail_.end()) {
        lane = cached->second;
        session = target->name;
        return true;
    }
    std::vector<pxn::RegistryEntry> entries;
    if (!resources_->registry().discover(entries).ok()) return false;  // # 从注册表发现远端 peer
    for (const auto& entry : entries) {
        if (entry.rail != target_rail) continue;  // # 只匹配目标 rail
        pxn::PeerResources* peer = nullptr;
        if (!resources_->mapPeer(entry, peer).ok()) return false;  // # 映射 peer 资源
        if (!sender_pipeline_->addPeer(*peer, lane).ok()) return false;  // # 建立 lane 并登记 peer
        lanes_by_rail_[target_rail] = lane;  // # 缓存该 rail 的 lane
        session = target->name;
        return true;
    }
    return false;  // # 没有匹配的远端则回落普通 RDMA
}

}  // namespace mooncake

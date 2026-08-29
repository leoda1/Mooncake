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

#ifndef MOONCAKE_TRANSFER_ENGINE_PXN_RDMA_TRANSPORT_H_
#define MOONCAKE_TRANSFER_ENGINE_PXN_RDMA_TRANSPORT_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "transport/rdma_transport/rdma_transport.h"

namespace mooncake {
namespace pxn {
class LocalResources;
class PxnPump;
class RailResolver;
class RelayPipeline;
class SenderLane;
class SenderPipeline;
}  // namespace pxn

class PxnRdmaTransport : public RdmaTransport {
   public:
    PxnRdmaTransport();
    ~PxnRdmaTransport() override;

    PxnRdmaTransport(const PxnRdmaTransport&) = delete;
    PxnRdmaTransport& operator=(const PxnRdmaTransport&) = delete;

    int install(std::string& local_server_name,
                std::shared_ptr<TransferMetadata> metadata,
                std::shared_ptr<Topology> topology) override;
    Status submitTransferTask(
        const std::vector<TransferTask*>& task_list) override;

    bool pxnReady() const {
        return resources_ != nullptr && sender_pipeline_ != nullptr &&
               relay_pipeline_ != nullptr && pump_ != nullptr;
    }

   private:
    struct SelectionContext;
    struct RelayLane {
        pxn::SenderLane* lane = nullptr;
        uint32_t rail_index = 0;
    };

    bool selectPxnLane(const TransferRequest& request,
                       SelectionContext& context, std::string& session,
                       pxn::SenderLane*& lane, uint32_t& rail_index);

    std::unique_ptr<pxn::PxnPump> pump_;
    std::unique_ptr<pxn::RelayPipeline> relay_pipeline_;
    std::unique_ptr<pxn::SenderPipeline> sender_pipeline_;
    std::unique_ptr<pxn::LocalResources> resources_;
    std::unique_ptr<pxn::RailResolver> rail_resolver_;
    std::mutex peer_mutex_;
    std::unordered_map<std::string, RelayLane> lanes_by_rail_;

    struct PxnStats {
        std::atomic<uint64_t> not_write{0};  // opcode != WRITE or len 0
        std::atomic<uint64_t> not_cuda{0};   // source is host memory
        std::atomic<uint64_t> no_target{0};  // target segment unusable
        std::atomic<uint64_t> no_device{0};  // selectDevice failed
        std::atomic<uint64_t> same_rail{0};  // same rail -> native RDMA
        std::atomic<uint64_t> no_relay{0};   // cross rail but no relay
        std::atomic<uint64_t> pxn_used{0};   // took the PXN path
        std::atomic<uint64_t> pxn_bytes{0};
        std::atomic<uint64_t> direct_bytes{0};
        std::atomic<uint64_t> route_cache_hit{0};
        std::atomic<uint64_t> route_cache_miss{0};
        std::atomic<uint64_t> reported{0};
    };
    PxnStats pxn_stats_;
    void reportPxnStats();
};

}  // namespace mooncake

#endif  // MOONCAKE_TRANSFER_ENGINE_PXN_RDMA_TRANSPORT_H_

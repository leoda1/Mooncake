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

#include <memory>

#include "transport/rdma_transport/rdma_transport.h"

namespace mooncake {
namespace pxn {
class LocalResources;
}

class PxnRdmaTransport : public RdmaTransport {
   public:
    ~PxnRdmaTransport() override;

    int install(std::string& local_server_name,
                std::shared_ptr<TransferMetadata> metadata,
                std::shared_ptr<Topology> topology) override;

    bool pxnReady() const { return resources_ != nullptr; }

   private:
    std::unique_ptr<pxn::LocalResources> resources_;
};

}  // namespace mooncake

#endif  // MOONCAKE_TRANSFER_ENGINE_PXN_RDMA_TRANSPORT_H_

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

PxnRdmaTransport::~PxnRdmaTransport() {
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
    return result;
}

}  // namespace mooncake

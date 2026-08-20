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

#include "transport/pxn/pxn_staging.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <cstring>
#include <string>
#include <string_view>

#include "transport/rdma_transport/rdma_transport.h"

namespace mooncake {
namespace pxn {
namespace {

static_assert(sizeof(cudaIpcMemHandle_t) == kCudaIpcHandleSize);

Status cudaError(std::string_view operation, cudaError_t error) {
    return Status::Memory(std::string(operation) + ": " +
                          cudaGetErrorString(error));
}

class CudaDeviceScope {
   public:
    explicit CudaDeviceScope(int device_id) {
        auto error = cudaGetDevice(&previous_device_);
        if (error != cudaSuccess) {
            status_ = cudaError("cudaGetDevice failed", error);
            return;
        }
        if (previous_device_ == device_id) return;
        error = cudaSetDevice(device_id);
        if (error != cudaSuccess) {
            status_ = cudaError("cudaSetDevice failed", error);
            return;
        }
        restore_ = true;
    }

    ~CudaDeviceScope() {
        if (!restore_) return;
        auto error = cudaSetDevice(previous_device_);
        if (error != cudaSuccess) {
            LOG(ERROR) << "Failed to restore CUDA device: "
                       << cudaGetErrorString(error);
        }
    }

    const Status& status() const { return status_; }

   private:
    int previous_device_ = -1;
    bool restore_ = false;
    Status status_;
};

class CudaRdmaStagingBackend final : public StagingBackend {
   public:
    CudaRdmaStagingBackend(RdmaTransport& transport, int device_id)
        : transport_(transport),
          device_id_(device_id),
          location_("cuda:" + std::to_string(device_id)) {}

    Status allocateArena(size_t size, uintptr_t& address) override {
        CudaDeviceScope device(device_id_);
        if (!device.status().ok()) return device.status();
        void* pointer = nullptr;
        auto error = cudaMalloc(&pointer, size);
        if (error != cudaSuccess) return cudaError("cudaMalloc failed", error);
        address = reinterpret_cast<uintptr_t>(pointer);
        return Status::OK();
    }

    Status freeArena(uintptr_t address) override {
        CudaDeviceScope device(device_id_);
        if (!device.status().ok()) return device.status();
        auto error = cudaFree(reinterpret_cast<void*>(address));
        return error == cudaSuccess ? Status::OK()
                                    : cudaError("cudaFree failed", error);
    }

    Status registerArena(uintptr_t address, size_t size) override {
        CudaDeviceScope device(device_id_);
        if (!device.status().ok()) return device.status();
        const int result = transport_.registerLocalMemory(
            reinterpret_cast<void*>(address), size, location_,
            /*remote_accessible=*/false, /*update_metadata=*/false);
        return result == 0
                   ? Status::OK()
                   : Status::Context("PXN RDMA memory registration failed: " +
                                     std::to_string(result));
    }

    Status unregisterArena(uintptr_t address) override {
        CudaDeviceScope device(device_id_);
        if (!device.status().ok()) return device.status();
        const int result = transport_.unregisterLocalMemory(
            reinterpret_cast<void*>(address), /*update_metadata=*/false);
        return result == 0
                   ? Status::OK()
                   : Status::Context("PXN RDMA memory unregistration failed: " +
                                     std::to_string(result));
    }

    Status exportArena(uintptr_t address, CudaIpcHandle& handle) override {
        CudaDeviceScope device(device_id_);
        if (!device.status().ok()) return device.status();
        cudaIpcMemHandle_t ipc_handle;
        auto error =
            cudaIpcGetMemHandle(&ipc_handle, reinterpret_cast<void*>(address));
        if (error != cudaSuccess) {
            return cudaError("cudaIpcGetMemHandle failed", error);
        }
        std::memcpy(handle.bytes, &ipc_handle, sizeof(ipc_handle));
        return Status::OK();
    }

    Status importArena(const CudaIpcHandle& handle,
                       uintptr_t& address) override {
        CudaDeviceScope device(device_id_);
        if (!device.status().ok()) return device.status();
        cudaIpcMemHandle_t ipc_handle;
        std::memcpy(&ipc_handle, handle.bytes, sizeof(ipc_handle));
        void* pointer = nullptr;
        auto error = cudaIpcOpenMemHandle(&pointer, ipc_handle,
                                          cudaIpcMemLazyEnablePeerAccess);
        if (error != cudaSuccess) {
            return cudaError("cudaIpcOpenMemHandle failed", error);
        }
        address = reinterpret_cast<uintptr_t>(pointer);
        return Status::OK();
    }

    Status closeImportedArena(uintptr_t address) override {
        CudaDeviceScope device(device_id_);
        if (!device.status().ok()) return device.status();
        auto error = cudaIpcCloseMemHandle(reinterpret_cast<void*>(address));
        return error == cudaSuccess
                   ? Status::OK()
                   : cudaError("cudaIpcCloseMemHandle failed", error);
    }

   private:
    RdmaTransport& transport_;
    int device_id_;
    std::string location_;
};

}  // namespace

std::unique_ptr<StagingBackend> makeCudaRdmaStagingBackend(
    RdmaTransport& transport, int device_id) {
    return std::make_unique<CudaRdmaStagingBackend>(transport, device_id);
}

}  // namespace pxn
}  // namespace mooncake

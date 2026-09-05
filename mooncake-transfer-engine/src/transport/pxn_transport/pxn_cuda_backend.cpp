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

#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "transport/pxn_transport/pxn_nvtx.h"

#include "transport/rdma_transport/rdma_transport.h"

namespace mooncake {
namespace pxn {
namespace {

static_assert(sizeof(cudaIpcMemHandle_t) == kCudaIpcHandleSize);
#if CUDA_VERSION >= 12030
constexpr CUdevice_attribute kCanUseStreamMemOpsAttribute =
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS;
#elif CUDA_VERSION >= 11070
constexpr CUdevice_attribute kCanUseStreamMemOpsAttribute =
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1;
#else
constexpr CUdevice_attribute kCanUseStreamMemOpsAttribute =
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS;
#endif

thread_local int bound_cuda_device = -1;

Status cudaError(std::string_view operation, cudaError_t error) {
    return Status::Memory(std::string(operation) + ": " +
                          cudaGetErrorString(error));
}

Status driverError(std::string_view operation, CUresult error) {
    const char* description = nullptr;
    (void)cuGetErrorString(error, &description);
    return Status::Memory(
        std::string(operation) + ": " +
        (description == nullptr ? "unknown CUDA error" : description));
}

class CudaDeviceScope {
   public:
    explicit CudaDeviceScope(int device_id) {
        if (bound_cuda_device == device_id) return;
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

class CudaEventRing {
   public:
    explicit CudaEventRing(int device_id) : device_id_(device_id) {}

    ~CudaEventRing() {
        CudaDeviceScope device(device_id_);
        if (!device.status().ok()) return;
        for (auto event : events_) {
            if (event != nullptr) (void)cudaEventDestroy(event);
        }
    }

    Status initialize(size_t count) {
        CudaDeviceScope device(device_id_);
        if (!device.status().ok()) return device.status();
        events_.assign(count, nullptr);
        for (size_t index = 0; index < count; ++index) {
            auto error = cudaEventCreateWithFlags(&events_[index],
                                                  cudaEventDisableTiming);
            if (error != cudaSuccess) {
                return cudaError("cudaEventCreateWithFlags failed", error);
            }
        }
        return Status::OK();
    }

    cudaEvent_t next() {
        if (events_.empty()) return nullptr;
        cudaEvent_t event = events_[cursor_];
        cursor_ = (cursor_ + 1 == events_.size()) ? 0 : cursor_ + 1;
        return event;
    }

    size_t size() const { return events_.size(); }

   private:
    int device_id_;
    size_t cursor_ = 0;
    std::vector<cudaEvent_t> events_;
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

class CudaReadyFence final : public SenderReadyFence {
   public:
    explicit CudaReadyFence(cudaEvent_t event) : event_(event) {}

    cudaEvent_t event() const { return event_; }
    void reset(cudaEvent_t event) { event_ = event; }

   private:
    cudaEvent_t event_ = nullptr;
};

class CudaSenderLaneHandle final : public SenderLaneHandle {
   public:
    explicit CudaSenderLaneHandle(int device_id) : device_id_(device_id) {}

    ~CudaSenderLaneHandle() override {
        CudaDeviceScope device(device_id_);
        if (!device.status().ok()) return;
        if (stream_ != nullptr) {
            auto error = cudaStreamSynchronize(stream_);
            if (error != cudaSuccess) {
                LOG(ERROR) << "Failed to synchronize PXN sender stream: "
                           << cudaGetErrorString(error);
            }
        }
        if (publish_event_ != nullptr) {
            (void)cudaEventDestroy(publish_event_);
        }
        if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
        if (control_ != nullptr) (void)cudaHostUnregister(control_);
    }

    int device_id_;
    ControlBlock* control_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t publish_event_ = nullptr;
    CUdeviceptr device_doorbell_ = 0;
    bool stream_mem_ops_ = false;
};

class CudaSenderBackend final : public SenderBackend {
   public:
    explicit CudaSenderBackend(int device_id)
        : device_id_(device_id), event_ring_(device_id) {}

    Status bindThread() override {
        auto error = cudaSetDevice(device_id_);
        if (error != cudaSuccess)
            return cudaError("cudaSetDevice failed", error);
        bound_cuda_device = device_id_;
        return Status::OK();
    }

    Status createLane(const SenderLaneEndpoint& endpoint,
                      std::unique_ptr<SenderLaneHandle>& handle) override {
        CudaDeviceScope device(device_id_);
        if (!device.status().ok()) return device.status();

        auto candidate = std::make_unique<CudaSenderLaneHandle>(device_id_);
        auto error =
            cudaHostRegister(endpoint.control, sizeof(ControlBlock),
                             cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (error != cudaSuccess) {
            return cudaError("cudaHostRegister PXN control block failed",
                             error);
        }
        candidate->control_ = endpoint.control;

        void* device_control = nullptr;
        error = cudaHostGetDevicePointer(&device_control, endpoint.control, 0);
        if (error != cudaSuccess) {
            return cudaError("cudaHostGetDevicePointer failed", error);
        }
        const auto control_base = reinterpret_cast<uintptr_t>(endpoint.control);
        const auto doorbell = reinterpret_cast<uintptr_t>(
            &endpoint.control->lanes[endpoint.lane_index].header.doorbell);
        candidate->device_doorbell_ =
            reinterpret_cast<CUdeviceptr>(device_control) +
            static_cast<CUdeviceptr>(doorbell - control_base);

        error = cudaStreamCreateWithFlags(&candidate->stream_,
                                          cudaStreamNonBlocking);
        if (error != cudaSuccess) {
            return cudaError("cudaStreamCreateWithFlags failed", error);
        }

        if (event_ring_.size() < endpoint.geometry.slots_per_lane) {
            auto status = event_ring_.initialize(endpoint.geometry.slots_per_lane);
            if (!status.ok()) return status;
        }

        int stream_mem_ops = 0;
        CUdevice cuda_device;
        if (cuDeviceGet(&cuda_device, device_id_) == CUDA_SUCCESS &&
            cuDeviceGetAttribute(&stream_mem_ops, kCanUseStreamMemOpsAttribute,
                                 cuda_device) == CUDA_SUCCESS &&
            stream_mem_ops != 0) {
            candidate->stream_mem_ops_ = true;
        } else {
            error = cudaEventCreateWithFlags(&candidate->publish_event_,
                                             cudaEventDisableTiming);
            if (error != cudaSuccess) {
                return cudaError("cudaEventCreateWithFlags failed", error);
            }
        }
        LOG(INFO) << "PXN CUDA sender lane " << endpoint.lane_index
                  << " stream_mem_ops=" << candidate->stream_mem_ops_
                  << ", doorbell="
                  << (candidate->stream_mem_ops_ ? "cuStreamWriteValue64"
                                                 : "cudaEvent+CPU poll");

        handle = std::move(candidate);
        return Status::OK();
    }

    Status recordReady(std::unique_ptr<SenderReadyFence>& fence) override {
        CudaDeviceScope device(device_id_);
        if (!device.status().ok()) return device.status();
        auto* existing = static_cast<CudaReadyFence*>(fence.get());
        cudaEvent_t event = event_ring_.next();
        auto error = cudaEventRecord(event, cudaStreamPerThread);
        if (error != cudaSuccess) {
            error = cudaStreamSynchronize(cudaStreamPerThread);
            if (error != cudaSuccess) {
                return cudaError("cudaStreamSynchronize failed", error);
            }
            event = nullptr;  // already synchronized; no wait needed
        }
        if (existing != nullptr) {
            existing->reset(event);
        } else {
            fence = std::make_unique<CudaReadyFence>(event);
        }
        return Status::OK();
    }

    Status queryReady(const SenderReadyFence& fence, bool& ready) override {
        ready = false;
        const auto* cuda_fence = static_cast<const CudaReadyFence*>(&fence);
        if (cuda_fence == nullptr) {
            return Status::InvalidArgument("invalid PXN ready event");
        }
        CudaDeviceScope device(device_id_);
        if (!device.status().ok()) return device.status();
        if (cuda_fence->event() == nullptr) {
            ready = true;
            return Status::OK();
        }
        auto error = cudaEventQuery(cuda_fence->event());
        if (error == cudaErrorNotReady) return Status::OK();
        if (error != cudaSuccess) {
            return cudaError("cudaEventQuery ready event failed", error);
        }
        ready = true;
        return Status::OK();
    }

    Status publish(SenderLaneHandle& handle, const SenderReadyFence& fence,
                   const std::vector<SenderCopy>& copies, uint64_t sequence,
                   SenderPublishState& state) override {
        PXN_NVTX_SENDER("pxn::sender::publish");
        auto* lane = static_cast<CudaSenderLaneHandle*>(&handle);
        const auto* ready = static_cast<const CudaReadyFence*>(&fence);
        if (lane == nullptr || ready == nullptr || copies.empty()) {
            return Status::InvalidArgument("invalid PXN CUDA sender state");
        }
        warnOnForeignSourceDevice(copies.front().source);
        CudaDeviceScope device(device_id_);
        if (!device.status().ok()) return device.status();

        cudaError_t error = cudaSuccess;
        if (ready->event() != nullptr) {
            error = cudaStreamWaitEvent(lane->stream_, ready->event(), 0);
            if (error != cudaSuccess) {
                return cudaError("cudaStreamWaitEvent failed", error);
            }
        }

        auto& sources = scratch_sources_;
        auto& destinations = scratch_destinations_;
        auto& sizes = scratch_sizes_;
        sources.clear();
        destinations.clear();
        sizes.clear();
        sources.reserve(copies.size());
        destinations.reserve(copies.size());
        sizes.reserve(copies.size());
        for (const auto& copy : copies) {
            sources.push_back(reinterpret_cast<void*>(copy.source));
            destinations.push_back(reinterpret_cast<void*>(copy.destination));
            sizes.push_back(copy.length);
        }

#if CUDART_VERSION >= 12080
        cudaMemcpyAttributes attributes{};
        attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        size_t attribute_index = 0;
        auto& mutable_sizes = sizes;
#if CUDART_VERSION >= 13000
        error = cudaMemcpyBatchAsync(
            const_cast<const void**>(destinations.data()),
            const_cast<const void**>(sources.data()), mutable_sizes.data(),
            copies.size(), &attributes, &attribute_index, 1, lane->stream_);
#else
        size_t failed_index = copies.size();
        error = cudaMemcpyBatchAsync(destinations.data(), sources.data(),
                                     mutable_sizes.data(), copies.size(),
                                     &attributes, &attribute_index, 1,
                                     &failed_index, lane->stream_);
#endif
        if (error != cudaSuccess) {
            return cudaError("cudaMemcpyBatchAsync failed", error);
        }
#else
        for (size_t index = 0; index < copies.size(); ++index) {
            error =
                cudaMemcpyAsync(destinations[index], sources[index],
                                sizes[index], cudaMemcpyDefault, lane->stream_);
            if (error != cudaSuccess) {
                return cudaError("cudaMemcpyAsync failed", error);
            }
        }
#endif

        if (lane->stream_mem_ops_) {
            auto result =
                cuStreamWriteValue64(reinterpret_cast<CUstream>(lane->stream_),
                                     lane->device_doorbell_, sequence,
                                     CU_STREAM_WRITE_VALUE_DEFAULT);
            if (result != CUDA_SUCCESS) {
                return driverError("cuStreamWriteValue64 failed", result);
            }
            state = SenderPublishState::kPublished;
            return Status::OK();
        }

        error = cudaEventRecord(lane->publish_event_, lane->stream_);
        if (error != cudaSuccess) {
            return cudaError("cudaEventRecord publish event failed", error);
        }
        state = SenderPublishState::kPendingCpuDoorbell;
        return Status::OK();
    }

    Status pollCpuDoorbell(SenderLaneHandle& handle, bool& ready) override {
        ready = false;
        auto* lane = static_cast<CudaSenderLaneHandle*>(&handle);
        if (lane == nullptr || lane->publish_event_ == nullptr) {
            return Status::InvalidArgument("invalid PXN publish event");
        }
        CudaDeviceScope device(device_id_);
        if (!device.status().ok()) return device.status();
        auto error = cudaEventQuery(lane->publish_event_);
        if (error == cudaErrorNotReady) return Status::OK();
        if (error != cudaSuccess) {
            return cudaError("cudaEventQuery failed", error);
        }
        ready = true;
        return Status::OK();
    }

   private:
    void warnOnForeignSourceDevice(uintptr_t source) {
        if (source_device_checked_) return;
        source_device_checked_ = true;
        cudaPointerAttributes attributes{};
        auto error = cudaPointerGetAttributes(&attributes,
                                              reinterpret_cast<void*>(source));
        if (error != cudaSuccess) {
            // Not a CUDA pointer, or attributes unavailable; nothing to say.
            (void)cudaGetLastError();
            return;
        }
        if (attributes.type != cudaMemoryTypeDevice ||
            attributes.device == device_id_) {
            return;
        }
        LOG(WARNING) << "PXN staging copies are bouncing through host memory: "
                        "the sender is bound to cuda:"
                     << device_id_ << " but the source buffer lives on cuda:"
                     << attributes.device
                     << ". The PXN transport captures its device with "
                        "cudaGetDevice() at install time, so select the GPU "
                        "(cudaSetDevice) before creating the TransferEngine to "
                        "keep staging on NVLink.";
    }

    int device_id_;
    bool source_device_checked_ = false;
    CudaEventRing event_ring_;
    std::vector<void*> scratch_sources_;
    std::vector<void*> scratch_destinations_;
    std::vector<size_t> scratch_sizes_;
};

}  // namespace

std::unique_ptr<StagingBackend> makeCudaRdmaStagingBackend(
    RdmaTransport& transport, int device_id) {
    return std::make_unique<CudaRdmaStagingBackend>(transport, device_id);
}

std::unique_ptr<SenderBackend> makeCudaSenderBackend(int device_id) {
    return std::make_unique<CudaSenderBackend>(device_id);
}

}  // namespace pxn
}  // namespace mooncake

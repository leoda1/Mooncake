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

#ifndef MOONCAKE_TRANSFER_ENGINE_PXN_NVTX_H_
#define MOONCAKE_TRANSFER_ENGINE_PXN_NVTX_H_

// NVTX ranges for the two PXN roles a single engine plays at once: sender
// (NVLink copies into a peer's arena) and relay (RDMA on a peer's behalf).
// Both run on this process's pump threads, so one nsys capture shows whether
// they overlap or serialize against each other.
//
// Off unless built with -DPXN_ENABLE_NVTX=ON: the macros compile to nothing,
// so no cost is paid in a normal build.

#ifdef PXN_ENABLE_NVTX

#include <nvtx3/nvToolsExt.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstdint>

namespace mooncake {
namespace pxn {

// Distinct colors so the two roles are separable at a glance in the timeline.
inline constexpr uint32_t kNvtxSenderColor = 0xFF4FC3F7;  // light blue
inline constexpr uint32_t kNvtxRelayColor = 0xFFFFB74D;   // orange

class NvtxRange {
   public:
    NvtxRange(const char* name, uint32_t color) {
        nvtxEventAttributes_t attributes = {};
        attributes.version = NVTX_VERSION;
        attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attributes.colorType = NVTX_COLOR_ARGB;
        attributes.color = color;
        attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attributes.message.ascii = name;
        nvtxRangePushEx(&attributes);
    }

    ~NvtxRange() { nvtxRangePop(); }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace pxn
}  // namespace mooncake

#define PXN_NVTX_CONCAT_INNER(a, b) a##b
#define PXN_NVTX_CONCAT(a, b) PXN_NVTX_CONCAT_INNER(a, b)

#define PXN_NVTX_SENDER(name)                                     \
    ::mooncake::pxn::NvtxRange PXN_NVTX_CONCAT(pxn_nvtx_, __LINE__)( \
        name, ::mooncake::pxn::kNvtxSenderColor)

#define PXN_NVTX_RELAY(name)                                      \
    ::mooncake::pxn::NvtxRange PXN_NVTX_CONCAT(pxn_nvtx_, __LINE__)( \
        name, ::mooncake::pxn::kNvtxRelayColor)

// Instantaneous marker, for points with no duration (e.g. a doorbell write).
#define PXN_NVTX_MARK(name) nvtxMarkA(name)

// Names the calling thread in the timeline, so the sender and relay rows are
// told apart without matching thread ids.
#define PXN_NVTX_NAME_THREAD(name) \
    nvtxNameOsThreadA(static_cast<uint32_t>(::syscall(SYS_gettid)), name)

#else  // PXN_ENABLE_NVTX

#define PXN_NVTX_SENDER(name) \
    do {                      \
    } while (0)
#define PXN_NVTX_RELAY(name) \
    do {                     \
    } while (0)
#define PXN_NVTX_MARK(name) \
    do {                    \
    } while (0)
#define PXN_NVTX_NAME_THREAD(name) \
    do {                           \
    } while (0)

#endif  // PXN_ENABLE_NVTX

#endif  // MOONCAKE_TRANSFER_ENGINE_PXN_NVTX_H_

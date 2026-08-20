# Mooncake Transfer Engine — 已探索记忆

## 核心问题：vllm 调用了哪些 Transfer Engine 接口

vllm 通过 mooncake_connector.py 与 Transfer Engine 交互，调用了以下 **4 个核心接口**：

---

### 接口 1：`engine.initialize()`
- **Python 签名**：`initialize(local_hostname: str, metadata_server: str, protocol: str, device_name: str) -> int`
- **C++ 对应**：`TransferEnginePy::initialize()` → `TransferEngineImpl::init()`
- **调用位置**：`mooncake_connector.py` L411
- **参数（vllm 实际传值）**：
  - `local_hostname`：本机 IP，如 `"192.168.1.1:0"`
  - `metadata_server`：硬编码为 `"P2PHANDSHAKE"`（直连 P2P 模式，跳过 etcd）
  - `protocol`：`"rdma"` 或 `"tcp"`（env `VLLM_MOONCAKE_PROTOCOL`）
  - `device_name`：RDMA 网卡名或空串 `""`
- **功能**：初始化 TransferEngine，自动发现拓扑并安装 RDMA/TCP transport
- **P2PHANDSHAKE 模式逻辑**（见 `transfer_engine_impl.cpp` L111）：
  - 不连接 metadata 服务器，直接 P2P 握手
  - `findAvailableTcpPort()` 随机选 TCP 端口供 RPC 使用
  - `local_server_name_` 更新为 `"ip:随机端口"`
- **返回**：0 成功，非 0 失败（抛 RuntimeError）

---

### 接口 2：`engine.get_rpc_port()`
- **Python 签名**：`get_rpc_port() -> int`
- **C++ 对应**：`TransferEnginePy::getRpcPort()` → `TransferEngineImpl::getRpcPort()`
- **调用位置**：`mooncake_connector.py` L415
- **功能**：返回初始化后 engine 监听的 TCP RPC 端口
- **用途**：构造 remote session 标识符 `f"{hostname}:{rpc_port}"`
- **返回**：`metadata_->localRpcMeta().rpc_port`

---

### 接口 3：`engine.batch_register_memory()`
- **Python 签名**：`batch_register_memory(buffer_addresses: List[int], capacities: List[int]) -> int`
- **C++ 对应**：`TransferEnginePy::batchRegisterMemory()` → `TransferEngineImpl::registerLocalMemoryBatch()`
- **调用位置**：`mooncake_connector.py` L720，`register_kv_caches()` 中
- **参数**：
  - `buffer_addresses`：KV cache tensor 的 GPU 内存地址列表（`tensor.data_ptr()`）
  - `capacities`：每块内存的字节大小（`tensor.nbytes`）
- **功能**：向底层所有 transport（RDMA/TCP）批量注册内存，使其可被远端 RDMA 访问
- **调用时机**：模型初始化时一次性调用，不在请求路径上
- **返回**：0 成功，非 0 失败（抛 RuntimeError）

---

### 接口 4：`engine.batch_transfer_sync_write()`
- **Python 签名**：`batch_transfer_sync_write(target_hostname: str, buffers: List[int], peer_buffer_addresses: List[int], lengths: List[int]) -> int`
- **C++ 对应**：`TransferEnginePy::batchTransferSyncWrite()` → `batchTransferSync(..., WRITE)`
- **调用位置**：`mooncake_connector.py` L669，`_send_blocks()` 中
- **参数**：
  - `target_hostname`：`"remote_ip:remote_rpc_port"` 格式
  - `buffers`：本地源内存地址列表（Prefill GPU 端）
  - `peer_buffer_addresses`：远端目标内存地址列表（Decode GPU 端）
  - `lengths`：每对传输的字节数列表
- **功能**：批量同步 RDMA WRITE，阻塞直到全部完成。内部流程：
  1. `openSegment(target_hostname)` 建立/复用连接句柄
  2. `allocateBatchID()` 分配批次 ID
  3. 构造 `TransferRequest[]`（opcode=WRITE）
  4. `submitTransfer(batch_id, entries)` 提交
  5. 轮询 `getBatchTransferStatus(batch_id)` 直到 COMPLETED/FAILED
  6. 超时（30s + 数据量/1GiB 估算）或失败时重试（最多 numContexts()+1 次）
- **调用链**：`_mooncake_sender()` ZMQ 线程 → `_sender_worker()` 线程池 → `send_kv_to_decode()` → `_send_blocks()` → `batch_transfer_sync_write()`
- **返回**：0 成功，非 0 失败（抛 RuntimeError）

---

## vllm 未使用的接口（Transfer Engine 完整 API）

以下接口存在于 `TransferEnginePy` / Python 绑定中，但 vllm 当前未调用：

| 接口 | 说明 |
|------|------|
| `initializeExt()` | 带 metadata_type 参数的扩展初始化 |
| `allocateManagedBuffer()` / `freeManagedBuffer()` | engine 内部 buddy allocator 管理的缓冲区 |
| `transferSyncWrite()` / `transferSyncRead()` | 单次同步传输（非批量） |
| `transferSubmitWrite()` + `transferCheckStatus()` | 异步提交 + 轮询状态 |
| `batchTransferSyncRead()` | 批量同步 RDMA READ |
| `batchTransferAsyncWrite()` / `batchTransferAsyncRead()` | 批量异步提交，返回 batch_id |
| `getBatchTransferStatus()` | 查询批次完成状态 |
| `batchTransferOnCuda()` / `transferWriteOnCuda()` 等 | CUDA stream 感知的传输（需 USE_CUDA） |
| `getFirstBufferAddress()` | 按 segment 名获取第一个缓冲区地址 |
| `registerMemory()` / `unregisterMemory()` | 单块内存注册/注销 |
| `batchUnregisterMemory()` | 批量注销内存 |
| `getLocalTopology()` | 返回本地设备拓扑（仅诊断工具用） |
| `getNotifies()` | 拉取异步通知消息 |
| `writeBytesToBuffer()` / `readBytesFromBuffer()` | 内存读写工具函数 |
| `sendNotifyByID()` / `sendNotifyByName()` | 发送通知（C++ 层，Python 未暴露给 vllm） |

---

## 关键文件索引

| 文件 | 说明 |
|------|------|
| `mooncake-transfer-engine/src/transfer_engine_impl.cpp` | C++ TransferEngine 核心实现，含 init/P2PHANDSHAKE 逻辑 |
| `mooncake-transfer-engine/include/transfer_engine.h` | C++ 公开头文件 |
| `mooncake-integration/transfer_engine/transfer_engine_py.h` | Python wrapper 类声明（所有 Python 可调用方法） |
| `mooncake-integration/transfer_engine/transfer_engine_py.cpp` | Python wrapper 实现 |
| `mooncake-wheel/mooncake/mooncake_connector_v1.py` | Mooncake 官方版 connector（异步 sender 版） |
| `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py` | vllm 集成的 connector（线程池 sender 版） |


## mooncake 完成 RDMA 参数交换的过程
具体到代码：

1. TCP 连接建立发生在内核里的三次握手，但代码入口是客户端 `connect()` 和服务端 `listen()/accept()`。
服务端监听在 [transfer_metadata_plugin.cpp:639](/Users/joker/Desktop/project/baai/Mooncake/mooncake-transfer-engine/src/transfer_metadata_plugin.cpp#L639) 开始，真正 `listen()`/`accept()` 在 [transfer_metadata_plugin.cpp:706](/Users/joker/Desktop/project/baai/Mooncake/mooncake-transfer-engine/src/transfer_metadata_plugin.cpp#L706) 和 [transfer_metadata_plugin.cpp:717](/Users/joker/Desktop/project/baai/Mooncake/mooncake-transfer-engine/src/transfer_metadata_plugin.cpp#L717)。
客户端发起连接在 [transfer_metadata_plugin.cpp:885](/Users/joker/Desktop/project/baai/Mooncake/mooncake-transfer-engine/src/transfer_metadata_plugin.cpp#L885)，真正 `connect()` 在 [transfer_metadata_plugin.cpp:908](/Users/joker/Desktop/project/baai/Mooncake/mooncake-transfer-engine/src/transfer_metadata_plugin.cpp#L908)。
你不用在业务代码里看到 “SYN/SYN-ACK/ACK” 这三个包，内核在 `connect()` 成功返回之前已经做完了。

2. Mooncake RDMA 参数交换发生在 `RdmaEndPoint::setupConnectionsByActive()` 里组包并发送。
主动侧第一次发现 endpoint 还没连上时，会在 [worker_pool.cpp:244](/Users/joker/Desktop/project/baai/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp#L244) 调 `setupConnectionsByActive()`。
它在 [rdma_endpoint.cpp:134](/Users/joker/Desktop/project/baai/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L134) 构造 `HandShakeDesc`，内容主要是 `local_nic_path`、`peer_nic_path`、`qp_num`，然后在 [rdma_endpoint.cpp:146](/Users/joker/Desktop/project/baai/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L146) 调 `sendHandshake()`。
`sendHandshake()` 在 [transfer_metadata.cpp:836](/Users/joker/Desktop/project/baai/Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L836)，它把 `HandShakeDesc` 编成 JSON 后，通过刚才那条 TCP 连接发给对端。

3. 对端收到后，在握手 daemon 里解包，再回到 RDMA 代码。
接收入口在 [transfer_metadata.cpp:810](/Users/joker/Desktop/project/baai/Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L810)。
RDMA transport 的回调在 [rdma_transport.cpp:618](/Users/joker/Desktop/project/baai/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp#L618)，它会找到对应 endpoint，然后调用被动侧 [rdma_endpoint.cpp:178](/Users/joker/Desktop/project/baai/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L178) 的 `setupConnectionsByPassive()`，把本端 `qp_num` 等信息回填给主动侧。

4. 真正把 RDMA QP 连起来，是在 `doSetupConnection()`，不是在 TCP 里。
两边最后都会走到 [rdma_endpoint.cpp:330](/Users/joker/Desktop/project/baai/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L330)，然后在 [rdma_endpoint.cpp:360](/Users/joker/Desktop/project/baai/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L360) 到 [rdma_endpoint.cpp:446](/Users/joker/Desktop/project/baai/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L446) 里把 QP 从 `RESET -> INIT -> RTR -> RTS`。
这里真正用到的参数是 `peer_qp_num`、`peer_gid`、`peer_lid`。其中：
- `qp_num` 是通过 Mooncake 这次 TCP 握手交换的。
- `gid/lid` 不是 `HandShakeDesc` 里发过去的，而是从元数据 `segment_desc->devices` 查出来的，见 [rdma_endpoint.cpp:166](/Users/joker/Desktop/project/baai/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L166) 和 [rdma_endpoint.cpp:209](/Users/joker/Desktop/project/baai/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L209)。


## 问题 1：`TransferEngine()` 和 RPC

`self.engine = TransferEngine()` 实例化的是 `TransferEnginePy`（pybind11 包装类），内部持有一个 `TransferEngineImpl` 实例。

**RPC = Remote Procedure Call（远程过程调用）**。但这里所谓的"RPC 服务"**不是 gRPC 那种重量级服务**，而是一个极轻量的 **TCP 握手守护线程**（`SocketHandShakePlugin::startDaemon()`），它做的事情是：

- 在一个随机分配的 TCP 端口上 `listen()`
- 启动一个后台线程 `accept()` 连接
- 处理两类请求：
  1. **Handshake**：交换 RDMA QP 编号（`HandShakeDesc`，JSON 格式）
  2. **Metadata 同步**：交换内存注册信息（buffer 地址、rkey 等）
- 这个端口就是 `get_rpc_port()` 返回的端口

所以 `TransferEngineImpl` 的核心**不是** KV cache 传输本身的 RPC 服务，而是**连接建立和元数据交换的控制面**。数据面走的是 RDMA verbs。

---

## 问题 2：`initialize()` 的完整逻辑

你的理解**不完全正确**。`initialize()` 并**不注册 KV cache**，也**不立即把 MR/rkey 发给其他人**。完整流程是：

### `initialize(hostname, "P2PHANDSHAKE", "rdma", "")` 做了什么：

| 步骤 | 动作 | 说明 |
|------|------|------|
| 1 | `setFilesLimit()` | 提高进程 fd 上限 |
| 2 | 解析 hostname | 提取 host 和端口信息 |
| 3 | **分配随机 TCP 端口** | `findAvailableTcpPort()` → 这就是 `get_rpc_port()` 的端口 |
| 4 | 创建 `TransferMetadata` | P2PHANDSHAKE 模式：设 `p2p_handshake_mode_=true`，**不连接 etcd/Redis** |
| 5 | 创建 `MultiTransport` | 安装 RDMA transport（探测本机 IB 设备、创建 PD、CQ 等 RDMA 资源） |
| 6 | `addRpcMetaEntry()` | **启动 TCP 握手守护线程**（在分配的端口上 listen） |
| 7 | `installTransport("rdma")` | 初始化 RDMA context，创建 QP 池（但**不连接任何 peer**） |

### 关键：**连接是懒建立的**

- `initialize()` **不会** 建立任何 RDMA 连接
- `batch_register_memory()` 注册本地 MR（`ibv_reg_mr`），获得 lkey/rkey，**但不发给任何人**
- **首次 `batch_transfer_sync_write("host:port", ...)` 时**才通过 `openSegment()` → `sendHandshake()` 发起 TCP 握手，交换 QP 编号，建立 RDMA 连接
- 对端的 rkey/buffer 信息也是在首次传输时通过 TCP 拉取的

### 时序图：

```
节点 A (Prefiller)                          节点 B (Decoder)
─────────────────                          ─────────────────
initialize()                               initialize()
  ├─ 分配 TCP 端口 12345                     ├─ 分配 TCP 端口 23456
  ├─ 创建 RDMA context                      ├─ 创建 RDMA context  
  └─ 启动 TCP daemon (listen:12345)          └─ 启动 TCP daemon (listen:23456)

batch_register_memory(ptrs, sizes)          batch_register_memory(ptrs, sizes)
  └─ ibv_reg_mr() → 得到 lkey/rkey           └─ ibv_reg_mr() → 得到 lkey/rkey
     (仅本地，不发给任何人)                      (仅本地，不发给任何人)

batch_transfer_sync_write("B:23456", ...)
  ├─ openSegment("B:23456")                 
  │    └─ 首次：缓存 handle                  
  ├─ submitTransfer() → 触发 RDMA:           
  │    ├─ TCP 连接到 B:23456  ──────────────→ TCP daemon accept()
  │    ├─ 交换 QP 编号 (JSON)  ←────────────→ 交换 QP 编号
  │    ├─ 拉取 B 的 MR metadata ←───────────→ 返回 rkey/buffer 信息
  │    ├─ RESET→INIT→RTR→RTS (QP 状态机)     同步做 RESET→INIT→RTR→RTS
  │    └─ ibv_post_send(RDMA WRITE)  ──────→ 数据直达 GPU 内存
  └─ 轮询完成 → 返回 0
```

---
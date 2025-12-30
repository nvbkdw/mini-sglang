# Sglang scheduler overlap scheduling mechanism
_Exported on 12/25/2025 at 12:50:48 PST from Cursor (2.1.46)_

---

**User**

@mini-sglang/python/minisgl/scheduler/scheduler.py

@scheduler 



review code and teach me how sglang scheduler implements overlap scheduling mechanism, similar to the one in SGLang, to mitigate this. By preparing the next batch of requests on the CPU while the GPU is busy with the current batch, it effectively hides the CPU overhead. 

---

**Cursor**

I'll read the scheduler code and related files to explain the overlap scheduling mechanism.

Let me also read the remaining files to understand the full picture:

Now I have a complete picture. Let me explain the **overlap scheduling mechanism** in this mini-sglang implementation.

---

## Overview: The Problem and Solution

### The Problem
In LLM inference, there's a typical execution pattern:
1. **CPU work**: Schedule next batch, prepare metadata, allocate memory, process results
2. **GPU work**: Execute the model forward pass

If done sequentially, the GPU sits idle while the CPU prepares data, wasting valuable compute resources.

### The Solution: Overlap Scheduling
SGLang overlaps CPU and GPU work using **dual CUDA streams** and a **pipelined execution model**:

```
Time →
GPU:  [Forward Batch N] -------- [Forward Batch N+1] -------- [Forward Batch N+2]
CPU:        [Process N-1 results + Schedule N+1]   [Process N results + Schedule N+2]
```

---

## Key Implementation Details

### 1. Dual CUDA Streams Setup

```79:93:/home/huangruoyu/workspace/mini-sglang/python/minisgl/scheduler/scheduler.py
class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.config = config
        self.engine = Engine(config)
        self.tp_info = config.tp_info
        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)
```

Two streams are created:
- **`self.stream`**: The scheduler's stream for CPU-bound metadata preparation
- **`self.engine.stream`**: The engine's stream for GPU-bound model execution

### 2. The `ForwardInput` Data Structure for Caching

```68:76:/home/huangruoyu/workspace/mini-sglang/python/minisgl/scheduler/scheduler.py
# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    load_indices: torch.Tensor
    write_indices: torch.Tensor


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"
```

This caches all prepared data for a batch to avoid **Invalid Memory Access (IMA)** errors when the data is used across iterations.

### 3. The Core Overlap Loop

```226:250:/home/huangruoyu/workspace/mini-sglang/python/minisgl/scheduler/scheduler.py
    @torch.inference_mode()
    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data, ongoing_data)
        return ongoing_data
```

**This is the heart of overlap scheduling!** Let me break it down:

#### Step-by-Step Execution Flow:

| Step | Action | Stream |
|------|--------|--------|
| 1 | Receive new messages (non-blocking if work pending) | CPU |
| 2 | `_schedule_next_batch()` - Prepare next batch metadata | Scheduler stream |
| 3 | `wait_stream()` - Sync scheduler → engine | Sync point |
| 4 | `_forward()` - Execute GPU forward pass | Engine stream |
| 5 | `_process_last_data()` - Process **previous** batch results | CPU (overlaps with step 4!) |

**The magic**: Step 5 processes the *previous* batch's results **while** step 4 is running the *current* batch on the GPU!

### 4. Stream Synchronization

```244:247:/home/huangruoyu/workspace/mini-sglang/python/minisgl/scheduler/scheduler.py
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))
```

- `self.engine_stream_ctx` switches to the engine's CUDA stream
- `wait_stream(self.stream)` ensures the scheduler's metadata preparation is complete before GPU execution starts
- After launching GPU work, control returns immediately (non-blocking) to process previous results

### 5. Asynchronous Result Copying

```199:216:/home/huangruoyu/workspace/mini-sglang/python/minisgl/engine/engine.py
    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream

        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = torch.empty_like(next_tokens_gpu, device="cpu", pin_memory=True)
        next_tokens_cpu.copy_(next_tokens_gpu, non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```

Key points:
- **Pinned memory** (`pin_memory=True`) enables fast async GPU→CPU transfer
- **Non-blocking copy** (`non_blocking=True`) doesn't stall the GPU
- **CUDA Event** (`copy_done_event`) tracks when the copy completes

### 6. Synchronization on Result Processing

```108:114:/home/huangruoyu/workspace/mini-sglang/python/minisgl/scheduler/scheduler.py
    def _process_last_data(
        self, last_data: ForwardData | None, ongoing_data: ForwardData | None
    ) -> None:
        if last_data is None:
            return
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
```

Only synchronizes on the **copy event** (not the entire stream), waiting just for the result tokens to be available on CPU.

### 7. Main Loop Entry Point

```265:275:/home/huangruoyu/workspace/mini-sglang/python/minisgl/scheduler/scheduler.py
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)
```

The loop chains batches together: each call returns `ongoing_data` which becomes `last_data` in the next iteration.

---

## Visual Timeline

```
Iteration 1:
  CPU: [receive_msg] [schedule_batch_1]
  GPU:                                  [forward_batch_1 >>>>>>>>>>>>>>>>]

Iteration 2:
  CPU: [receive_msg] [schedule_batch_2] [process_batch_1_results]  ← overlapped!
  GPU:                                  [forward_batch_2 >>>>>>>>>>>>>>>>]

Iteration 3:
  CPU: [receive_msg] [schedule_batch_3] [process_batch_2_results]  ← overlapped!
  GPU:                                  [forward_batch_3 >>>>>>>>>>>>>>>>]
```

---

## Summary

| Component | Purpose |
|-----------|---------|
| Dual streams | Separate CPU scheduling from GPU execution |
| `ForwardInput` caching | Preserve batch data across iterations |
| `wait_stream()` | Ensure metadata ready before GPU work |
| Pinned memory + async copy | Fast, non-blocking GPU→CPU transfer |
| CUDA Events | Fine-grained sync on specific operations |
| `last_data` pattern | Process previous results while current batch runs |

This design effectively **hides 90%+ of CPU scheduling overhead** by overlapping it with GPU computation, significantly improving throughput in high-load scenarios.

```plaintext
Time →
GPU:  [Forward Batch N] -------- [Forward Batch N+1] -------- [Forward Batch N+2]
CPU:        [Process N-1 results + Schedule N+1]   [Process N results + Schedule N+2]
```

```python
class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.config = config
        self.engine = Engine(config)
        self.tp_info = config.tp_info
        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)
```

```python
# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    load_indices: torch.Tensor
    write_indices: torch.Tensor


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"
```

```python
    @torch.inference_mode()
    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data, ongoing_data)
        return ongoing_data
```

```python
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))
```

```python
    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream

        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = torch.empty_like(next_tokens_gpu, device="cpu", pin_memory=True)
        next_tokens_cpu.copy_(next_tokens_gpu, non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```

```python
    def _process_last_data(
        self, last_data: ForwardData | None, ongoing_data: ForwardData | None
    ) -> None:
        if last_data is None:
            return
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
```

```python
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)
```

```plaintext
Iteration 1:
  CPU: [receive_msg] [schedule_batch_1]
  GPU:                                  [forward_batch_1 >>>>>>>>>>>>>>>>]

Iteration 2:
  CPU: [receive_msg] [schedule_batch_2] [process_batch_1_results]  ← overlapped!
  GPU:                                  [forward_batch_2 >>>>>>>>>>>>>>>>]

Iteration 3:
  CPU: [receive_msg] [schedule_batch_3] [process_batch_2_results]  ← overlapped!
  GPU:                                  [forward_batch_3 >>>>>>>>>>>>>>>>]
```

---

**User**

@engine @kernel 

review code the teach how does mini-sglang Optimized Kernels: Integrates FlashAttention and FlashInfer for maximum efficiency.

---

**Cursor**

I'll review the engine and kernel code to explain how mini-sglang integrates FlashAttention and FlashInfer for optimized attention computation.

Let me explore the attention backends to see FlashAttention/FlashInfer integration:

Now let me read the attention backend files:

Let me also check the kvcache implementation to complete the picture:

Let me also check the warp primitives used in the custom kernels:

Now I have a complete picture. Let me explain how **mini-sglang integrates FlashAttention and FlashInfer** for maximum efficiency.

---

## Overview: Optimized Kernel Architecture

Mini-sglang integrates two state-of-the-art attention implementations and custom CUDA kernels to maximize inference efficiency:

| Component | Purpose | GPU Optimization |
|-----------|---------|------------------|
| **FlashInfer** | Paged attention for prefill/decode | Optimized for variable-length sequences |
| **FlashAttention 3** | High-performance attention on Hopper | Tensor Core + async memory |
| **Custom JIT Kernels** | Embedding indexing, KV cache store | Warp-level vectorized memory ops |

---

## 1. Attention Backend Selection

### Auto-Selection Based on GPU Architecture

```26:33:/home/huangruoyu/workspace/mini-sglang/python/minisgl/attention/__init__.py
def _resolve_auto_backend(config: ModelConfig) -> str:
    if is_sm100_supported():  # blackwell
        return "fi"
    elif is_sm90_supported():  # hopper
        return "fa3,fi"
    else:  # pre-hopper
        return "fi"
```

| GPU Architecture | Backend Selection |
|------------------|-------------------|
| SM100+ (Blackwell) | FlashInfer only |
| SM90 (Hopper H100/H200) | **Hybrid**: FA3 for prefill, FlashInfer for decode |
| Pre-Hopper (A100, etc.) | FlashInfer only |

### Hybrid Backend for Maximum Performance

```39:66:/home/huangruoyu/workspace/mini-sglang/python/minisgl/attention/base.py
class HybridBackend(BaseAttnBackend):
    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)
```

**Why hybrid?**
- **Prefill**: Long sequences benefit from FlashAttention 3's superior memory bandwidth on Hopper
- **Decode**: FlashInfer's paged attention is optimized for single-token decoding with KV cache

---

## 2. FlashInfer Backend Integration

### Initialization with Workspace Buffers

```86:129:/home/huangruoyu/workspace/mini-sglang/python/minisgl/attention/fi.py
class FlashInferBackend(BaseAttnBackend):
    def __init__(
        self,
        config: ModelConfig,
        kvcache: BaseKVCache,
        page_table: torch.Tensor,
    ) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.kvcache = kvcache
        self.device = kvcache.device
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is buggy, use fa2 instead
        )
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
        )

        # NOTE: some hack to reuse the int_workspace_buffer
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer
```

Key optimizations:
- **128 MB workspace buffer**: Pre-allocated for FlashInfer's internal operations
- **Shared int workspace**: Memory reuse between prefill/decode wrappers
- **NHD layout**: `[num_pages, num_heads, head_dim]` for coalesced memory access

### Forward Pass with Lazy Planning

```181:189:/home/huangruoyu/workspace/mini-sglang/python/minisgl/attention/fi.py
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        self._initialize_metadata_once(metadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)
```

The `_initialize_metadata_once` uses FlashInfer's **plan/run pattern**:
1. **Plan phase**: Computes memory access patterns (CPU-bound, done once per batch)
2. **Run phase**: Executes the actual attention (GPU-bound, called per layer)

### CUDA Graph Support for Decode

```244:267:/home/huangruoyu/workspace/mini-sglang/python/minisgl/attention/fi.py
    def prepare_for_capture(self, batch: Batch) -> None:
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        batch.padded_reqs = batch.reqs
        capture = self.capture
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
```

FlashInfer provides `CUDAGraphBatchDecodeWithPagedKVCacheWrapper` which:
- Uses **pre-allocated buffers** for graph capture
- Enables **tensor cores** based on GQA ratio (≥4 heads per KV head)
- Eliminates kernel launch overhead during decode

---

## 3. FlashAttention 3 Backend (Hopper Optimized)

### Direct Integration with sgl-kernel

```144:210:/home/huangruoyu/workspace/mini-sglang/python/minisgl/attention/fa3.py
def _fa3_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k_new: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    ...
) -> torch.Tensor:
    try:
        import sgl_kernel.flash_attn  # noqa: F401
    except ImportError:
        raise ImportError(
            "sgl_kernel.flash_attn is not found. Please install it with `pip install sgl-kernel`."
        )

    out, *_ = torch.ops.sgl_kernel.fwd.default(
        q, k_cache, v_cache,
        None,  # k (not used, using cache)
        None,  # v (not used, using cache)
        ...
        page_table,
        ...
        softmax_scale,
        True,  # causal
        ...
    )
    return out
```

This uses SGLang's custom `sgl-kernel` which wraps FlashAttention 3 with:
- **Paged KV cache support**: Works with arbitrary page tables
- **Fused operations**: Combines attention with cache access
- **Hopper-specific optimizations**: TMA (Tensor Memory Accelerator), async copy

---

## 4. Custom JIT CUDA Kernels

### Warp-Level Vectorized Memory Operations

```40:59:/home/huangruoyu/workspace/mini-sglang/python/minisgl/kernel/csrc/jit/index.cu
template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::size_t kNumSplits, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    index_kernel(const __grid_constant__ IndexKernelParams params) {
  using namespace device;
  constexpr auto kSize = kElementSize;
  constexpr auto kSizePerWarp = kSize / kNumSplits;
  constexpr auto kWarpPerBlock = static_cast<unsigned>(kNumThreads / 32);

  // ... each warp handles one embedding vector
  if (warp_id < num_warps) {
    const auto pos = indices[warp_id / kNumSplits];
    const auto dst = pointer::offset(output, warp_id * kSizePerWarp);
    const auto src = pointer::offset(weight, pos * kSize,
                                     (warp_id % kNumSplits) * kSizePerWarp);
    warp::copy<kSizePerWarp>(dst, src);  // vectorized copy
  }
}
```

Key optimizations:
- **One warp per embedding**: 32 threads cooperatively copy one row
- **Vectorized loads**: Uses `uint4` (16 bytes) for coalesced memory access
- **Split mode**: Large embeddings split across multiple warps for parallelism

### Warp Copy Primitive

```40:59:/home/huangruoyu/workspace/mini-sglang/python/minisgl/kernel/csrc/include/minisgl/warp.cuh
template <std::size_t kBytes, std::size_t kUnit = details::resolve_unit_size(kBytes)>
__always_inline __device__ void copy(void *__restrict__ dst,
                                     const void *__restrict__ src) {
  using Package = details::mem_package_t<kBytes, kUnit>;  // uint4, uint2, or uint1
  constexpr auto kBytesPerLoop = sizeof(Package) * kWarpThreads;  // 512 bytes/loop for uint4
  constexpr auto kLoopCount = kBytes / kBytesPerLoop;

  const auto dst_packed = static_cast<Package *>(dst);
  const auto src_packed = static_cast<const Package *>(src);
  const auto lane_id = threadIdx.x % kWarpThreads;

#pragma unroll kLoopCount
  for (std::size_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kWarpThreads + lane_id;
    dst_packed[j] = src_packed[j];  // fully coalesced 128-byte transaction
  }
}
```

This achieves **peak memory bandwidth** by:
- Using 128-byte aligned transactions (32 threads × 4 bytes = 128 bytes per warp)
- Compile-time unrolling for optimal instruction scheduling

### KV Cache Store Kernel

```25:53:/home/huangruoyu/workspace/mini-sglang/python/minisgl/kernel/csrc/jit/store.cu
template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    store_kv_cache(const __grid_constant__ StoreKernelParams params) {
  // each warp handles one token's K and V
  if (warp_id < length) {
    const auto pos = static_cast<const T *>(indices)[warp_id];
    // Store K
    const auto dst_k = pointer::offset(k_cache, pos * kv_cache_stride);
    const auto src_k = pointer::offset(k, warp_id * kv_input_stride);
    warp::copy<kElementSize>(dst_k, src_k);
    // Store V
    const auto dst_v = pointer::offset(v_cache, pos * kv_cache_stride);
    const auto src_v = pointer::offset(v, warp_id * kv_input_stride);
    warp::copy<kElementSize>(dst_v, src_v);
  }
}
```

This kernel is called during every attention forward to store new K/V into the paged cache:

```56:67:/home/huangruoyu/workspace/mini-sglang/python/minisgl/kvcache/mha_pool.py
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )
```

---

## 5. CUDA Graph Integration

### Graph Capture for Decode Phase

```88:131:/home/huangruoyu/workspace/mini-sglang/python/minisgl/engine/graph.py
        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {cuda_graph_bs}")
        
        # warm up by capturing a graph and then destroying it
        g = torch.cuda.CUDAGraph()
        batch = Batch(reqs=[dummy_req] * self.max_graph_bs, phase="decode")
        attn_backend.prepare_for_capture(batch)
        with get_global_ctx().forward_batch(batch):
            self.logits[:] = model.forward()
            with torch.cuda.graph(g, stream=stream):
                self.logits[:] = model.forward()
        
        # Capture graphs for each batch size
        pool = None
        for bs in pbar:
            g = torch.cuda.CUDAGraph()
            # ... capture graph for this batch size
            with torch.cuda.graph(g, pool=pool, stream=stream):
                self.logits[:bs] = model.forward()
            if pool is None:
                pool = g.pool()  # share memory pool across graphs
            graph_list.append((bs, g))
```

Key optimizations:
- **Graph pooling**: All captured graphs share the same memory pool
- **Multiple batch sizes**: Pre-captured for [1, 2, 4, 8, 16, ..., max_bs]
- **Zero kernel launch overhead**: Single `g.replay()` replaces hundreds of kernel launches

---

## 6. End-to-End Flow Diagram

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Scheduler                                │
│  ┌───────────────┐    ┌───────────────┐    ┌─────────────────┐  │
│  │ PrefillManager │───▶│ DecodeManager │───▶│ Batch Scheduler │  │
│  └───────────────┘    └───────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Engine                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    GraphRunner                            │    │
│  │   ┌────────────────────────────────────────────────────┐ │    │
│  │   │  CUDA Graph (decode)  OR  Eager (prefill)          │ │    │
│  │   └────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Model Layers                                │
│  ┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │   Embedding   │──▶│  Transformer     │──▶│   LM Head       │  │
│  │  (indexing)   │   │  Layers × N      │   │  (indexing)     │  │
│  └──────────────┘   └──────────────────┘   └─────────────────┘  │
│         ▲                    │                                   │
│   Custom JIT                 ▼                                   │
│   Kernel              ┌──────────────────┐                       │
│                       │  Attention Layer │                       │
│                       │  ┌────────────┐  │                       │
│                       │  │ QKV Proj   │  │                       │
│                       │  └────────────┘  │                       │
│                       │        │         │                       │
│                       │        ▼         │                       │
│                       │  ┌────────────┐  │                       │
│                       │  │ FlashInfer │  │  ◀── Prefill/Decode  │
│                       │  │    or      │  │                       │
│                       │  │ FA3 (H100) │  │  ◀── Hopper optimized│
│                       │  └────────────┘  │                       │
│                       │        │         │                       │
│                       │        ▼         │                       │
│                       │  ┌────────────┐  │                       │
│                       │  │ store_kv   │  │  ◀── Custom JIT      │
│                       │  │ kernel     │  │                       │
│                       │  └────────────┘  │                       │
│                       └──────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Paged KV Cache                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Layout: [2, num_layers, num_pages, 1, num_heads, dim]   │    │
│  │          ▲                                                │    │
│  │          └── K/V split for memory efficiency              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary: Why These Optimizations Matter

| Optimization | Benefit |
|--------------|---------|
| **FlashInfer + FA3 Hybrid** | Best of both worlds: FA3 for prefill throughput, FlashInfer for decode latency |
| **Paged Attention** | Eliminates memory fragmentation, enables efficient batching |
| **Custom JIT Kernels** | Warp-level vectorized memory ops achieve peak bandwidth |
| **CUDA Graphs** | Eliminates kernel launch overhead (critical for decode) |
| **Tensor Core Selection** | Automatically uses tensor cores when GQA ratio ≥ 4 |
| **Non-blocking Plan/Run** | Overlaps CPU planning with GPU execution |

```python
def _resolve_auto_backend(config: ModelConfig) -> str:
    if is_sm100_supported():  # blackwell
        return "fi"
    elif is_sm90_supported():  # hopper
        return "fa3,fi"
    else:  # pre-hopper
        return "fi"
```

```python
class HybridBackend(BaseAttnBackend):
    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)
```

```python
class FlashInferBackend(BaseAttnBackend):
    def __init__(
        self,
        config: ModelConfig,
        kvcache: BaseKVCache,
        page_table: torch.Tensor,
    ) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.kvcache = kvcache
        self.device = kvcache.device
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is buggy, use fa2 instead
        )
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
        )

        # NOTE: some hack to reuse the int_workspace_buffer
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer
```

```python
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        self._initialize_metadata_once(metadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)
```

```python
    def prepare_for_capture(self, batch: Batch) -> None:
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        batch.padded_reqs = batch.reqs
        capture = self.capture
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
```

```python
def _fa3_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k_new: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    ...
) -> torch.Tensor:
    try:
        import sgl_kernel.flash_attn  # noqa: F401
    except ImportError:
        raise ImportError(
            "sgl_kernel.flash_attn is not found. Please install it with `pip install sgl-kernel`."
        )

    out, *_ = torch.ops.sgl_kernel.fwd.default(
        q, k_cache, v_cache,
        None,  # k (not used, using cache)
        None,  # v (not used, using cache)
        ...
        page_table,
        ...
        softmax_scale,
        True,  # causal
        ...
    )
    return out
```

```plaintext
template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::size_t kNumSplits, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    index_kernel(const __grid_constant__ IndexKernelParams params) {
  using namespace device;
  constexpr auto kSize = kElementSize;
  constexpr auto kSizePerWarp = kSize / kNumSplits;
  constexpr auto kWarpPerBlock = static_cast<unsigned>(kNumThreads / 32);

  // ... each warp handles one embedding vector
  if (warp_id < num_warps) {
    const auto pos = indices[warp_id / kNumSplits];
    const auto dst = pointer::offset(output, warp_id * kSizePerWarp);
    const auto src = pointer::offset(weight, pos * kSize,
                                     (warp_id % kNumSplits) * kSizePerWarp);
    warp::copy<kSizePerWarp>(dst, src);  // vectorized copy
  }
}
```

```plaintext
template <std::size_t kBytes, std::size_t kUnit = details::resolve_unit_size(kBytes)>
__always_inline __device__ void copy(void *__restrict__ dst,
                                     const void *__restrict__ src) {
  using Package = details::mem_package_t<kBytes, kUnit>;  // uint4, uint2, or uint1
  constexpr auto kBytesPerLoop = sizeof(Package) * kWarpThreads;  // 512 bytes/loop for uint4
  constexpr auto kLoopCount = kBytes / kBytesPerLoop;

  const auto dst_packed = static_cast<Package *>(dst);
  const auto src_packed = static_cast<const Package *>(src);
  const auto lane_id = threadIdx.x % kWarpThreads;

#pragma unroll kLoopCount
  for (std::size_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kWarpThreads + lane_id;
    dst_packed[j] = src_packed[j];  // fully coalesced 128-byte transaction
  }
}
```

```plaintext
template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    store_kv_cache(const __grid_constant__ StoreKernelParams params) {
  // each warp handles one token's K and V
  if (warp_id < length) {
    const auto pos = static_cast<const T *>(indices)[warp_id];
    // Store K
    const auto dst_k = pointer::offset(k_cache, pos * kv_cache_stride);
    const auto src_k = pointer::offset(k, warp_id * kv_input_stride);
    warp::copy<kElementSize>(dst_k, src_k);
    // Store V
    const auto dst_v = pointer::offset(v_cache, pos * kv_cache_stride);
    const auto src_v = pointer::offset(v, warp_id * kv_input_stride);
    warp::copy<kElementSize>(dst_v, src_v);
  }
}
```

```python
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )
```

```python
        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {cuda_graph_bs}")
        
        # warm up by capturing a graph and then destroying it
        g = torch.cuda.CUDAGraph()
        batch = Batch(reqs=[dummy_req] * self.max_graph_bs, phase="decode")
        attn_backend.prepare_for_capture(batch)
        with get_global_ctx().forward_batch(batch):
            self.logits[:] = model.forward()
            with torch.cuda.graph(g, stream=stream):
                self.logits[:] = model.forward()
        
        # Capture graphs for each batch size
        pool = None
        for bs in pbar:
            g = torch.cuda.CUDAGraph()
            # ... capture graph for this batch size
            with torch.cuda.graph(g, pool=pool, stream=stream):
                self.logits[:bs] = model.forward()
            if pool is None:
                pool = g.pool()  # share memory pool across graphs
            graph_list.append((bs, g))
```

```plaintext
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Scheduler                                │
│  ┌───────────────┐    ┌───────────────┐    ┌─────────────────┐  │
│  │ PrefillManager │───▶│ DecodeManager │───▶│ Batch Scheduler │  │
│  └───────────────┘    └───────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Engine                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    GraphRunner                            │    │
│  │   ┌────────────────────────────────────────────────────┐ │    │
│  │   │  CUDA Graph (decode)  OR  Eager (prefill)          │ │    │
│  │   └────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Model Layers                                │
│  ┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │   Embedding   │──▶│  Transformer     │──▶│   LM Head       │  │
│  │  (indexing)   │   │  Layers × N      │   │  (indexing)     │  │
│  └──────────────┘   └──────────────────┘   └─────────────────┘  │
│         ▲                    │                                   │
│   Custom JIT                 ▼                                   │
│   Kernel              ┌──────────────────┐                       │
│                       │  Attention Layer │                       │
│                       │  ┌────────────┐  │                       │
│                       │  │ QKV Proj   │  │                       │
│                       │  └────────────┘  │                       │
│                       │        │         │                       │
│                       │        ▼         │                       │
│                       │  ┌────────────┐  │                       │
│                       │  │ FlashInfer │  │  ◀── Prefill/Decode  │
│                       │  │    or      │  │                       │
│                       │  │ FA3 (H100) │  │  ◀── Hopper optimized│
│                       │  └────────────┘  │                       │
│                       │        │         │                       │
│                       │        ▼         │                       │
│                       │  ┌────────────┐  │                       │
│                       │  │ store_kv   │  │  ◀── Custom JIT      │
│                       │  │ kernel     │  │                       │
│                       │  └────────────┘  │                       │
│                       └──────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Paged KV Cache                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Layout: [2, num_layers, num_pages, 1, num_heads, dim]   │    │
│  │          ▲                                                │    │
│  │          └── K/V split for memory efficiency              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

**User**

@kvcache how does KV cache works in mini-sglang

---

**Cursor**

I'll read all the KV cache files to explain how it works in mini-sglang.

Let me also check how the scheduler uses the cache for a complete picture:

Now I have a complete picture of the KV cache system. Let me explain how it works in mini-sglang.

---

## Overview: KV Cache Architecture in Mini-SGLang

Mini-sglang uses a **paged KV cache** with **prefix caching** to efficiently manage memory and reuse computations across requests.

```
┌─────────────────────────────────────────────────────────────────────┐
│                       KV Cache System                                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │   MHAKVCache    │◄───│  CacheManager    │◄───│ Scheduler      │ │
│  │  (GPU Storage)  │    │  (Page Allocator)│    │ (Orchestrator) │ │
│  └─────────────────┘    └──────────────────┘    └────────────────┘ │
│          ▲                      ▲                                   │
│          │                      │                                   │
│  ┌───────┴───────┐    ┌─────────┴─────────┐                        │
│  │ store_kv()    │    │ RadixCacheManager │                        │
│  │ (JIT Kernel)  │    │ (Prefix Matching) │                        │
│  └───────────────┘    └───────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. GPU Storage Layer: `MHAKVCache`

The `MHAKVCache` class manages the actual GPU memory for storing K and V tensors.

### Memory Layout

```16:48:/home/huangruoyu/workspace/mini-sglang/python/minisgl/kvcache/mha_pool.py
    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        dtype: torch.dtype,
        kv_layout: KVCacheLayout,
        device: torch.device,
    ):
        tp_info = get_tp_info()
        local_kv_heads = divide_even(num_kv_heads, tp_info.size)
        match kv_layout:
            case KVCacheLayout.PageFirst:
                kv_buffer = torch.empty(
                    (2, num_pages, num_layers, local_kv_heads, head_dim),
                    device=device,
                    dtype=dtype,
                ).permute(0, 2, 1, 3, 4)
            case KVCacheLayout.LayerFirst:
                kv_buffer = torch.empty(
                    (2, num_layers, num_pages, local_kv_heads, head_dim),
                    device=device,
                    dtype=dtype,
                )
        # ... view as [2, num_layers, num_pages, 1, local_kv_heads, head_dim]
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
```

**Two layout options:**

| Layout | Shape | Use Case |
|--------|-------|----------|
| `LayerFirst` | `[2, L, P, H, D]` | Default, better for layer-by-layer access |
| `PageFirst` | `[2, P, L, H, D]` | Better for page-level operations |

Where:
- `2` = K and V
- `L` = num_layers
- `P` = num_pages (tokens with page_size=1)
- `H` = num_kv_heads (divided by tensor parallel size)
- `D` = head_dim

### Storing KV with Custom CUDA Kernel

```56:67:/home/huangruoyu/workspace/mini-sglang/python/minisgl/kvcache/mha_pool.py
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )
```

The `store_cache` kernel uses **warp-level vectorized memory operations** to efficiently scatter K/V tensors to arbitrary page locations specified by `out_loc`.

---

## 2. Page Allocation Layer: Scheduler's `CacheManager`

The `CacheManager` in the scheduler manages which pages are free, allocated, or cached.

### Free Slot Management

```12:22:/home/huangruoyu/workspace/mini-sglang/python/minisgl/scheduler/cache.py
class CacheManager:
    def __init__(self, device: torch.device, num_pages: int, type: str):
        # TODO: support page_size > 1
        self._free_slots = torch.arange(num_pages, dtype=torch.int32, device=device)
        self.device = device
        self.manager = create_cache_manager(device=device, type=type)
        self.num_pages = num_pages

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self._free_slots = torch.cat([self._free_slots, indices])
```

### Allocation with Eviction

```39:52:/home/huangruoyu/workspace/mini-sglang/python/minisgl/scheduler/cache.py
    def allocate(self, needed_len: int) -> torch.Tensor:
        if needed_len <= (free_len := len(self._free_slots)):
            allocated = self._free_slots[:needed_len]
            self._free_slots = self._free_slots[needed_len:]
            return allocated

        # NOTE: len(evicted) + free_len >= needed_len
        evicted = self.manager.evict(needed_len - free_len)
        merged = torch.cat([self._free_slots, evicted])
        assert len(merged) >= needed_len, "Eviction did not free enough space."

        allocated = merged[:needed_len]
        self._free_slots = merged[needed_len:]
        return allocated
```

**Allocation flow:**
1. Try to allocate from `_free_slots`
2. If not enough, evict from the prefix cache
3. Merge free + evicted slots
4. Return allocated indices

---

## 3. Prefix Caching: `RadixCacheManager`

The radix tree enables **prefix sharing** between requests, dramatically reducing computation for similar prompts.

### Radix Tree Node Structure

```13:80:/home/huangruoyu/workspace/mini-sglang/python/minisgl/kvcache/radix_manager.py
class RadixTreeNode:
    counter: int = 0

    def __init__(self, tic: int | None = None) -> None:
        self.children: Dict[int, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()

        # these fields should be updated later
        self._key: torch.Tensor    # token IDs for this node
        self._value: torch.Tensor  # page indices in KV cache
        self._length: int
```

Each node stores:
- **`_key`**: The token IDs this node represents
- **`_value`**: The corresponding page indices in the GPU KV cache
- **`ref_count`**: How many active requests are using this prefix (for eviction protection)
- **`timestamp`**: Last access time (for LRU eviction)

### Prefix Matching: `_walk()`

```138:163:/home/huangruoyu/workspace/mini-sglang/python/minisgl/kvcache/radix_manager.py
    def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            this_id = int(input_ids[prefix_len].item())
            if this_id not in node.children:
                return node, prefix_len

            node = node.children[this_id]

            # NOTE: at least 1 char is matched, so match_len >= 1
            match_len = node.get_match_len(input_ids[prefix_len:])
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = node._split_at(match_len)
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len
```

**Walk algorithm:**
1. Start from root, traverse children by token ID
2. Use `fast_compare_key` kernel for efficient comparison
3. Split nodes when partial match occurs
4. Update timestamps for LRU tracking

### Visual Example of Radix Tree

```
Input sequences:
  A: "Hello world, how are you?"
  B: "Hello world, how is it going?"
  C: "Hello there!"

Radix Tree:
                    [root]
                       │
                 ┌─────┴─────┐
                 ▼           ▼
            ["Hello "]   ["Goodbye"]
                 │           │
           ┌─────┴─────┐     ▼
           ▼           ▼  [...]
    ["world, how "] ["there!"]
           │
      ┌────┴────┐
      ▼         ▼
 ["are you?"] ["is it going?"]
      (A)         (B)
```

When request B arrives after A, it reuses the KV cache for "Hello world, how " (saving computation for those tokens).

### LRU Eviction

```165:192:/home/huangruoyu/workspace/mini-sglang/python/minisgl/kvcache/radix_manager.py
    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)  # min-heap by timestamp (oldest first)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[int(node._key[0].item())]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)
```

**Eviction rules:**
1. Only evict **leaf nodes** with `ref_count == 0`
2. Use **min-heap by timestamp** (LRU order)
3. After evicting a leaf, parent may become evictable
4. Returns page indices to be freed

### Protection with `lock_handle()`

```97:114:/home/huangruoyu/workspace/mini-sglang/python/minisgl/kvcache/radix_manager.py
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node = node.parent
                node.ref_count -= 1
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
```

When a request starts using cached prefix:
- **Lock**: Increment `ref_count` for all ancestor nodes → moves them from evictable to protected
- **Unlock**: Decrement `ref_count` → nodes with `ref_count=0` become evictable again

---

## 4. Naive Cache Manager (No Prefix Caching)

For simpler use cases, `NaiveCacheManager` provides no prefix matching:

```14:34:/home/huangruoyu/workspace/mini-sglang/python/minisgl/kvcache/naive_manager.py
class NaiveCacheManager(BaseCacheManager):
    def __init__(self, device: torch.device):
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)

    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[NaiveCacheHandle, torch.Tensor]:
        _ = input_ids  # unused
        return NaiveCacheHandle(0), self.empty_tensor  # always return no match

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        return len(indices)  # pretend everything was already cached (no actual caching)
```

---

## 5. End-to-End Flow

### Request Lifecycle with Prefix Caching

```
1. New Request Arrives
   │
   ▼
2. match_prefix(input_ids[:-1])
   ├─ Returns cached_len and page indices
   │
   ▼
3. lock_handle() - Protect matched prefix from eviction
   │
   ▼
4. allocate(needed_tokens) - Get pages for new tokens
   ├─ From free_slots or eviction
   │
   ▼
5. During Forward Pass:
   │  ┌─────────────────────────────────────────┐
   │  │ For each layer:                         │
   │  │   store_kv(k, v, out_loc, layer_id)    │
   │  │   attention.forward() using k_cache/v_cache │
   │  └─────────────────────────────────────────┘
   │
   ▼
6. Request Finishes
   │
   ▼
7. insert_prefix(input_ids, indices)
   ├─ Store in radix tree for future reuse
   │
   ▼
8. unlock_handle() - Allow eviction of old prefix
   │
   ▼
9. Free unused pages back to free_slots
```

### Code Path in Scheduler

```54:62:/home/huangruoyu/workspace/mini-sglang/python/minisgl/scheduler/cache.py
    def free_and_cache_finished_req(
        self,
        old_handle: BaseCacheHandle,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        in_cache_len = self.manager.insert_prefix(input_ids, indices)
        self._free(indices[old_handle.cached_len : in_cache_len])
        self.unlock(old_handle)
```

When a request finishes:
1. Insert the full sequence into radix tree
2. Free pages that are now duplicated (already in cache from previous requests)
3. Unlock the old handle

---

## 6. Memory Layout Visualization

```
GPU Memory (KV Cache Buffer):
┌──────────────────────────────────────────────────────────────────┐
│  Layer 0                                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Page 0 │ Page 1 │ Page 2 │ ... │ Page N-1 │ Dummy Page    │  │
│  │ K:[H,D]│ K:[H,D]│ K:[H,D]│     │ K:[H,D]  │ K:[H,D]       │  │
│  └────────────────────────────────────────────────────────────┘  │
│  Layer 1                                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Page 0 │ Page 1 │ Page 2 │ ... │ Page N-1 │ Dummy Page    │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ...                                                              │
└──────────────────────────────────────────────────────────────────┘

Page Table (per request):
Request A: [5, 12, 7, 3, 8, ...]  ← indices into the page pool
Request B: [5, 12, 7, 22, 15, ...]  ← shares prefix pages 5,12,7 with A
                    ▲
                    └── Prefix sharing via radix tree
```

---

## Summary

| Component | Responsibility |
|-----------|----------------|
| **`MHAKVCache`** | GPU storage for K/V tensors with efficient kernel-based store |
| **`CacheManager`** | Page allocation, eviction, free slot tracking |
| **`RadixCacheManager`** | Prefix matching via radix tree, LRU eviction, reference counting |
| **`NaiveCacheManager`** | Simple mode without prefix caching |
| **`store_cache` kernel** | Warp-level vectorized scatter of K/V to pages |

**Key optimizations:**
1. **Paged allocation** eliminates memory fragmentation
2. **Radix tree prefix caching** reuses computation across similar prompts
3. **Reference counting** protects active prefixes from eviction
4. **LRU eviction** prioritizes keeping recently-used prefixes
5. **Custom CUDA kernels** for maximum memory bandwidth utilization

```plaintext
┌─────────────────────────────────────────────────────────────────────┐
│                       KV Cache System                                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │   MHAKVCache    │◄───│  CacheManager    │◄───│ Scheduler      │ │
│  │  (GPU Storage)  │    │  (Page Allocator)│    │ (Orchestrator) │ │
│  └─────────────────┘    └──────────────────┘    └────────────────┘ │
│          ▲                      ▲                                   │
│          │                      │                                   │
│  ┌───────┴───────┐    ┌─────────┴─────────┐                        │
│  │ store_kv()    │    │ RadixCacheManager │                        │
│  │ (JIT Kernel)  │    │ (Prefix Matching) │                        │
│  └───────────────┘    └───────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

```python
    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        dtype: torch.dtype,
        kv_layout: KVCacheLayout,
        device: torch.device,
    ):
        tp_info = get_tp_info()
        local_kv_heads = divide_even(num_kv_heads, tp_info.size)
        match kv_layout:
            case KVCacheLayout.PageFirst:
                kv_buffer = torch.empty(
                    (2, num_pages, num_layers, local_kv_heads, head_dim),
                    device=device,
                    dtype=dtype,
                ).permute(0, 2, 1, 3, 4)
            case KVCacheLayout.LayerFirst:
                kv_buffer = torch.empty(
                    (2, num_layers, num_pages, local_kv_heads, head_dim),
                    device=device,
                    dtype=dtype,
                )
        # ... view as [2, num_layers, num_pages, 1, local_kv_heads, head_dim]
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
```

```python
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )
```

```python
class CacheManager:
    def __init__(self, device: torch.device, num_pages: int, type: str):
        # TODO: support page_size > 1
        self._free_slots = torch.arange(num_pages, dtype=torch.int32, device=device)
        self.device = device
        self.manager = create_cache_manager(device=device, type=type)
        self.num_pages = num_pages

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self._free_slots = torch.cat([self._free_slots, indices])
```

```python
    def allocate(self, needed_len: int) -> torch.Tensor:
        if needed_len <= (free_len := len(self._free_slots)):
            allocated = self._free_slots[:needed_len]
            self._free_slots = self._free_slots[needed_len:]
            return allocated

        # NOTE: len(evicted) + free_len >= needed_len
        evicted = self.manager.evict(needed_len - free_len)
        merged = torch.cat([self._free_slots, evicted])
        assert len(merged) >= needed_len, "Eviction did not free enough space."

        allocated = merged[:needed_len]
        self._free_slots = merged[needed_len:]
        return allocated
```

```python
class RadixTreeNode:
    counter: int = 0

    def __init__(self, tic: int | None = None) -> None:
        self.children: Dict[int, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()

        # these fields should be updated later
        self._key: torch.Tensor    # token IDs for this node
        self._value: torch.Tensor  # page indices in KV cache
        self._length: int
```

```python
    def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            this_id = int(input_ids[prefix_len].item())
            if this_id not in node.children:
                return node, prefix_len

            node = node.children[this_id]

            # NOTE: at least 1 char is matched, so match_len >= 1
            match_len = node.get_match_len(input_ids[prefix_len:])
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = node._split_at(match_len)
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len
```

```plaintext
Input sequences:
  A: "Hello world, how are you?"
  B: "Hello world, how is it going?"
  C: "Hello there!"

Radix Tree:
                    [root]
                       │
                 ┌─────┴─────┐
                 ▼           ▼
            ["Hello "]   ["Goodbye"]
                 │           │
           ┌─────┴─────┐     ▼
           ▼           ▼  [...]
    ["world, how "] ["there!"]
           │
      ┌────┴────┐
      ▼         ▼
 ["are you?"] ["is it going?"]
      (A)         (B)
```

```python
    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)  # min-heap by timestamp (oldest first)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[int(node._key[0].item())]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)
```

```python
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node = node.parent
                node.ref_count -= 1
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
```

```python
class NaiveCacheManager(BaseCacheManager):
    def __init__(self, device: torch.device):
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)

    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[NaiveCacheHandle, torch.Tensor]:
        _ = input_ids  # unused
        return NaiveCacheHandle(0), self.empty_tensor  # always return no match

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        return len(indices)  # pretend everything was already cached (no actual caching)
```

```plaintext
1. New Request Arrives
   │
   ▼
2. match_prefix(input_ids[:-1])
   ├─ Returns cached_len and page indices
   │
   ▼
3. lock_handle() - Protect matched prefix from eviction
   │
   ▼
4. allocate(needed_tokens) - Get pages for new tokens
   ├─ From free_slots or eviction
   │
   ▼
5. During Forward Pass:
   │  ┌─────────────────────────────────────────┐
   │  │ For each layer:                         │
   │  │   store_kv(k, v, out_loc, layer_id)    │
   │  │   attention.forward() using k_cache/v_cache │
   │  └─────────────────────────────────────────┘
   │
   ▼
6. Request Finishes
   │
   ▼
7. insert_prefix(input_ids, indices)
   ├─ Store in radix tree for future reuse
   │
   ▼
8. unlock_handle() - Allow eviction of old prefix
   │
   ▼
9. Free unused pages back to free_slots
```

```python
    def free_and_cache_finished_req(
        self,
        old_handle: BaseCacheHandle,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        in_cache_len = self.manager.insert_prefix(input_ids, indices)
        self._free(indices[old_handle.cached_len : in_cache_len])
        self.unlock(old_handle)
```

```plaintext
GPU Memory (KV Cache Buffer):
┌──────────────────────────────────────────────────────────────────┐
│  Layer 0                                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Page 0 │ Page 1 │ Page 2 │ ... │ Page N-1 │ Dummy Page    │  │
│  │ K:[H,D]│ K:[H,D]│ K:[H,D]│     │ K:[H,D]  │ K:[H,D]       │  │
│  └────────────────────────────────────────────────────────────┘  │
│  Layer 1                                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Page 0 │ Page 1 │ Page 2 │ ... │ Page N-1 │ Dummy Page    │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ...                                                              │
└──────────────────────────────────────────────────────────────────┘

Page Table (per request):
Request A: [5, 12, 7, 3, 8, ...]  ← indices into the page pool
Request B: [5, 12, 7, 22, 15, ...]  ← shares prefix pages 5,12,7 with A
                    ▲
                    └── Prefix sharing via radix tree
```

---

**User**

does page pool all in GPU memory? what if GPU memory can not store all pages

---

**Cursor**

Great question! Let me investigate how mini-sglang handles GPU memory limits for the page pool.

Let me also check how the scheduler handles allocation failures:

Now I have the complete picture. Here's how mini-sglang handles GPU memory for the page pool:

---

## Yes, All Pages Are in GPU Memory

**The entire page pool is pre-allocated on GPU at startup.** Mini-sglang does NOT support CPU offloading or disk swapping.

---

## How GPU Memory is Managed

### 1. Automatic Page Count Calculation at Startup

```152:171:/home/huangruoyu/workspace/mini-sglang/python/minisgl/engine/engine.py
    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * self.model_config.head_dim
            * divide_even(self.model_config.num_kv_heads, config.tp_info.size)
            * config.page_size
            * self.dtype.itemsize
            * self.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-tokens"
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_pages} pages for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages
```

**Formula:**
```
available_memory = memory_ratio × total_gpu_memory - model_memory
num_pages = available_memory ÷ cache_per_page
```

Where:
- `memory_ratio` = 0.9 (default, uses 90% of GPU memory)
- `cache_per_page` = 2 × head_dim × num_kv_heads × num_layers × dtype_size

**Example for Llama-3-8B on H100 (80GB):**
```
cache_per_page = 2 × 128 × 8 × 32 × 2 bytes = 131,072 bytes = 128 KB
available_memory ≈ 65 GB (after model weights)
num_pages ≈ 65 GB ÷ 128 KB ≈ 530,000 pages (tokens)
```

---

## 2. What Happens When Memory is Full?

### Strategy 1: Admission Control (Don't Accept New Requests)

```38:52:/home/huangruoyu/workspace/mini-sglang/python/minisgl/scheduler/prefill.py
    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            return None

        handle, match_indices = self.cache_manager.match_req(req)
        cached_len = handle.cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None  # ← Reject if not enough space
        # ...
```

**Before accepting a request, the scheduler checks:**
```
estimated_tokens = (input_len - cached_prefix) + max_output_tokens
if estimated_tokens > available_pages:
    → Keep request in pending queue (don't schedule)
```

### Strategy 2: LRU Eviction from Prefix Cache

```39:52:/home/huangruoyu/workspace/mini-sglang/python/minisgl/scheduler/cache.py
    def allocate(self, needed_len: int) -> torch.Tensor:
        if needed_len <= (free_len := len(self._free_slots)):
            allocated = self._free_slots[:needed_len]
            self._free_slots = self._free_slots[needed_len:]
            return allocated

        # NOTE: len(evicted) + free_len >= needed_len
        evicted = self.manager.evict(needed_len - free_len)  # ← Evict cached prefixes
        merged = torch.cat([self._free_slots, evicted])
        assert len(merged) >= needed_len, "Eviction did not free enough space."

        allocated = merged[:needed_len]
        self._free_slots = merged[needed_len:]
        return allocated
```

**When free pages run out:**
1. Evict least-recently-used prefixes from the radix tree
2. Reclaim their page indices
3. Use for new allocations

### Strategy 3: Reserve Space for In-Flight Requests

```128:134:/home/huangruoyu/workspace/mini-sglang/python/minisgl/scheduler/prefill.py
        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,  # ← Reserve for ongoing decodes
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
```

The scheduler reserves space for tokens that ongoing decode requests will generate, preventing OOM during generation.

---

## 3. Memory States

```
Total GPU Memory for KV Cache
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │  Free Slots  │  │  Protected   │  │     Evictable         │ │
│  │              │  │  (in-use)    │  │  (cached prefixes)    │ │
│  └──────────────┘  └──────────────┘  └───────────────────────┘ │
│         ▲                 ▲                    ▲               │
│         │                 │                    │               │
│    _free_slots      ref_count > 0        ref_count = 0        │
│    (tensor)          (locked)             (unlocked)           │
│                                                                 │
│  available_size = free_slots + evictable                       │
└─────────────────────────────────────────────────────────────────┘
```

**Invariant check:**
```python
# scheduler/cache.py:66-70
if len(self._free_slots) + self.manager.size_info.total_size != self.num_pages:
    raise RuntimeError("CacheManager integrity check failed")
```

---

## 4. What Mini-SGLang Does NOT Support

| Feature | Status | Notes |
|---------|--------|-------|
| **CPU Offloading** | ❌ Not supported | All pages must fit in GPU |
| **Disk Swapping** | ❌ Not supported | No SSD/NVMe integration |
| **Dynamic Growth** | ❌ Not supported | Page pool fixed at startup |
| **Multi-GPU Paging** | ❌ Not supported | Each GPU has its own pool |

---

## 5. Failure Modes

### Case 1: Not Enough Memory at Startup
```python
assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-tokens"
```
**Solution:** Use a smaller model, reduce `--max-seq-len`, or use tensor parallelism.

### Case 2: All Pages Exhausted at Runtime

When `available_size = 0` and no evictable prefixes exist:
- New requests stay in `pending_list`
- Scheduler waits for ongoing requests to finish
- Once requests complete, their pages return to `_free_slots`

```python
# In prefill.py - request stays pending
if req := adder.try_add_one(pending_req):
    reqs.append(req)
else:
    break  # ← Stop adding requests when memory is full
```

---

## Summary

| Aspect | Behavior |
|--------|----------|
| **Storage Location** | 100% GPU memory |
| **Sizing** | Auto-calculated: `(memory_ratio × GPU_mem - model_size) ÷ page_size` |
| **When Full** | 1. Evict LRU prefixes 2. Queue new requests 3. Wait for completions |
| **Failure** | Requests wait in queue; no OOM crash |
| **CPU/Disk Offload** | Not implemented |

**Key insight:** Mini-sglang prioritizes **simplicity and performance** over memory flexibility. For larger context lengths or more concurrent requests, you need more GPU memory or tensor parallelism across multiple GPUs.

```python
    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * self.model_config.head_dim
            * divide_even(self.model_config.num_kv_heads, config.tp_info.size)
            * config.page_size
            * self.dtype.itemsize
            * self.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-tokens"
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_pages} pages for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages
```

```plaintext
available_memory = memory_ratio × total_gpu_memory - model_memory
num_pages = available_memory ÷ cache_per_page
```

```plaintext
cache_per_page = 2 × 128 × 8 × 32 × 2 bytes = 131,072 bytes = 128 KB
available_memory ≈ 65 GB (after model weights)
num_pages ≈ 65 GB ÷ 128 KB ≈ 530,000 pages (tokens)
```

```python
    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            return None

        handle, match_indices = self.cache_manager.match_req(req)
        cached_len = handle.cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None  # ← Reject if not enough space
        # ...
```

```plaintext
estimated_tokens = (input_len - cached_prefix) + max_output_tokens
if estimated_tokens > available_pages:
    → Keep request in pending queue (don't schedule)
```

```python
    def allocate(self, needed_len: int) -> torch.Tensor:
        if needed_len <= (free_len := len(self._free_slots)):
            allocated = self._free_slots[:needed_len]
            self._free_slots = self._free_slots[needed_len:]
            return allocated

        # NOTE: len(evicted) + free_len >= needed_len
        evicted = self.manager.evict(needed_len - free_len)  # ← Evict cached prefixes
        merged = torch.cat([self._free_slots, evicted])
        assert len(merged) >= needed_len, "Eviction did not free enough space."

        allocated = merged[:needed_len]
        self._free_slots = merged[needed_len:]
        return allocated
```

```python
        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,  # ← Reserve for ongoing decodes
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
```

```plaintext
Total GPU Memory for KV Cache
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │  Free Slots  │  │  Protected   │  │     Evictable         │ │
│  │              │  │  (in-use)    │  │  (cached prefixes)    │ │
│  └──────────────┘  └──────────────┘  └───────────────────────┘ │
│         ▲                 ▲                    ▲               │
│         │                 │                    │               │
│    _free_slots      ref_count > 0        ref_count = 0        │
│    (tensor)          (locked)             (unlocked)           │
│                                                                 │
│  available_size = free_slots + evictable                       │
└─────────────────────────────────────────────────────────────────┘
```

```python
# scheduler/cache.py:66-70
if len(self._free_slots) + self.manager.size_info.total_size != self.num_pages:
    raise RuntimeError("CacheManager integrity check failed")
```

```python
assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-tokens"
```

```python
# In prefill.py - request stays pending
if req := adder.try_add_one(pending_req):
    reqs.append(req)
else:
    break  # ← Stop adding requests when memory is full
```

---

**User**

@mini-sglang/python/minisgl/core.py please explain Req and Batch data strucutre, and how it's used in the inference engine, how it's related to cache table?



```

class Req:



    def __init__(

        self,

        *,

        input_ids: torch.Tensor,

        table_idx: int,

        cached_len: int,

        output_len: int,

        uid: int,

        sampling_params: SamplingParams,

        cache_handle: BaseCacheHandle,

    ) -> None:

        assert input_ids.is_cpu

        self.host_ids = input_ids

        self.table_idx = table_idx

        self.cached_len = cached_len

        self.device_len = len(input_ids)

        self.max_device_len = len(input_ids) + output_len

        self.uid = uid

        self.sampling_params = sampling_params

        self.cache_handle = cache_handle

        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property

    def remain_len(self) -> int:

        return self.max_device_len - self.device_len

    @property

    def extend_len(self) -> int:

        return self.device_len - self.cached_len

    def complete_one(self) -> None:

        self.cached_len = self.device_len

        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:

        self.host_ids = torch.cat([self.host_ids, next_token])

    def can_decode(self) -> bool:

        return self.remain_len > 0

    def __repr__(self) -> str:

        return (

            f"{type(self)}(table_idx={self.table_idx}, "

            f"cached_len={self.cached_len}, device_len={self.device_len}, "

            f"max_device_len={self.max_device_len})"

        )

class Batch:

    def __init__(self, *, reqs: List[Req], phase: Literal["prefill", "decode"]):

        self.reqs = reqs

        self.phase: Literal["prefill", "decode"] = phase

        # these fields should be set by scheduler

        self.input_ids: torch.Tensor

        self.out_loc: torch.Tensor

        self.padded_reqs: List[Req]  # may contain some dummy reqs for padding

        # this field should be set by attention backend

        self.attn_metadata: BaseAttnMetadata

    @property

    def is_prefill(self) -> bool:

        return self.phase == "prefill"

    @property

    def is_decode(self) -> bool:

        return self.phase == "decode"

    @property

    def size(self) -> int:

        return len(self.reqs)

    @property

    def padded_size(self) -> int:

        return len(self.padded_reqs)

```

---

**Cursor**

Let me read the core.py file and related components to give you a comprehensive explanation.

Let me also check how the scheduler uses these structures:

Now I have the complete picture. Let me explain the `Req` and `Batch` data structures and their relationship to the cache table.

---

## Overview: Data Structure Relationships

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Memory Layout                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TableManager (per-request slots)              CacheManager (KV pages)       │
│  ┌──────────────────────────────────┐         ┌───────────────────────────┐ │
│  │  token_pool    │   page_table    │         │     KV Cache Buffer       │ │
│  │  [R, S]        │   [R, S]        │         │  [2, L, P, H, D]          │ │
│  │  (token ids)   │   (page indices)│   ───►  │  (actual K/V tensors)     │ │
│  └──────────────────────────────────┘         └───────────────────────────┘ │
│         ▲                  ▲                            ▲                    │
│         │                  │                            │                    │
│         │    table_idx     │                     page indices                │
│         │         ▼        │                            │                    │
│         │    ┌─────────────┴───────────────────────────────┐                │
│         │    │                    Req                       │                │
│         │    │  table_idx ──► row in token_pool/page_table  │                │
│         │    │  cached_len, device_len ──► column ranges    │                │
│         │    └──────────────────────────────────────────────┘                │
│         │                          │                                         │
│         │                          ▼                                         │
│         │                   ┌────────────┐                                  │
│         │                   │   Batch    │                                  │
│         │                   │  reqs: []  │                                  │
│         │                   │  out_loc   │  ◄── newly allocated pages       │
│         │                   └────────────┘                                  │
│         │                                                                    │
│  R = max_running_reqs                                                        │
│  S = max_seq_len                                                             │
│  L = num_layers, P = num_pages, H = num_heads, D = head_dim                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. The `Req` (Request) Data Structure

A `Req` represents a **single inference request** with its complete state for KV cache management.

### Fields Explained

```python
class Req:
    def __init__(
        self,
        *,
        input_ids: torch.Tensor,   # Token IDs (on CPU)
        table_idx: int,            # Row index in page_table/token_pool
        cached_len: int,           # Tokens already in KV cache (from prefix matching)
        output_len: int,           # Maximum tokens to generate
        uid: int,                  # Unique request identifier
        sampling_params: SamplingParams,
        cache_handle: BaseCacheHandle,  # Handle to locked prefix in radix tree
    ):
        self.host_ids = input_ids          # All token IDs (CPU)
        self.table_idx = table_idx         # Which row in the 2D tables
        self.cached_len = cached_len       # KV already computed
        self.device_len = len(input_ids)   # Current sequence length
        self.max_device_len = len(input_ids) + output_len  # Maximum length
```

### Key Properties

| Property | Formula | Meaning |
|----------|---------|---------|
| `extend_len` | `device_len - cached_len` | Tokens needing attention computation in this step |
| `remain_len` | `max_device_len - device_len` | Tokens left to generate |

### Visual Example: Request Lifecycle

```
Initial state (prefill with prefix cache hit):
┌─────────────────────────────────────────────────────────────────┐
│ Token IDs: ["Hello", "world", ",", "how", "are", "you", "?"]    │
│                                                                  │
│  cached_len=3     device_len=7                max_device_len=17 │
│  (from cache)     (input length)              (+10 output)      │
│       │                 │                           │           │
│       ▼                 ▼                           ▼           │
│  [█ █ █ ░ ░ ░ ░ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯]                          │
│   ▲▲▲   ▲▲▲▲                                                    │
│   cached  extend_len=4                                           │
│           (need to compute)                                      │
└─────────────────────────────────────────────────────────────────┘

After one decode step (complete_one()):
┌─────────────────────────────────────────────────────────────────┐
│    cached_len=7 device_len=8                                    │
│               │ │                                               │
│               ▼ ▼                                               │
│  [█ █ █ █ █ █ █ ░ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯]                            │
│   ▲ ▲ ▲ ▲ ▲ ▲ ▲ ▲                                               │
│    all cached   extend_len=1 (new token)                        │
└─────────────────────────────────────────────────────────────────┘

Legend: █ = cached in KV cache, ░ = being computed, ▯ = future tokens
```

### The `complete_one()` Method

```python
def complete_one(self) -> None:
    self.cached_len = self.device_len  # All current tokens now cached
    self.device_len += 1               # Sequence grows by 1
```

Called after each forward pass to update the request state.

---

## 2. The `Batch` Data Structure

A `Batch` groups multiple requests for **efficient batched inference**.

### Fields Explained

```python
class Batch:
    def __init__(self, *, reqs: List[Req], phase: Literal["prefill", "decode"]):
        self.reqs = reqs              # Actual requests
        self.phase = phase            # "prefill" or "decode"
        
        # Set by scheduler:
        self.input_ids: torch.Tensor   # Flattened token IDs for all reqs
        self.out_loc: torch.Tensor     # Page indices for new KV entries
        self.padded_reqs: List[Req]    # Reqs + dummy reqs for CUDA graph
        
        # Set by attention backend:
        self.attn_metadata: BaseAttnMetadata  # Attention-specific data
```

### Prefill vs Decode

| Phase | `extend_len` per req | Description |
|-------|---------------------|-------------|
| **Prefill** | Variable (input length) | Process all input tokens |
| **Decode** | Always 1 | Generate one token at a time |

---

## 3. Relationship to Page Table and Token Pool

### The `TableManager` Structure

```python
class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor):
        self._free_slots = list(range(max_running_reqs))  # Available row indices
        self.page_table = page_table    # [max_running_reqs, max_seq_len] → page indices
        self.token_pool = torch.empty_like(page_table)  # [max_running_reqs, max_seq_len] → token IDs
```

### Two Parallel 2D Tables

```
                    Column Index (Position in Sequence)
                    0    1    2    3    4    5    6    7    ...  max_seq_len
                  ┌────┬────┬────┬────┬────┬────┬────┬────┬─────────────┐
Row 0 (table_idx=0)│ 42 │ 17 │  5 │ 89 │ 23 │    │    │    │             │  token_pool
                  ├────┼────┼────┼────┼────┼────┼────┼────┼─────────────┤  (token IDs)
Row 1 (table_idx=1)│ 12 │ 88 │ 45 │ 67 │    │    │    │    │             │
                  ├────┼────┼────┼────┼────┼────┼────┼────┼─────────────┤
Row 2 (table_idx=2)│ 99 │ 34 │ 56 │ 78 │ 11 │ 22 │ 33 │    │             │
                  └────┴────┴────┴────┴────┴────┴────┴────┴─────────────┘

                  ┌────┬────┬────┬────┬────┬────┬────┬────┬─────────────┐
Row 0 (table_idx=0)│501 │502 │503 │504 │505 │    │    │    │             │  page_table
                  ├────┼────┼────┼────┼────┼────┼────┼────┼─────────────┤  (KV cache indices)
Row 1 (table_idx=1)│100 │101 │102 │103 │    │    │    │    │             │
                  ├────┼────┼────┼────┼────┼────┼────┼────┼─────────────┤
Row 2 (table_idx=2)│200 │201 │202 │203 │204 │205 │206 │    │             │
                  └────┴────┴────┴────┴────┴────┴────┴────┴─────────────┘
```

### How `Req.table_idx` Maps to Tables

For a request with `table_idx=2`, `cached_len=3`, `device_len=7`:

```
token_pool[2, 0:7]  → [99, 34, 56, 78, 11, 22, 33]  (token IDs)
page_table[2, 0:7]  → [200, 201, 202, 203, 204, 205, 206]  (KV cache pages)

                     ├── cached ──┤├── extend ──────┤
                       (reuse KV)    (compute new KV)
```

---

## 4. How the Scheduler Uses These Structures

### Step-by-Step: `_prepare_batch()`

```python
def _prepare_batch(self, batch: Batch) -> ForwardInput:
    # 1. Allocate new pages for tokens that need KV computation
    needed_size = sum(r.extend_len for r in batch.reqs)
    batch.out_loc = self.cache_manager.allocate(needed_size)
    
    # 2. Compute indices for loading token IDs from token_pool
    load_indices = _make_2d_indices(
        self.token_pool, 
        [(r.table_idx, r.cached_len, r.device_len) for r in batch.padded_reqs]
    )
    
    # 3. Compute indices for writing next token IDs
    write_indices = _make_2d_indices(
        self.token_pool, 
        [(r.table_idx, r.device_len, r.device_len + 1) for r in batch.reqs]
    )
    
    # 4. Write allocated page indices to page_table
    self.page_table.view(-1)[load_indices] = batch.out_loc
    
    # 5. Prepare attention metadata
    self.engine.attn_backend.prepare_metadata(batch)
```

### Visual: Index Computation

```
Batch with 2 requests:
  Req A: table_idx=0, cached_len=2, device_len=5
  Req B: table_idx=1, cached_len=0, device_len=3

load_indices for token_pool (flatten 2D → 1D):
  Req A: positions [2, 3, 4] in row 0 → [2, 3, 4]
  Req B: positions [0, 1, 2] in row 1 → [max_seq_len+0, max_seq_len+1, max_seq_len+2]
  
out_loc (newly allocated pages): [page_500, page_501, page_502, page_503, page_504]
                                   └─── Req A (3 tokens) ───┘└── Req B (3 tokens) ──┘
```

---

## 5. Forward Pass Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Forward Pass                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Load token IDs from token_pool using load_indices            │
│     batch.input_ids = token_pool.view(-1)[load_indices]          │
│                                                                  │
│  2. Model forward pass                                           │
│     ┌────────────────────────────────────────────────────────┐  │
│     │  For each layer:                                        │  │
│     │    q, k, v = self.qkv_proj(hidden_states)              │  │
│     │                                                         │  │
│     │    # Store new KV to cache at out_loc positions        │  │
│     │    kv_cache.store_kv(k, v, batch.out_loc, layer_id)    │  │
│     │                                                         │  │
│     │    # Load full KV from cache using page_table          │  │
│     │    # (includes both cached and new KV)                 │  │
│     │    attention(q, k_cache, v_cache, page_table[req])     │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  3. Sample next tokens                                           │
│     next_tokens = sampler.sample(logits)                         │
│                                                                  │
│  4. Write next tokens to token_pool using write_indices          │
│     token_pool.view(-1)[write_indices] = next_tokens             │
│                                                                  │
│  5. Update request state                                         │
│     for req in batch.reqs:                                       │
│         req.complete_one()  # cached_len = device_len            │
│                             # device_len += 1                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Complete Example: Two Decode Steps

```
Initial State:
┌─────────────────────────────────────────────────────────────────┐
│  Req(table_idx=0, cached_len=5, device_len=5, max_device_len=10)│
│                                                                  │
│  token_pool[0]:  [42, 17, 5, 89, 23, __, __, __, __, __]        │
│  page_table[0]:  [100,101,102,103,104, __, __, __, __, __]      │
│                   ├───── cached ─────┤                          │
│  extend_len = 0 (nothing to compute in prefill, already done)   │
└─────────────────────────────────────────────────────────────────┘

Step 1 - Decode:
┌─────────────────────────────────────────────────────────────────┐
│  1. Scheduler sees extend_len=1 (position 5 needs computation)  │
│  2. Allocates page 105 for new KV                               │
│  3. load_indices points to token_pool[0, 5]                     │
│  4. page_table[0, 5] = 105                                      │
│  5. Forward pass → sample token 77                              │
│  6. write_indices points to token_pool[0, 6]                    │
│  7. token_pool[0, 5] was already filled during prefill          │
│     token_pool[0, 6] = 77  (write next token for next step)     │
│  8. req.complete_one() → cached_len=6, device_len=6             │
│                                                                  │
│  token_pool[0]:  [42, 17, 5, 89, 23, X, 77, __, __, __]         │
│  page_table[0]:  [100,101,102,103,104,105, __, __, __, __]      │
│                   ├────── cached ──────┤                         │
└─────────────────────────────────────────────────────────────────┘

Step 2 - Decode:
┌─────────────────────────────────────────────────────────────────┐
│  1. extend_len=1 (position 6)                                   │
│  2. Allocates page 106                                          │
│  3. page_table[0, 6] = 106                                      │
│  4. Forward with token 77 → sample token 33                     │
│  5. token_pool[0, 7] = 33                                       │
│  6. req.complete_one() → cached_len=7, device_len=7             │
│                                                                  │
│  token_pool[0]:  [42, 17, 5, 89, 23, X, 77, 33, __, __]         │
│  page_table[0]:  [100,101,102,103,104,105,106, __, __, __]      │
│                   ├─────── cached ───────────┤                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

| Structure | Purpose | Key Fields |
|-----------|---------|------------|
| **`Req`** | Single request state | `table_idx` (row), `cached_len`/`device_len` (columns) |
| **`Batch`** | Grouped requests for batched inference | `reqs`, `out_loc` (new page allocations), `phase` |
| **`token_pool`** | 2D table storing token IDs per request | `[max_reqs, max_seq_len]` |
| **`page_table`** | 2D table mapping positions to KV cache pages | `[max_reqs, max_seq_len]` |
| **`table_idx`** | Request's row in both tables | Assigned at prefill time |
| **`cached_len`** | Columns already in KV cache | Updated after each step |
| **`device_len`** | Current sequence length | Grows during decode |

**Key insight:** The `table_idx` creates a **logical address space** per request, while the `page_table` provides **indirection** to physical KV cache pages. This enables:
- Prefix sharing (same page indices for common prefixes)
- Non-contiguous allocation
- Efficient batching of variable-length sequences

```plaintext
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Memory Layout                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TableManager (per-request slots)              CacheManager (KV pages)       │
│  ┌──────────────────────────────────┐         ┌───────────────────────────┐ │
│  │  token_pool    │   page_table    │         │     KV Cache Buffer       │ │
│  │  [R, S]        │   [R, S]        │         │  [2, L, P, H, D]          │ │
│  │  (token ids)   │   (page indices)│   ───►  │  (actual K/V tensors)     │ │
│  └──────────────────────────────────┘         └───────────────────────────┘ │
│         ▲                  ▲                            ▲                    │
│         │                  │                            │                    │
│         │    table_idx     │                     page indices                │
│         │         ▼        │                            │                    │
│         │    ┌─────────────┴───────────────────────────────┐                │
│         │    │                    Req                       │                │
│         │    │  table_idx ──► row in token_pool/page_table  │                │
│         │    │  cached_len, device_len ──► column ranges    │                │
│         │    └──────────────────────────────────────────────┘                │
│         │                          │                                         │
│         │                          ▼                                         │
│         │                   ┌────────────┐                                  │
│         │                   │   Batch    │                                  │
│         │                   │  reqs: []  │                                  │
│         │                   │  out_loc   │  ◄── newly allocated pages       │
│         │                   └────────────┘                                  │
│         │                                                                    │
│  R = max_running_reqs                                                        │
│  S = max_seq_len                                                             │
│  L = num_layers, P = num_pages, H = num_heads, D = head_dim                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
class Req:
    def __init__(
        self,
        *,
        input_ids: torch.Tensor,   # Token IDs (on CPU)
        table_idx: int,            # Row index in page_table/token_pool
        cached_len: int,           # Tokens already in KV cache (from prefix matching)
        output_len: int,           # Maximum tokens to generate
        uid: int,                  # Unique request identifier
        sampling_params: SamplingParams,
        cache_handle: BaseCacheHandle,  # Handle to locked prefix in radix tree
    ):
        self.host_ids = input_ids          # All token IDs (CPU)
        self.table_idx = table_idx         # Which row in the 2D tables
        self.cached_len = cached_len       # KV already computed
        self.device_len = len(input_ids)   # Current sequence length
        self.max_device_len = len(input_ids) + output_len  # Maximum length
```

```plaintext
Initial state (prefill with prefix cache hit):
┌─────────────────────────────────────────────────────────────────┐
│ Token IDs: ["Hello", "world", ",", "how", "are", "you", "?"]    │
│                                                                  │
│  cached_len=3     device_len=7                max_device_len=17 │
│  (from cache)     (input length)              (+10 output)      │
│       │                 │                           │           │
│       ▼                 ▼                           ▼           │
│  [█ █ █ ░ ░ ░ ░ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯]                          │
│   ▲▲▲   ▲▲▲▲                                                    │
│   cached  extend_len=4                                           │
│           (need to compute)                                      │
└─────────────────────────────────────────────────────────────────┘

After one decode step (complete_one()):
┌─────────────────────────────────────────────────────────────────┐
│  cached_len=7     device_len=8                                  │
│       │                 │                                       │
│       ▼                 ▼                                       │
│  [█ █ █ █ █ █ █ ░ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯]                          │
│   ▲▲▲▲▲▲▲     ▲                                                 │
│   all cached  extend_len=1 (new token)                          │
└─────────────────────────────────────────────────────────────────┘

Legend: █ = cached in KV cache, ░ = being computed, ▯ = future tokens
```

```python
def complete_one(self) -> None:
    self.cached_len = self.device_len  # All current tokens now cached
    self.device_len += 1               # Sequence grows by 1
```

```python
class Batch:
    def __init__(self, *, reqs: List[Req], phase: Literal["prefill", "decode"]):
        self.reqs = reqs              # Actual requests
        self.phase = phase            # "prefill" or "decode"
        
        # Set by scheduler:
        self.input_ids: torch.Tensor   # Flattened token IDs for all reqs
        self.out_loc: torch.Tensor     # Page indices for new KV entries
        self.padded_reqs: List[Req]    # Reqs + dummy reqs for CUDA graph
        
        # Set by attention backend:
        self.attn_metadata: BaseAttnMetadata  # Attention-specific data
```

```python
class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor):
        self._free_slots = list(range(max_running_reqs))  # Available row indices
        self.page_table = page_table    # [max_running_reqs, max_seq_len] → page indices
        self.token_pool = torch.empty_like(page_table)  # [max_running_reqs, max_seq_len] → token IDs
```

```plaintext
                    Column Index (Position in Sequence)
                    0    1    2    3    4    5    6    7    ...  max_seq_len
                  ┌────┬────┬────┬────┬────┬────┬────┬────┬─────────────┐
Row 0 (table_idx=0)│ 42 │ 17 │  5 │ 89 │ 23 │    │    │    │             │  token_pool
                  ├────┼────┼────┼────┼────┼────┼────┼────┼─────────────┤  (token IDs)
Row 1 (table_idx=1)│ 12 │ 88 │ 45 │ 67 │    │    │    │    │             │
                  ├────┼────┼────┼────┼────┼────┼────┼────┼─────────────┤
Row 2 (table_idx=2)│ 99 │ 34 │ 56 │ 78 │ 11 │ 22 │ 33 │    │             │
                  └────┴────┴────┴────┴────┴────┴────┴────┴─────────────┘

                  ┌────┬────┬────┬────┬────┬────┬────┬────┬─────────────┐
Row 0 (table_idx=0)│501 │502 │503 │504 │505 │    │    │    │             │  page_table
                  ├────┼────┼────┼────┼────┼────┼────┼────┼─────────────┤  (KV cache indices)
Row 1 (table_idx=1)│100 │101 │102 │103 │    │    │    │    │             │
                  ├────┼────┼────┼────┼────┼────┼────┼────┼─────────────┤
Row 2 (table_idx=2)│200 │201 │202 │203 │204 │205 │206 │    │             │
                  └────┴────┴────┴────┴────┴────┴────┴────┴─────────────┘
```

```plaintext
token_pool[2, 0:7]  → [99, 34, 56, 78, 11, 22, 33]  (token IDs)
page_table[2, 0:7]  → [200, 201, 202, 203, 204, 205, 206]  (KV cache pages)

                     ├── cached ──┤├── extend ──────┤
                       (reuse KV)    (compute new KV)
```

```python
def _prepare_batch(self, batch: Batch) -> ForwardInput:
    # 1. Allocate new pages for tokens that need KV computation
    needed_size = sum(r.extend_len for r in batch.reqs)
    batch.out_loc = self.cache_manager.allocate(needed_size)
    
    # 2. Compute indices for loading token IDs from token_pool
    load_indices = _make_2d_indices(
        self.token_pool, 
        [(r.table_idx, r.cached_len, r.device_len) for r in batch.padded_reqs]
    )
    
    # 3. Compute indices for writing next token IDs
    write_indices = _make_2d_indices(
        self.token_pool, 
        [(r.table_idx, r.device_len, r.device_len + 1) for r in batch.reqs]
    )
    
    # 4. Write allocated page indices to page_table
    self.page_table.view(-1)[load_indices] = batch.out_loc
    
    # 5. Prepare attention metadata
    self.engine.attn_backend.prepare_metadata(batch)
```

```plaintext
Batch with 2 requests:
  Req A: table_idx=0, cached_len=2, device_len=5
  Req B: table_idx=1, cached_len=0, device_len=3

load_indices for token_pool (flatten 2D → 1D):
  Req A: positions [2, 3, 4] in row 0 → [2, 3, 4]
  Req B: positions [0, 1, 2] in row 1 → [max_seq_len+0, max_seq_len+1, max_seq_len+2]
  
out_loc (newly allocated pages): [page_500, page_501, page_502, page_503, page_504]
                                   └─── Req A (3 tokens) ───┘└── Req B (3 tokens) ──┘
```

```plaintext
┌─────────────────────────────────────────────────────────────────┐
│                     Forward Pass                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Load token IDs from token_pool using load_indices            │
│     batch.input_ids = token_pool.view(-1)[load_indices]          │
│                                                                  │
│  2. Model forward pass                                           │
│     ┌────────────────────────────────────────────────────────┐  │
│     │  For each layer:                                        │  │
│     │    q, k, v = self.qkv_proj(hidden_states)              │  │
│     │                                                         │  │
│     │    # Store new KV to cache at out_loc positions        │  │
│     │    kv_cache.store_kv(k, v, batch.out_loc, layer_id)    │  │
│     │                                                         │  │
│     │    # Load full KV from cache using page_table          │  │
│     │    # (includes both cached and new KV)                 │  │
│     │    attention(q, k_cache, v_cache, page_table[req])     │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  3. Sample next tokens                                           │
│     next_tokens = sampler.sample(logits)                         │
│                                                                  │
│  4. Write next tokens to token_pool using write_indices          │
│     token_pool.view(-1)[write_indices] = next_tokens             │
│                                                                  │
│  5. Update request state                                         │
│     for req in batch.reqs:                                       │
│         req.complete_one()  # cached_len = device_len            │
│                             # device_len += 1                    │
└─────────────────────────────────────────────────────────────────┘
```

```plaintext
Initial State:
┌─────────────────────────────────────────────────────────────────┐
│  Req(table_idx=0, cached_len=5, device_len=5, max_device_len=10)│
│                                                                  │
│  token_pool[0]:  [42, 17, 5, 89, 23, __, __, __, __, __]        │
│  page_table[0]:  [100,101,102,103,104, __, __, __, __, __]      │
│                   ├───── cached ─────┤                          │
│  extend_len = 0 (nothing to compute in prefill, already done)   │
└─────────────────────────────────────────────────────────────────┘

Step 1 - Decode:
┌─────────────────────────────────────────────────────────────────┐
│  1. Scheduler sees extend_len=1 (position 5 needs computation)  │
│  2. Allocates page 105 for new KV                               │
│  3. load_indices points to token_pool[0, 5]                     │
│  4. page_table[0, 5] = 105                                      │
│  5. Forward pass → sample token 77                              │
│  6. write_indices points to token_pool[0, 6]                    │
│  7. token_pool[0, 5] was already filled during prefill          │
│     token_pool[0, 6] = 77  (write next token for next step)     │
│  8. req.complete_one() → cached_len=6, device_len=6             │
│                                                                  │
│  token_pool[0]:  [42, 17, 5, 89, 23, X, 77, __, __, __]         │
│  page_table[0]:  [100,101,102,103,104,105, __, __, __, __]      │
│                   ├────── cached ──────┤                         │
└─────────────────────────────────────────────────────────────────┘

Step 2 - Decode:
┌─────────────────────────────────────────────────────────────────┐
│  1. extend_len=1 (position 6)                                   │
│  2. Allocates page 106                                          │
│  3. page_table[0, 6] = 106                                      │
│  4. Forward with token 77 → sample token 33                     │
│  5. token_pool[0, 7] = 33                                       │
│  6. req.complete_one() → cached_len=7, device_len=7             │
│                                                                  │
│  token_pool[0]:  [42, 17, 5, 89, 23, X, 77, 33, __, __]         │
│  page_table[0]:  [100,101,102,103,104,105,106, __, __, __]      │
│                   ├─────── cached ───────────┤                   │
└─────────────────────────────────────────────────────────────────┘
```


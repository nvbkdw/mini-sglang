# RadixCacheManager unit tests
_Exported on 12/30/2025 at 21:20:21 PST from Cursor (2.1.46)_

---

**User**

@ @mini-sglang/python/minisgl/kvcache/mha_pool.py:1-80 when does the dataflow and cache manager evict paged kvcache and allocate cache for new tokens?



how doe MHAKVCache integrate with @mini-sglang/python/minisgl/kvcache/radix_manager.py  cache manager

---

**Cursor**

Let me explore the codebase to understand how `MHAKVCache` and `RadixCacheManager` integrate and when cache eviction/allocation happens.

Now I have a complete picture. Let me explain the architecture and data flow:

## Architecture Overview

The system has **two separate but coordinated components**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Scheduler                                  │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    CacheManager (scheduler/cache.py)            ││
│  │  ┌──────────────────┐    ┌─────────────────────────────────────┐││
│  │  │  _free_slots     │    │  RadixCacheManager (or Naive)       │││
│  │  │  [page indices]  │    │  - Tracks token→page mappings       │││
│  │  │                  │    │  - Manages prefix tree              │││
│  │  │  (physical page  │    │  - Handles eviction decisions       │││
│  │  │   allocator)     │    │  - LRU-based prefix sharing         │││
│  │  └──────────────────┘    └─────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           Engine                                    │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │              MHAKVCache (kvcache/mha_pool.py)                   ││
│  │  - Actual GPU tensor storage for K/V                            ││
│  │  - Shape: (2, num_layers, num_pages, num_kv_heads, head_dim)    ││
│  │  - store_kv() writes K/V to specified page indices              ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

## Key Insight: Separation of Concerns

| Component | Role |
|-----------|------|
| **MHAKVCache** | Physical storage - a big GPU tensor holding K/V values |
| **RadixCacheManager** | Logical metadata - tracks which token sequences map to which page indices |
| **CacheManager** (scheduler) | Orchestrates both - manages free page list + calls RadixCacheManager |

## When Does Eviction/Allocation Happen?

### 1. **Allocation** - `CacheManager.allocate()` (scheduler/cache.py:39-52)

```python:39:52:mini-sglang/python/minisgl/scheduler/cache.py
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

**Called when:** Scheduler prepares a batch for forward pass (`_prepare_batch`).

**Flow:**
1. First tries to use free slots from `_free_slots` 
2. If not enough → calls `RadixCacheManager.evict()` to reclaim LRU prefixes
3. Returns page indices where new K/V will be stored

### 2. **Eviction** - `RadixCacheManager.evict()` (radix_manager.py:166-193)

**Triggered by:** `allocate()` when free slots are insufficient.

**Strategy:** LRU-based eviction of leaf nodes (oldest timestamps first).

**Returns:** Page indices that are now free (these were storing K/V for evicted prefixes).

### 3. **Prefix Matching & Locking** - During request scheduling

```python:38:61:mini-sglang/python/minisgl/scheduler/prefill.py
def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
    # ...
    handle, match_indices = self.cache_manager.match_req(req)
    cached_len = handle.cached_len
    # ...
    self.cache_manager.lock(handle)  # Protect from eviction
    # ...
```

When a new request arrives:
1. `match_prefix()` finds if any cached prefix matches the input
2. Returns `(handle, indices)` - handle references the radix tree node, indices are page locations
3. `lock_handle()` protects the matched prefix from eviction during use

### 4. **Caching Completed Requests** - `free_and_cache_finished_req()`

```python:54:62:mini-sglang/python/minisgl/scheduler/cache.py
def free_and_cache_finished_req(
    self, old_handle: BaseCacheHandle, input_ids: torch.Tensor, indices: torch.Tensor,
) -> None:
    in_cache_len = self.manager.insert_prefix(input_ids, indices)
    self._free(indices[old_handle.cached_len : in_cache_len])
    self.unlock(old_handle)
```

After a request completes:
1. `insert_prefix()` adds the full sequence to the radix tree (for future reuse)
2. Frees any redundant indices (already cached by another request)
3. Unlocks the handle (making it evictable for LRU)

## Integration Flow Diagram

```
New Request Arrives
        │
        ▼
┌───────────────────┐
│ match_prefix()    │◄── RadixCacheManager searches tree
│ Returns handle +  │    for matching prefix
│ page indices      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ lock_handle()     │◄── Protects prefix from eviction
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ allocate()        │◄── Gets pages for new tokens
│ (may evict LRU)   │    MHAKVCache pages are recycled
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Forward Pass      │◄── Model runs, MHAKVCache.store_kv()
│ writes K/V        │    fills allocated pages
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ insert_prefix()   │◄── RadixCacheManager records
│ + unlock()        │    new sequence for future reuse
└───────────────────┘
```

## Key Points

1. **MHAKVCache doesn't know about eviction** - it's just storage. The RadixCacheManager decides WHICH pages to reuse.

2. **Page indices are the bridge** - RadixCacheManager tracks `token_ids → page_indices`, MHAKVCache stores `page_indices → K/V tensors`.

3. **Lazy eviction** - Eviction only happens when `allocate()` can't satisfy the request from free slots.

4. **Prefix sharing** - Multiple requests with the same prefix share the same pages (RadixCacheManager returns the same indices).

```plaintext
┌─────────────────────────────────────────────────────────────────────┐
│                          Scheduler                                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    CacheManager (scheduler/cache.py)            ││
│  │  ┌──────────────────┐    ┌─────────────────────────────────────┐││
│  │  │  _free_slots     │    │  RadixCacheManager (or Naive)       │││
│  │  │  [page indices]  │    │  - Tracks token→page mappings       │││
│  │  │                  │    │  - Manages prefix tree              │││
│  │  │  (physical page  │    │  - Handles eviction decisions       │││
│  │  │   allocator)     │    │  - LRU-based prefix sharing         │││
│  │  └──────────────────┘    └─────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           Engine                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │              MHAKVCache (kvcache/mha_pool.py)                   ││
│  │  - Actual GPU tensor storage for K/V                            ││
│  │  - Shape: (2, num_layers, num_pages, num_kv_heads, head_dim)   ││
│  │  - store_kv() writes K/V to specified page indices              ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

```plaintext
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

```plaintext
def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
    # ...
    handle, match_indices = self.cache_manager.match_req(req)
    cached_len = handle.cached_len
    # ...
    self.cache_manager.lock(handle)  # Protect from eviction
    # ...
```

```plaintext
def free_and_cache_finished_req(
    self, old_handle: BaseCacheHandle, input_ids: torch.Tensor, indices: torch.Tensor,
) -> None:
    in_cache_len = self.manager.insert_prefix(input_ids, indices)
    self._free(indices[old_handle.cached_len : in_cache_len])
    self.unlock(old_handle)
```

```plaintext
New Request Arrives
        │
        ▼
┌───────────────────┐
│ match_prefix()    │◄── RadixCacheManager searches tree
│ Returns handle +  │    for matching prefix
│ page indices      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ lock_handle()     │◄── Protects prefix from eviction
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ allocate()        │◄── Gets pages for new tokens
│ (may evict LRU)   │    MHAKVCache pages are recycled
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Forward Pass      │◄── Model runs, MHAKVCache.store_kv()
│ writes K/V        │    fills allocated pages
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ insert_prefix()   │◄── RadixCacheManager records
│ + unlock()        │    new sequence for future reuse
└───────────────────┘
```

---

**User**

how does attention back read from kvcache pages that is not contiguous tensor?

---

**Cursor**

Great question! Let me explore how the attention backend handles non-contiguous paged KV cache.

Now I understand the mechanism. Let me show you how paged attention handles non-contiguous KV cache:

## The Key: Page Table as Indirection Layer

The system uses **FlashInfer** (a paged attention library) that accepts:
1. **KV cache tensors** - the actual storage (non-contiguous in logical sequence order)
2. **Page indices** - telling which pages belong to which sequence

### Page Table Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                        page_table                                │
│              Shape: (max_requests, max_seq_len)                  │
├─────────────────────────────────────────────────────────────────┤
│  Request 0: [page_42, page_17, page_203, page_8, ...]           │
│  Request 1: [page_5, page_99, page_12, ...]                      │
│  Request 2: [page_42, page_17, page_77, ...]  ← shares prefix!  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MHAKVCache                                   │
│              Shape: (num_pages, num_heads, head_dim)             │
├─────────────────────────────────────────────────────────────────┤
│  Page 0:  [K/V for some token]                                   │
│  Page 1:  [K/V for some token]                                   │
│  ...                                                             │
│  Page 42: [K/V for token X]  ← used by Request 0 AND 2          │
│  ...                                                             │
└─────────────────────────────────────────────────────────────────┘
```

### How FlashInfer Reads Non-Contiguous Pages

The magic happens in `prepare_metadata()`:

```python:215:215:mini-sglang/python/minisgl/attention/fi.py
indices=torch.cat([self.page_table[req.table_idx, : req.device_len] for req in reqs]),
```

This creates a **flattened list of page indices** for all requests:

```
indices = [page_42, page_17, page_203, page_8,   # Request 0's pages
           page_5, page_99, page_12,              # Request 1's pages  
           page_42, page_17, page_77, ...]        # Request 2's pages
```

Combined with **cumulative sequence lengths** (`cu_seqlens_k`):

```python:203:203:mini-sglang/python/minisgl/attention/fi.py
cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **cpu_kwargs).cumsum_(dim=0)
```

```
cu_seqlens_k = [0, 4, 7, 10, ...]
                │  │  │
                │  │  └─ Request 2 starts at index 7
                │  └─ Request 1 starts at index 4
                └─ Request 0 starts at index 0
```

### FlashInfer's `plan()` + `run()` Pattern

```python:181:189:mini-sglang/python/minisgl/attention/fi.py
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

**Step 1: `plan()`** (called in `_initialize_metadata_once`):
- Prepares internal data structures for gathering pages
- Takes `indices` (page list) + `indptr` (cumulative lengths)

**Step 2: `run()`**:
- FlashInfer internally uses the indices to **gather** K/V from scattered pages
- Performs attention with proper masking

### Visual Flow

```
Query for Request 2 (positions 0,1,2)
              │
              ▼
┌─────────────────────────────────────┐
│  FlashInfer Kernel                  │
│                                     │
│  indices[7:10] = [42, 17, 77]       │
│                    │   │   │        │
│                    ▼   ▼   ▼        │
│  Gather K from:  ┌───┬───┬───┐      │
│    kv_cache[42]  │K₀ │K₁ │K₂ │      │
│    kv_cache[17]  └───┴───┴───┘      │
│    kv_cache[77]                     │
│                                     │
│  Same for V, then compute:          │
│  softmax(Q @ K^T) @ V               │
└─────────────────────────────────────┘
```

### Key Implementation Details

| Component | Purpose |
|-----------|---------|
| `page_table[req.table_idx]` | Maps request → its page indices |
| `cu_seqlens_k` (indptr) | Tells where each request's pages start in flattened indices |
| `indices` | Flattened page indices for all requests |
| FlashInfer kernel | Uses indices to gather K/V from arbitrary pages |

### Why This Works Efficiently

1. **GPU-native gather**: FlashInfer's CUDA kernels perform the page gathering inside the attention computation - no separate gather step needed.

2. **Fused operations**: The gather, attention computation, and masking happen in a single fused kernel.

3. **Memory coalescing**: FlashInfer is optimized for this access pattern, using techniques like:
   - Tiled loading of K/V pages
   - Shared memory buffering
   - Warp-level parallelism

This is why **paged attention** is so powerful - it enables non-contiguous memory allocation while maintaining high GPU utilization through specialized kernels that understand the indirection pattern.

```plaintext
┌─────────────────────────────────────────────────────────────────┐
│                        page_table                                │
│              Shape: (max_requests, max_seq_len)                  │
├─────────────────────────────────────────────────────────────────┤
│  Request 0: [page_42, page_17, page_203, page_8, ...]           │
│  Request 1: [page_5, page_99, page_12, ...]                      │
│  Request 2: [page_42, page_17, page_77, ...]  ← shares prefix!  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MHAKVCache                                   │
│              Shape: (num_pages, num_heads, head_dim)             │
├─────────────────────────────────────────────────────────────────┤
│  Page 0:  [K/V for some token]                                   │
│  Page 1:  [K/V for some token]                                   │
│  ...                                                             │
│  Page 42: [K/V for token X]  ← used by Request 0 AND 2          │
│  ...                                                             │
└─────────────────────────────────────────────────────────────────┘
```

```plaintext
indices=torch.cat([self.page_table[req.table_idx, : req.device_len] for req in reqs]),
```

```plaintext
indices = [page_42, page_17, page_203, page_8,   # Request 0's pages
           page_5, page_99, page_12,              # Request 1's pages  
           page_42, page_17, page_77, ...]        # Request 2's pages
```

```plaintext
cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **cpu_kwargs).cumsum_(dim=0)
```

```plaintext
cu_seqlens_k = [0, 4, 7, 10, ...]
                │  │  │
                │  │  └─ Request 2 starts at index 7
                │  └─ Request 1 starts at index 4
                └─ Request 0 starts at index 0
```

```plaintext
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

```plaintext
Query for Request 2 (positions 0,1,2)
              │
              ▼
┌─────────────────────────────────────┐
│  FlashInfer Kernel                  │
│                                     │
│  indices[7:10] = [42, 17, 77]       │
│                    │   │   │        │
│                    ▼   ▼   ▼        │
│  Gather K from:  ┌───┬───┬───┐      │
│    kv_cache[42]  │K₀ │K₁ │K₂ │      │
│    kv_cache[17]  └───┴───┴───┘      │
│    kv_cache[77]                     │
│                                     │
│  Same for V, then compute:          │
│  softmax(Q @ K^T) @ V               │
└─────────────────────────────────────┘
```


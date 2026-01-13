# Paged Attention: CPU Implementation & Hardware Utilization

## 1. The Problem: Memory Fragmentation

In Large Language Model (LLM) serving, **KV Cache** management is a critical bottleneck. The traditional "Naive" approach allocates a large contiguous chunk of memory for the maximum possible sequence length (e.g., 2048 tokens) for every request. This leads to severe efficiency problems known as **Memory Fragmentation**.

### 1.1. Naive vs. Paged Attention

The core difference lies in how "holes" in memory are handled.

| Type | Naive (Contiguous) | Paged Attention (Non-Contiguous) |
| :--- | :--- | :--- |
| **Internal Fragmentation** | **Severe**: If a request reserves 2048 tokens but uses only 100, the remaining 1948 slots are **wasted**. | **Minimal**: We allocate only one small block (e.g., 16 tokens) at a time. Waste is limited to the unused part of the *last* block. |
| **External Fragmentation** | **High**: Requires huge *contiguous* holes. Even if 10GB is free total, allocation fails if memory is fragmented. | **Zero**: No need for contiguous holes. Any free block anywhere in RAM can be used. |
| **Utilization** | Often < 60% due to reservation. | Near 100% due to on-demand allocation. |

### 1.2. Visualization: Internal Fragmentation

Imagine we have memory for 10 tokens. Request A reserves max length (5) but uses only 2.

```
[ A1 ] [ A2 ] [ -- ] [ -- ] [ -- ] [ Free ] [ Free ] [ Free ] [ Free ] [ Free ]
^----------- Reserved -----------^
```

* **Wasted**: 3 slots (`--`) are locked.
* **Blocked**: Request B needs 6 tokens. Even though 8 slots are effectively free, B cannot fit because the free space is split.

### 1.3. Visualization: External Fragmentation ("Swiss Cheese" Problem)

Imagine requests arriving and finishing over time. We have 10GB free total. Request C needs 1GB (contiguous).

**Naive Approach**:

```
[ Free 200MB ] [ Req A (Active) ] [ Free 500MB ] [ Req B (Active) ] [ Free 300MB ] ...
```

* **Result**: Request C **FAILS** (OOM). The largest contiguous chunk is only 500MB.

**Paged Attention Approach**:
Request C needs 1GB but can be split into many small blocks.

```
[ Used by C ] [ Req A (Active) ] [ Used by C ] [ Req B (Active) ] [ Used by C ] ...
```

* **Result**: Request C **SUCCEEDS**. We fill every small gap with parts of Request C.

---

## 2. The Hardware Challenge: CPU Caching & Latency

**Paged Attention** solves the memory problem by borrowing concepts from OS **Virtual Memory**. However, breaking contiguous memory introduces CPU hardware inefficiencies.

### 2.1. Memory Hierarchy Overview

* **Registers**: Fastest, immediate data.
* **L1/L2 Cache**: Fast, small (KB-MB). Heavily relies on spatial locality.
* **Main Memory (RAM)**: Slow, massive.

### 2.2. The Cost of Indirection

In the Naive approach, `K[t]` and `K[t+1]` are physically adjacent (`ptr++`).
In **Paged Attention**, `K[t]` and `K[t+1]` might be in completely different physical blocks.

#### Visualization: Cache Hit vs. Miss

```
[ Block A (Physical Addr: 0x1000) ]       [ Block B (Physical Addr: 0x9000) ]
| ... | Token 15 |                        | Token 16 | ... |
+----------------+                        +----------+
        |                                      ^
        | (Next Token)                         |
        +--------------------------------------+
               HUGE JUMP (Cache Miss!)
```

1. **Intra-Block (Token 14 -> 15)**: `ptr++`. Physically adjacent. **Cache Hit**.
2. **Inter-Block (Token 15 -> 16)**:
    * **Translation**: Look up Block Table -> "Physical Block B" (Extra Load).
    * **Physical Jump**: 0x1000 -> 0x9000.
    * **Result**: The CPU prefetcher cannot predict this jump, causing a stall (RAM Latency).

### 2.3. Cache Prefetching Mechanics

Modern CPUs read data in **Cache Lines** (typically 64 bytes).

#### Why Intra-Block Access is Fast

When you access `float` at index 0, the CPU fetches the entire 64-byte line (16 floats).

```
[ L1 Cache Line (64 Bytes) ]
-----------------------------------------------------------------------
| Float 0 | Float 1 | Float 2 | ... | Float 15 | (Next Line needed...)
-----------------------------------------------------------------------
    ^         ^
    |         |
  Request    Hit! (Already in cache)
```

* **Hit Rate**: Accessing floats 1-15 is free (Cached).
* **Prefetching**: Hardware detects the linear pattern (`i, i+1`) and fetches the *next* line before you ask.

#### Prefetcher Failure on Block Boundaries

* **Sequential Prefetching**: Works inside a block.
* **Aggressive Prefetching**: If the prefetch distance is too large, it may cross the block boundary and fetch **garbage data** from a physically adjacent but logically unrelated block, wasting bandwidth.

### 2.4. Physical Proximity & Fragmentation

A common question: *"Aren't consecutive blocks likely to be physically close anyway?"*
**Answer: No.**

Due to memory fragmentation (Parking Lot Analogy), logical blocks are randomly scattered.

* Logical Block 0 -> Physical Slot 5
* Logical Block 1 -> Physical Slot 200 (Slots 6-199 were taken by others)

We must assume **Zero Spatial Locality** between blocks.

### 2.5. The Trade-off: Why Small Block Size is Bad

If the Cache Line is 64 bytes, why not make the Block Size 64 bytes (1 Token) to eliminate all internal fragmentation?

This would be disastrous due to **Prefetcher Stalls** and **TLB Misses**.

| Feature | Naive (Contiguous) | Paged Attention (Small Block) |
| :--- | :--- | :--- |
| **Prefetcher** | **Proactive**: Latency ~ 0. | **Disabled**: CPU cannot predict jumps. Waits for full DRAM latency. |
| **Address Calculation** | **Fast (ALU)**: `Base + Index`. | **Slow (Memory)**: Must load `BlockTable` first (Pointer Chasing). |
| **TLB (Translation)** | **High Hit Rate**: 1 Huge Page covers 2MB. | **High Miss Rate**: Every small block is on a different 4KB page. |

**Conclusion**: We need a `block_size` (e.g., 16 or 32 tokens) large enough to amortize the jump cost, but small enough to minimize internal fragmentation.

---

## 3. Implementation Details (Nano-vLLM)

Located in `src/memory.c` and `kernels/cpu/kernels.c`.

### 3.1. Data Structures (`include/structs.h`)

#### A. The Physical Memory Pool (`KVCacheManager`)

A single massive array representing all available physical memory.

```c
typedef struct {
    float* pool_k;
    float* pool_v;
    int* free_block_indices; // Stack to track available blocks
} KVCacheManager;
```

**Memory Pool Visualization**
The `pool_k` is a flattened 1D array representing a 5D tensor.

```
[ Physical RAM (One Contiguous Array) ]
--------------------------------------------------------------------------------------
| Block 0      | Block 1      | ... | Block 7      | ... | Block N      |
--------------------------------------------------------------------------------------
|  (Free)      |  (Seq B)     |     |  (Seq A)     |     |  (Free)      |
--------------------------------------------------------------------------------------
^              ^                    ^
Base Ptr       Ptr + Stride         Ptr + 7*Stride
```

**Inside a Single Block (Zoom In)**
Data is packed tightly for SIMD efficiency.

```
[ Block 7 Content (allocated to Seq A) ]
-------------------------------------------------------------
| Token T  | Token T+1 | ... | Token T+15 |
-------------------------------------------------------------
     |
     v
     [ Head 0 | Head 1 | ... | Head K ]
         |
         v
         [ float, float, ... (Contiguous, SIMD Friendly) ]
```

#### B. The Page Table (`BlockTable`)

```c
typedef struct {
    int* block_indices; // Maps logical_block_idx -> physical_block_idx
    int num_blocks;
} BlockTable;
```

### 3.2. Address Translation Logic

This runs on the CPU for every token generation step.

```c
// Simplified Logic
void paged_attention(...) {
    for (int t = 0; t <= pos; t++) {
        // 1. Virtual -> Logical Block
        int logical_block = t / block_size;
        int block_offset  = t % block_size;

        // 2. Logical Block -> Physical Block (Indirection)
        int physical_block = block_table->block_indices[logical_block];

        // 3. Physical Block -> RAM Address
        long offset = get_physical_offset(mgr, physical_block, block_offset, ...);
        float* k_ptr = mgr->pool_k + offset;

        // 4. Compute (SIMD Friendly Inner Loop)
        float score = dot_product(q, k_ptr, head_dim);
    }
}
```

---

## 4. Summary & Benefits

1. **Zero-Copy Reorganization**: Forking sequences (e.g., Beam Search) only requires copying the `BlockTable` integers, not the heavy KV data.
2. **Throughput**: Massive memory savings allow for larger `BATCH_SIZE` (e.g., 4x or 8x concurrent requests).
3. **GPU Preparation**: This exact logic applies to GPU VRAM and CUDA kernels.

| Feature | Naive Implementation | Paged Attention (CPU) |
| :--- | :--- | :--- |
| **Memory Layout** | Contiguous (Virtual & Physical) | Non-contiguous (Physical) |
| **Allocation** | Up-front (Max Seq Len) | On-demand (Per Block) |
| **Access Cost** | Low (Pointer Arithmetic) | Medium (Table Lookup + Stride) |
| **Memory Efficiency** | Low (High Fragmentation) | High (Near Optimal) |
| **Hardware Usage** | Simple Prefetching | Requires Cache Awareness |

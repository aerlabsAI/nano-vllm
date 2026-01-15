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

## 3. Workload Analysis: Prefill vs. Decoding

To fully understand the impact of Paged Attention, we must distinguish between the two phases of LLM inference, as they have different hardware bottlenecks.

### 3.1. Prefill Phase (Prompt Processing)

*   **Operation**: Processes all input tokens in parallel to generate the initial KV cache.
*   **Bottleneck**: **Compute Bound**. The CPU/GPU is saturated by dense Matrix Multiplications (GEMMs) for Q, K, V projections and Attention.
*   **Paged Attention Impact**:
    *   **Writes**: We must allocate and write the initial KV cache into non-contiguous blocks.
    *   **Overhead**: The overhead of block allocation and indirect addressing is usually **negligible** (hidden) compared to the massive computation load.
    *   **Benefit**: Allows processing longer prompts or larger batches without failing due to fragmentation.

### 3.2. Decoding Phase (Token Generation)

*   **Operation**: Generates one token at a time, autoregressively.
*   **Bottleneck**: **Memory Bandwidth Bound**. For each new token, the arithmetic intensity is low (Matrix-Vector multiplication). The speed is limited by how fast we can move the *entire* KV cache from RAM to the CPU cores.
*   **Paged Attention Impact**:
    *   **Reads**: This is where the "Cost of Indirection" (Section 2.2) is most visible. We are reading gigabytes of data per second.
    *   **Throughput vs. Latency**:
        *   *Latency*: Single-request latency might slightly degrade due to non-contiguous reads and prefetcher stalls.
        *   *Throughput*: System throughput increases significantly. By eliminating fragmentation, we can fit more concurrent requests (larger **Batch Size**) into RAM.
    *   **Key Insight**: Since decoding is memory-bound, **wasting memory (fragmentation) = wasting bandwidth**. Paged Attention ensures that every byte of bandwidth transfers useful data, not empty padding.


## 4. Current Implementation: Naive Baseline

The current implementation in `include/core/model.hpp` represents the **Naive (Contiguous)** approach. It allocates the maximum possible memory upfront, which is simple but inefficient.

### 3.1. Data Structures (`include/core/model.hpp`)

The KV Cache is stored as two large contiguous `std::vector<float>` arrays within the `RunState` struct.

```cpp
// Runtime state buffers
struct RunState
{
    // ... (other state variables)

    // KV Cache
    // Layout: [n_layers, max_seq_len, n_kv_heads, head_dim]
    std::vector<float> key_cache;
    std::vector<float> value_cache;
};
```

These vectors are resized once during initialization to hold the maximum possible sequence length for all layers:

```cpp
void resize_run_state()
{
    // ...
    // KV Cache
    size_t cache_size = static_cast<size_t>(config.n_layers) * static_cast<size_t>(config.max_seq_len)
                      * static_cast<size_t>(config.n_kv_heads) * static_cast<size_t>(config.head_dim);

    state.key_cache.resize(cache_size);
    state.value_cache.resize(cache_size);
}
```

### 3.2. Address Calculation

Because the memory is contiguous, we use simple pointer arithmetic (linear addressing) to access the Key/Value vectors. There is no `BlockTable` or virtual-to-physical translation yet.

```cpp
void attention(int layer, int pos, float *out)
{
    // ...
    int layer_offset = layer * config.max_seq_len * n_kv_heads * head_dim;

    for (int h = 0; h < n_heads; h++) {
        // ...
        int kv_h = h / kv_mul; // Handling GQA (Grouped Query Attention)

        // Score Calculation
        for (int t = 0; t <= pos; t++) {
            // Linear Access: Base + (Time Step * Stride)
            float *k_head = state.key_cache.data() + layer_offset + t * n_kv_heads * head_dim + kv_h * head_dim;

            float score = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                score += q_head[i] * k_head[i];
            }
            // ...
        }
        // ...
    }
}
```

---

## 5. Summary & Next Steps

The current codebase implements the **Naive Approach** described in Section 1.

1. **Status**: Functional but memory-inefficient.
2. **Fragmentation**: As shown in Section 1.2, this implementation "locks" `max_seq_len` worth of memory for every request, even if only a few tokens are generated.
3. **Next Goal**: Refactor `RunState` and `attention` logic to use **Paged Attention**, moving from contiguous `std::vector` to a block-based memory pool and page table lookup, as theoretically described in Sections 1 and 2.

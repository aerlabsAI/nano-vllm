# Building an LLM Inference Engine from Scratch (Part 1)

> **How PagedAttention Solves the Memory Fragmentation Problem**

---

## 1. Why did we start the nano-vLLM project?

The nano-vLLM project draws strong inspiration from minimalist implementations such as nanoGPT[^1] by Andrej Karpathy and llm.c[^2]. Rather than targeting state-of-the-art performance, these projects focus on exposing the core principles of complex models and systems through minimal code. By providing implementations that are readable, traceable, and explainable in terms of why they are designed the way they are, they have established themselves as effective educational tools rather than simple toy projects.

Although the LLM inference ecosystem already includes mature open-source engines such as vLLM[^4] and SGLang[^5], and implementations under the name nano-vLLM[^6] already exist, this project is motivated by an educational goal. It aims to help both others and ourselves gain a deeper understanding of LLM inference systems by reimplementing them from scratch.

In our nano-vLLM project, an educational goal means going beyond reading papers or blog posts and understanding how core ideas are realized at the system level through concrete structures and execution flows. With a deep understanding of these structures, debugging issues or tuning performance in real systems becomes significantly more manageable. To support this goal, the project deliberately minimizes performance-driven optimizations, does not mandate the use of CUDA, and excludes complex optimizations that could obscure the underlying concepts. This is not a sacrifice of performance, but a design choice that prioritizes clarity and understanding.

In addition, the field of LLM inference optimization is evolving rapidly. While new techniques and systems continue to emerge, having a solid grasp of the fundamental concepts makes it much easier to learn and adapt to new ideas. We especially study mechanisms like PagedAttention[^3], treating them as worked examples for building intuition. nano-vLLM is a learning-focused project designed to help build this foundation.

---

## 2. Code Structure

```
nano-vllm/
├── include/
│   ├── core/           # Core inference components
│   │   ├── model.hpp       # LlamaModel, weights, forward pass
│   │   ├── attention.hpp   # Standard & Paged Attention
│   │   ├── tokenizer.hpp   # Tokenization
│   │   ├── sampler.hpp     # Token sampling
│   │   └── runner.hpp      # Single-request runner
│   ├── ops/            # Neural network operations
│   │   ├── linear.hpp      # Linear projection
│   │   ├── activation.hpp  # SiLU, etc.
│   │   ├── normalization.hpp # RMSNorm
│   │   └── positional.hpp  # RoPE
│   ├── scheduler/      # Batch scheduling
│   │   ├── scheduler.hpp
│   │   ├── request.hpp
│   │   ├── request_processor.hpp
│   │   ├── block_manager.hpp
│   │   ├── batched_runner.hpp
│   │   └── benchmark.hpp
│   └── utils/          # Utilities
│       ├── logger.hpp
│       ├── metrics.hpp
│       ├── json_parser.hpp
│       ├── argparser.hpp
│       └── path.hpp
└── src/
    └── main.cpp
```

---

## 3. Core Inference Flow

LLM inference consists of two phases. The **Prefill Phase** processes all prompt tokens to build the initial KV cache. The **Decode Phase** then generates tokens one at a time, autoregressively.

```
User Prompt → Tokenize → [Prefill Phase] → [Decode Phase] → Detokenize → Output
                              ↓                    ↓
                         Process all           Generate one
                         prompt tokens         token at a time
```

Each request transitions through states defined in `include/scheduler/request.hpp`: `PENDING → PREFILLING → DECODING → FINISHED` (or `FAILED`). The `RequestProcessor` class (`include/scheduler/request_processor.hpp`) handles the complete lifecycle for a single request:

```cpp
// RequestProcessor handles the complete lifecycle (simplified)
void process(Request &request) {
    // 1. Tokenize prompt
    request.prompt_tokens = tokenizer_.encode(request.prompt, true, false);

    // 2. Prefill: process all prompt tokens
    for (size_t i = 0; i < request.prompt_tokens.size() - 1; i++) {
        model_.forward(request.prompt_tokens[i], pos++);
    }

    // 3. Decode: generate tokens autoregressively
    int token = request.prompt_tokens.back();
    while (request.can_generate_more()) {
        model_.forward(token, request.current_pos++);
        int next_token = sampler.sample(model_.state.logits.data());
        request.generated_tokens.push_back(next_token);
        token = next_token;
    }
}
```

---

## 4. The Problem: Memory Fragmentation

In Large Language Model (LLM) serving, **KV Cache** management is a critical bottleneck. The traditional "Naive" approach allocates a large contiguous chunk of memory for the maximum possible sequence length (e.g., 2048 tokens) for every request. This leads to severe efficiency problems known as **Memory Fragmentation**.

### 4.1. Naive vs. Paged Attention

The core difference lies in how "holes" in memory are handled.

| Type                       | Naive (Contiguous)                                                                                                | Paged Attention (Non-Contiguous)                                                                                                    |
| :------------------------- | :---------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------- |
| **Internal Fragmentation** | **Severe**: If a request reserves 2048 tokens but uses only 100, the remaining 1948 slots are **wasted**.         | **Minimal**: We allocate only one small block (e.g., 16 tokens) at a time. Waste is limited to the unused part of the *last* block. |
| **External Fragmentation** | **High**: Requires huge *contiguous* holes. Even if 10GB is free total, allocation fails if memory is fragmented. | **Zero**: No need for contiguous holes. Any free block anywhere in RAM can be used.                                                 |
| **Utilization**            | Often < 60% due to reservation.                                                                                   | Near 100% due to on-demand allocation.                                                                                              |

### 4.2. Visualization: Internal Fragmentation

Imagine we have memory for 10 tokens. Request A reserves max length (5) but uses only 2.

```
[ A1 ] [ A2 ] [ -- ] [ -- ] [ -- ] [ Free ] [ Free ] [ Free ] [ Free ] [ Free ]
^----------- Reserved -----------^
```

- **Wasted**: 3 slots (`--`) are locked.
- **Blocked**: Request B needs 6 tokens. Even though 8 slots are effectively free, B cannot fit because the free space is split.

### 4.3. Visualization: External Fragmentation ("Swiss Cheese" Problem)

Imagine requests arriving and finishing over time. We have 10GB free total. Request C needs 1GB (contiguous).

**Naive Approach**:

```
[ Free 200MB ] [ Req A (Active) ] [ Free 500MB ] [ Req B (Active) ] [ Free 300MB ] ...
```

- **Result**: Request C **FAILS** (OOM). The largest contiguous chunk is only 500MB.

**Paged Attention Approach**:
Request C needs 1GB but can be split into many small blocks.

```
[ Used by C ] [ Req A (Active) ] [ Used by C ] [ Req B (Active) ] [ Used by C ] ...
```

- **Result**: Request C **SUCCEEDS**. We fill every small gap with parts of Request C.

---

## 5. The Hardware Challenge: CPU Caching & Latency

**Paged Attention** solves the memory problem by borrowing concepts from OS **Virtual Memory**. However, breaking contiguous memory introduces CPU hardware inefficiencies.

### 5.1. Memory Hierarchy Overview

- **Registers**: Fastest, immediate data.
- **L1/L2 Cache**: Fast, small (KB-MB). Heavily relies on spatial locality.
- **Main Memory (RAM)**: Slow, massive.

### 5.2. The Cost of Indirection

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
   - **Translation**: Look up Block Table -> "Physical Block B" (Extra Load).
   - **Physical Jump**: 0x1000 -> 0x9000.
   - **Result**: The CPU prefetcher cannot predict this jump, causing a stall (RAM Latency).

### 5.3. Cache Prefetching Mechanics

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

- **Hit Rate**: Accessing floats 1-15 is free (Cached).
- **Prefetching**: Hardware detects the linear pattern (`i, i+1`) and fetches the *next* line before you ask.

#### Prefetcher Failure on Block Boundaries

- **Sequential Prefetching**: Works inside a block.
- **Aggressive Prefetching**: If the prefetch distance is too large, it may cross the block boundary and fetch **garbage data** from a physically adjacent but logically unrelated block, wasting bandwidth.

### 5.4. Physical Proximity & Fragmentation

A common question: *"Aren't consecutive blocks likely to be physically close anyway?"*
**Answer: No.**

Due to memory fragmentation (Parking Lot Analogy), logical blocks are randomly scattered.

- Logical Block 0 -> Physical Slot 5
- Logical Block 1 -> Physical Slot 200 (Slots 6-199 were taken by others)

We must assume **Zero Spatial Locality** between blocks.

### 5.5. The Trade-off: Why Small Block Size is Bad

If the Cache Line is 64 bytes, why not make the Block Size 64 bytes (1 Token) to eliminate all internal fragmentation?

This would be disastrous due to **Prefetcher Stalls** and **TLB Misses**.

| Feature                 | Naive (Contiguous)                         | Paged Attention (Small Block)                                        |
| :---------------------- | :----------------------------------------- | :------------------------------------------------------------------- |
| **Prefetcher**          | **Proactive**: Latency ~ 0.                | **Disabled**: CPU cannot predict jumps. Waits for full DRAM latency. |
| **Address Calculation** | **Fast (ALU)**: `Base + Index`.            | **Slow (Memory)**: Must load `BlockTable` first (Pointer Chasing).   |
| **TLB (Translation)**   | **High Hit Rate**: 1 Huge Page covers 2MB. | **High Miss Rate**: Every small block is on a different 4KB page.    |

**Conclusion**: We need a `block_size` (e.g., 16 or 32 tokens) large enough to amortize the jump cost, but small enough to minimize internal fragmentation.

---

## 6. Workload Analysis: Prefill vs. Decoding

To fully understand the impact of Paged Attention, we must distinguish between the two phases of LLM inference, as they have different hardware bottlenecks.

### 6.1. Prefill Phase (Prompt Processing)

- **Operation**: Processes all input tokens in parallel to generate the initial KV cache.
- **Bottleneck**: **Compute Bound**. The CPU/GPU is saturated by dense Matrix Multiplications (GEMMs) for Q, K, V projections and Attention.
- **Paged Attention Impact**:
  - **Writes**: We must allocate and write the initial KV cache into non-contiguous blocks.
  - **Overhead**: The overhead of block allocation and indirect addressing is usually **negligible** (hidden) compared to the massive computation load.
  - **Benefit**: Allows processing longer prompts or larger batches without failing due to fragmentation.

### 6.2. Decoding Phase (Token Generation)

- **Operation**: Generates one token at a time, autoregressively.
- **Bottleneck**: **Memory Bandwidth Bound**. For each new token, the arithmetic intensity is low (Matrix-Vector multiplication). The speed is limited by how fast we can move the *entire* KV cache from RAM to the CPU cores.
- **Paged Attention Impact**:
  - **Reads**: This is where the "Cost of Indirection" (Section 5.2) is most visible. We are reading gigabytes of data per second.
  - **Throughput vs. Latency**:
    - *Latency*: Single-request latency might slightly degrade due to non-contiguous reads and prefetcher stalls.
    - *Throughput*: System throughput increases significantly. By eliminating fragmentation, we can fit more concurrent requests (larger **Batch Size**) into RAM.
  - **Key Insight**: Since decoding is memory-bound, **wasting memory (fragmentation) = wasting bandwidth**. Paged Attention ensures that every byte of bandwidth transfers useful data, not empty padding.

---

## 7. Current Implementation: Naive Baseline

The current implementation in `include/core/model.hpp` represents the **Naive (Contiguous)** approach. It allocates the maximum possible memory upfront, which is simple but inefficient.

### 7.1. Data Structures (`include/core/model.hpp`)

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

### 7.2. Address Calculation

Because the memory is contiguous, we use simple pointer arithmetic (linear addressing) to access the Key/Value vectors. There is no `BlockTable` or virtual-to-physical translation yet.

The implementation also handles **Grouped Query Attention (GQA)**, where `n_kv_heads < n_heads`. Multiple query heads share the same K/V head, computed as `kv_h = h / kv_mul` where `kv_mul = n_heads / n_kv_heads`. For example, Llama 2 70B uses 64 query heads with 8 KV heads (`kv_mul = 8`), reducing KV cache size by 8x compared to standard Multi-Head Attention.

```cpp
void attention(int layer, int pos, float *out)
{
    // ...
    int   kv_mul = n_heads / n_kv_heads;         // GQA multiplier
    float scale  = 1.0f / sqrtf(head_dim);       // Scaled Dot-Product Attention
    int layer_offset = layer * config.max_seq_len * n_kv_heads * head_dim;

    for (int h = 0; h < n_heads; h++) {
        // ...
        int kv_h = h / kv_mul; // Map query head to shared KV head

        // Score Calculation: Q * K^T / sqrt(d_k)
        for (int t = 0; t <= pos; t++) {
            // Linear Access: Base + (Time Step * Stride)
            float *k_head = state.key_cache.data() + layer_offset + t * n_kv_heads * head_dim + kv_h * head_dim;

            float score = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                score += q_head[i] * k_head[i];
            }
            score *= scale;
            att_head[t] = score;
        }
        // Softmax + Weighted sum: softmax(Q*K^T/sqrt(d_k)) * V
        // ...
    }
}
```

---

## 8. PagedAttention Implementation

### 8.1. BlockManager

The `BlockManager` (`include/scheduler/block_manager.hpp`) manages physical block allocation with thread-safe per-request tracking:

```cpp
class BlockManager {
    int num_blocks_;              // Total physical blocks
    int block_size_;              // Tokens per block
    std::vector<bool> free_blocks_;  // Track free blocks
    int num_free_blocks_;         // Fast free count
    mutable std::mutex mutex_;    // Thread safety

    // Per-request block tracking
    std::unordered_map<int, std::vector<int>> request_blocks_;

public:
    // Allocate a single block (returns -1 if OOM)
    int allocate_block();

    // Allocate multiple blocks for a token sequence (with rollback on failure)
    std::vector<int> allocate_sequence(int num_tokens) {
        int num_blocks_needed = (num_tokens + block_size_ - 1) / block_size_;
        // Find and allocate free blocks, rollback if insufficient...
        return allocated_block_ids;
    }

    // Thread-safe per-request allocation
    std::vector<int> allocate_for_request(int request_id, int num_tokens);
    int allocate_block_for_request(int request_id);

    // Free all blocks when request completes
    void free_request(int request_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int block_id : request_blocks_[request_id]) {
            free_block_internal(block_id);
        }
        request_blocks_.erase(request_id);
    }

    float get_utilization() const;  // 0.0 to 1.0
};
```

### 8.2. Address Translation

**Logical to Physical Translation:**

```cpp
// Given: logical token position t
int logical_block  = t / block_size;         // Which logical block
int block_offset   = t % block_size;         // Position within block
int physical_block = block_table[logical_block];  // Look up physical location

// Access KV cache at physical location
// Layout: [num_physical_blocks, block_size, n_kv_heads, head_dim]
float* k_ptr = key_cache + physical_block * block_size * n_kv_heads * head_dim
             + block_offset * n_kv_heads * head_dim + kv_h * head_dim;
```

### 8.3. Paged Attention Kernel

```cpp
void paged_attention(float* out, const float* q,
                     const float* key_cache, const float* value_cache,
                     const int* block_table, float* att_scores,
                     int num_tokens, int block_size,
                     int head_dim, int n_heads, int n_kv_heads) {

    int   kv_mul = n_heads / n_kv_heads;
    float scale  = 1.0f / sqrtf(head_dim);

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / kv_mul;

        // Score calculation: Q * K^T / sqrt(d_k) with block table lookup
        for (int t = 0; t < num_tokens; t++) {
            int logical_block  = t / block_size;
            int block_offset   = t % block_size;
            int physical_block = block_table[logical_block];

            const float* k_head = key_cache
                + physical_block * block_size * n_kv_heads * head_dim
                + block_offset * n_kv_heads * head_dim
                + kv_h * head_dim;

            float score = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                score += q_head[i] * k_head[i];
            }
            score *= scale;
            att_head[t] = score;
        }

        // Softmax...

        // Weighted sum with same block table lookup for V
        for (int t = 0; t < num_tokens; t++) {
            // Same block table lookup for value cache
        }
    }
}
```

---

## 9. Memory Savings Analysis

The `KVCacheMetrics` class (`include/utils/metrics.hpp`) tracks and compares memory usage between the two approaches:

```cpp
class KVCacheMetrics {
public:
    void set_sequence_length(int len);
    void set_blocks_used(int blocks);

    // KV Cache = n_layers × seq_tokens × n_kv_heads × head_dim × sizeof(float) × 2
    static size_t calculate_kv_cache_bytes(int n_layers, int seq_tokens,
                                           int n_kv_heads, int head_dim) {
        return n_layers * seq_tokens * n_kv_heads * head_dim * sizeof(float) * 2;
    }

    void print_comparison(int n_layers, int n_kv_heads, int head_dim,
                          int max_seq_len, int block_size) const {
        // Standard Attention: reserves full max_seq_len
        size_t standard_memory = calculate_kv_cache_bytes(n_layers, max_seq_len, ...);

        // PagedAttention: only blocks actually used
        int paged_tokens = blocks_used_ * block_size;
        size_t paged_memory = calculate_kv_cache_bytes(n_layers, paged_tokens, ...);

        // Print formatted comparison table with savings percentage
    }
};
```

**Example (Llama 7B, max_seq_len=2048, actual=256):**

| Method             | Memory Used      |
| ------------------ | ---------------- |
| Standard Attention | ~2 GB (reserved) |
| PagedAttention     | ~256 MB (actual) |
| **Savings**        | ~87.5%           |

---

## 10. Summary & Next Steps

The current codebase implements the **Naive Approach** described in Section 4.

1. **Status**: Functional but memory-inefficient.
2. **Fragmentation**: As shown in Section 4.2, this implementation "locks" `max_seq_len` worth of memory for every request, even if only a few tokens are generated.
3. **Next Goal**: Refactor `RunState` and `attention` logic to use **Paged Attention**, moving from contiguous `std::vector` to a block-based memory pool and page table lookup, as theoretically described in Sections 5-8.

**Next: [Part 2](./Part2_ContinuousBatching_ChunkedPrefill.md)** will cover how to efficiently handle multiple concurrent requests with Continuous Batching and Chunked Prefill.

---

## References

[^1]: Andrej Karpathy. nanoGPT. <https://github.com/karpathy/nanoGPT>

[^2]: Andrej Karpathy. llm.c. <https://github.com/karpathy/llm.c>

[^3]: Zhao et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. arXiv:2309.06180. <https://arxiv.org/abs/2309.06180>

[^4]: vLLM. <https://github.com/vllm-project/vllm>

[^5]: SGLang. <https://github.com/sgl-project/sglang>

[^6]: nano-vLLM (existing implementation). <https://github.com/GeeeekExplorer/nano-vllm>

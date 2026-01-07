# Paged Attention: CPU Implementation & Hardware Utilization

## 1. Introduction
In Large Language Model (LLM) serving, **KV Cache** management is a critical bottleneck. The traditional "Naive" approach allocates a large contiguous chunk of memory for the maximum possible sequence length (e.g., 2048 tokens) for every request. This leads to:
- **Internal Fragmentation**: A request that only uses 100 tokens still hogs memory for 2048 tokens.
- **Memory Waste**: We cannot "reclaim" the unused tail of the buffer for other requests.

**Paged Attention** solves this by borrowing concepts from Operating System (OS) **Virtual Memory**. Instead of contiguous allocation, we break the KV Cache into fixed-size **Blocks** (pages). These blocks can be stored anywhere in physical memory (RAM), allowing us to dynamically allocate memory only as needed.

## 2. CPU Hardware Context
To understand why this implementation is efficient (and how it prepares us for GPU), we must look at the CPU memory hierarchy.

### 2.1. Memory Hierarchy
*   **Registers**: Fastest, immediate data (variables like `score`, `i`).
*   **L1/L2 Cache**: Fast, small (KB to MB). Stores frequently accessed data.
*   **L3 Cache**: Slower, larger (MBs). Shared across cores.
*   **Main Memory (RAM)**: Slow, massive (GBs). Stores our "Physical KV Block Pool".

### 2.2. The Cost of Indirection
In the Naive approach, accessing `K[t]` and `K[t+1]` is just `ptr++`. This is extremely cache-friendly because the CPU prefetcher can pull the next cache line easily.

In **Paged Attention**, `K[t]` and `K[t+1]` might be in completely different physical blocks if `t` crosses a block boundary. This introduces **Indirection**:
1.  **Logical Address**: "Token 500 in Sequence 1".
2.  **Translation**: Look up Block Table -> "Physical Block 42".
3.  **Physical Address**: Access RAM at `Pool_Base + Offset`.

**Hardware Trade-off**: We accept slightly higher compute overhead (calculating addresses) and potential cache misses (jumping blocks) in exchange for **massive memory savings** and higher **throughput** (batch size).

## 3. Implementation Details (Nano-vLLM)

Our implementation is located in `src/memory.c` and `kernels/cpu/kernels.c`.

### 3.1. Data Structures (`include/structs.h`)

#### A. The Physical Memory Pool (`KVCacheManager`)
We pre-allocate one massive array in RAM that represents all available physical memory for the KV cache. This avoids the overhead of calling `malloc` during generation (which is slow and causes OS-level fragmentation).

```c
typedef struct {
    // ...
    // [n_layers, num_blocks, block_size, n_kv_heads, head_dim]
    float* pool_k; 
    float* pool_v;
    int* free_block_indices; // Stack to track available blocks
} KVCacheManager;
```

*   **Hardware View**: This is a single contiguous region in Virtual Address Space (User Space), mapped to Physical RAM. It maximizes TLB (Translation Lookaside Buffer) hits compared to thousands of tiny `malloc`s.

**Memory Pool Visualization**

The `pool_k` (and `pool_v`) is a flattened 1D array representing a 5D tensor.

```
[ Physical RAM (One Contiguous Array) ]
--------------------------------------------------------------------------------------
| Block 0      | Block 1      | ... | Block 7      | ... | Block N      |
--------------------------------------------------------------------------------------
|  (Free)      |  (Seq B)     |     |  (Seq A)     |     |  (Free)      |
--------------------------------------------------------------------------------------
^              ^                    ^
|              |                    |
Base Ptr       Ptr + Stride         Ptr + 7*Stride
```

**Inside a Single Block (Zoom In)**
A block contains `block_size` tokens. Data is packed tightly for CPU cache efficiency.

```
[ Block 7 Content (allocated to Seq A) ]
-------------------------------------------------------------
| Token T  | Token T+1 | ... | Token T+15 |
-------------------------------------------------------------
     |
     | (Zoom In: Token Structure)
     v
     -------------------------------------------------------
     | Head 0 | Head 1 | ... | Head K (n_kv_heads)         |
     -------------------------------------------------------
         |
         | (Zoom In: Head Vector)
         v
         [ float, float, ... (head_dim elements) ] 
         -> Contiguous in memory (SIMD Friendly)
```

#### B. The Page Table (`BlockTable`)
Each sequence has a private "Page Table" that maps its logical timeline to physical blocks.

```c
typedef struct {
    int* block_indices; // Maps logical_block_idx -> physical_block_idx
    int num_blocks;
} BlockTable;
```

### 3.2. The Address Translation Logic
This is the heart of Paged Attention. It runs on the CPU for every token generation step.

**File**: `kernels/cpu/kernels.c`

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
        long offset = get_physical_offset(mgr, layer, physical_block, block_offset, ...);
        float* k_ptr = mgr->pool_k + offset;
        
        // 4. Compute (SIMD Optimized)
        // The inner loop is contiguous (head_dim floats), friendly to AVX/NEON.
        float score = dot_product(q, k_ptr, head_dim); 
    }
}
```

### 3.3. Access Patterns & Optimization
1.  **Contiguous Inner Loop**: Inside a single token's head (`head_dim = 64 floats`), data is contiguous. This allows the CPU to use **SIMD instructions** (Single Instruction, Multiple Data) efficiently.
2.  **Block Locality**: While we jump between blocks, we stay within a block for `block_size` (e.g., 16) tokens. This amortizes the cost of the table lookup.

## 4. Why This Implementation Matters
1.  **Zero-Copy Reorganization**: If we want to fork a sequence (e.g., Beam Search), we don't copy the KV cache. We just copy the `BlockTable` (integers). Both sequences point to the same physical blocks.
2.  **Throughput**: By saving memory, we can increase `BATCH_SIZE`. Instead of running 1 request, we can run 4 or 8 on the same CPU.
3.  **GPU Preparation**: GPUs have High Bandwidth Memory (HBM) which is very fast but limited in capacity. The exact same logic applies: we will move the `pool_k/v` to GPU VRAM and the `BlockTable` lookup will happen inside the CUDA kernel.

## 5. Summary
| Feature | Naive Implementation | Paged Attention (CPU) |
| :--- | :--- | :--- |
| **Memory Layout** | Contiguous (Virtual & Physical) | Non-contiguous (Physical) |
| **Allocation** | Up-front (Max Seq Len) | On-demand (Per Block) |
| **Access Cost** | Low (Pointer Arithmetic) | Medium (Table Lookup + Stride) |
| **Memory Efficiency** | Low (Fragmentation) | High (Near Optimal) |
| **Hardware Usage** | Simple Prefetching | Requires Cache Awareness |

This implementation demonstrates how software-defined memory management can overcome hardware limitations, a technique essential for modern AI systems.


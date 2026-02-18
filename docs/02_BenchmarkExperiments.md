# Benchmark Experiments: Continuous Batching & Chunked Prefill on CPU

This document presents benchmark results comparing different inference configurations
on CPU using the stories15M model (dim=288, 6 layers, 6 heads, vocab=32000).

## Experiment Setup

- **Model**: stories15M (60MB, Llama2 architecture)
- **Platform**: macOS arm64 (Apple Silicon)
- **Workload**: 4 requests with short prompts (3-14 tokens each), generating 20-50 tokens
- **Workload file**: `examples/comparison_workload.json`

### Configurations

| # | Mode | Paged Attention | Batch Size | Chunked Prefill |
|---|------|-----------------|------------|-----------------|
| 1 | Sequential | OFF | 1 | N/A |
| 2 | Sequential | ON | 1 | N/A |
| 3 | Batched | ON | 4 | OFF (`--max-tokens-per-batch 65536`) |
| 4 | Batched | ON | 4 | ON (`--max-tokens-per-batch 8`) |

### Commands

```bash
# 1. Sequential + Standard Attention
./build/main models --input-json examples/comparison_workload.json \
    --without-paged-attn --save-results results/1_seq_std.json

# 2. Sequential + Paged Attention
./build/main models --input-json examples/comparison_workload.json \
    --save-results results/2_seq_paged.json

# 3. Batched + Paged Attention + No Chunking
./build/main models --input-json examples/comparison_workload.json \
    -b 4 --max-tokens-per-batch 65536 --save-results results/3_batch_paged_nochunk.json

# 4. Batched + Paged Attention + Chunked Prefill (8 tokens)
./build/main models --input-json examples/comparison_workload.json \
    -b 4 --max-tokens-per-batch 8 --save-results results/4_batch_paged_chunk8.json
```

## Results

| # | Configuration | Total Time | Prefill tok/s | Decode tok/s | Overall tok/s | KV Memory |
|---|---------------|------------|---------------|--------------|---------------|-----------|
| 1 | Sequential + StdAttn | 437.30 ms | 873.77 | 352.95 | 400.18 | 3.38 MB |
| 2 | Sequential + PagedAttn | 433.93 ms | 898.42 | 355.00 | 403.29 | 2.74 MB |
| 3 | Batched(4) + PagedAttn + No Chunk | 453.68 ms | 776.61 | 343.82 | 385.73 | 2.74 MB |
| 4 | Batched(4) + PagedAttn + Chunk(8) | 483.87 ms | 737.57 | 349.42 | 361.67 | 2.74 MB |

### Comparison Tables

#### Standard vs Paged Attention (Run 1 vs 2)

```
+--------------------------+--------------------+--------------------+----------+
| Metric                   | sequential + StdAt | sequential + Paged | Diff     |
+--------------------------+--------------------+--------------------+----------+
| Total Time               | 437.30 ms          | 433.93 ms          | -0.8%    |
| Prefill Time             | 40.06 ms           | 38.96 ms           | -2.7%    |
| Decode Time              | 396.66 ms          | 394.36 ms          | -0.6%    |
+--------------------------+--------------------+--------------------+----------+
| Prefill Throughput       | 873.69 tok/s       | 898.36 tok/s       | +2.8%    |
| Decode Throughput        | 352.95 tok/s       | 355.01 tok/s       | +0.6%    |
| Overall Throughput       | 400.18 tok/s       | 403.29 tok/s       | +0.8%    |
+--------------------------+--------------------+--------------------+----------+
| KV Cache Memory          | 3.38 MB            | 2.74 MB            | -18.8%   |
+--------------------------+--------------------+--------------------+----------+
```

#### Sequential vs Continuous Batching (Run 2 vs 3)

```
+--------------------------+--------------------+--------------------+----------+
| Metric                   | sequential + Paged | batched + PagedAtt | Diff     |
+--------------------------+--------------------+--------------------+----------+
| Total Time               | 433.93 ms          | 453.68 ms          | +4.6%    |
| Prefill Time             | 38.96 ms           | 45.07 ms           | +15.7%   |
| Decode Time              | 394.36 ms          | 407.19 ms          | +3.3%    |
+--------------------------+--------------------+--------------------+----------+
| Prefill Throughput       | 898.36 tok/s       | 776.57 tok/s       | -13.6%   |
| Decode Throughput        | 355.01 tok/s       | 343.82 tok/s       | -3.2%    |
| Overall Throughput       | 403.29 tok/s       | 385.73 tok/s       | -4.4%    |
+--------------------------+--------------------+--------------------+----------+
| KV Cache Memory          | 2.74 MB            | 2.74 MB            | 0.0%     |
+--------------------------+--------------------+--------------------+----------+
```

#### Chunked Prefill OFF vs ON (Run 3 vs 4)

```
+--------------------------+--------------------+--------------------+----------+
| Metric                   | No Chunk (65536)   | Chunk (8)          | Diff     |
+--------------------------+--------------------+--------------------+----------+
| Total Time               | 453.68 ms          | 483.87 ms          | +6.7%    |
| Prefill Time             | 45.07 ms           | 47.45 ms           | +5.3%    |
| Decode Time              | 407.19 ms          | 400.66 ms          | -1.6%    |
+--------------------------+--------------------+--------------------+----------+
| Prefill Throughput       | 776.57 tok/s       | 737.62 tok/s       | -5.0%    |
| Decode Throughput        | 343.82 tok/s       | 349.42 tok/s       | +1.6%    |
| Overall Throughput       | 385.73 tok/s       | 361.67 tok/s       | -6.2%    |
+--------------------------+--------------------+--------------------+----------+
| KV Cache Memory          | 2.74 MB            | 2.74 MB            | 0.0%     |
+--------------------------+--------------------+--------------------+----------+
```

## Analysis

### 1. Standard vs Paged Attention

- **Throughput**: Nearly identical (~0.8% faster with paged attention).
  The block indirection overhead is negligible for this small model.
- **Memory**: Paged attention uses **18.8% less KV cache** (2.74 MB vs 3.38 MB).
  Standard attention pre-allocates the full `max_seq_len` (1024 tokens),
  while paged attention only allocates blocks actually used
  (13 blocks = 208 tokens worth).
- **Takeaway**: Paged attention is a clear win on CPU -- same speed, less memory.

### 2. Sequential vs Continuous Batching

- **Throughput**: Batched mode is **4.4% slower** overall.
  Prefill is 13.6% slower due to scheduler overhead
  (batch formation, block allocation for 4 concurrent requests).
- **Why slower on CPU?** Requests execute serially within a batch --
  there is no parallel matrix multiplication. The scheduler adds
  overhead without a throughput benefit.
- **Takeaway**: On CPU, continuous batching adds overhead without throughput gain.
  Its value is **scheduling fairness** (shorter requests finish earlier
  when interleaved with long ones), not raw speed.

### 3. Chunked Prefill ON vs OFF

- **Throughput**: Chunked prefill is **6.2% slower** overall.
  More scheduler iterations (125 vs 51) means more overhead per token.
- **Scheduling behavior**: With `chunk=8`, requests are admitted in waves --
  Request 0 was fully decoded before Requests 1-3 even started prefill.
  Without chunking, all 4 prefill together then all decode together.
- **Decode throughput**: Slightly better with chunking (+1.6%)
  because shorter requests finish and free up memory/blocks sooner.
- **Takeaway**: Chunked prefill trades total throughput for **latency fairness**.
  On CPU, the overhead is measurable but the benefit (preventing head-of-line blocking
  for short requests) only matters with mixed-length workloads.

## CPU vs GPU: Why Results Differ

On **GPU**, continuous batching and chunked prefill provide significant benefits because:

1. Batched requests share **parallel matrix operations** (GEMM) --
   more requests per batch = better GPU utilization
2. Chunked prefill prevents a single long prompt from monopolizing the GPU
   while short decode steps starve

On **CPU**, every request calls `model.forward()` **sequentially** --
the "batch" is just a scheduling abstraction with no compute parallelism,
so the overhead of scheduling is pure cost.

### Summary Table

| Feature | CPU Impact | GPU Impact |
|---------|-----------|------------|
| Paged Attention | Same speed, **-18.8% memory** | Same speed, significant memory savings at scale |
| Continuous Batching | **-4.4% throughput** (overhead) | Major throughput gain (parallel GEMM) |
| Chunked Prefill | **-6.2% throughput** (overhead) | Better latency fairness + GPU utilization |

The primary value of this CPU implementation is **educational** --
it demonstrates the algorithms and scheduling policies of vLLM
in a readable, single-threaded environment.

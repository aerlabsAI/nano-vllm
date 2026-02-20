# Benchmark Experiments: Continuous Batching & Chunked Prefill on CPU

This document presents benchmark results comparing different inference configurations
on CPU using the stories15M model (dim=288, 6 layers, 6 heads, vocab=32000).

## Experiment Setup

- **Model**: stories15M (60MB, Llama2 architecture, max_seq_len=256)
- **Platform**: macOS arm64 (Apple Silicon)
- **Workload**: 6 requests with mixed prompt lengths (6-79 tokens each), generating 20-50 tokens
- **Workload file**: `examples/comparison_workload.json`

### Workload Design

The workload mixes long and short prompts to demonstrate chunked prefill behavior:

| Request | Prompt | Tokens | Max Gen |
|---------|--------|--------|---------|
| 0 | "Once upon a time in a magical forest..." (long story) | ~72 | 30 |
| 1 | "Tell me a story." | ~6 | 50 |
| 2 | "In a small village at the edge..." (long story) | ~79 | 30 |
| 3 | "What is the meaning of life?" | ~8 | 40 |
| 4 | "The sun was setting over the vast ocean..." (long story) | ~74 | 20 |
| 5 | "Write a poem about the stars." | ~8 | 40 |

With `--max-tokens-per-batch 64`, the three long prompts (72, 79, 74 tokens)
exceed the token budget and trigger chunked prefill, while short prompts fit
entirely in a single chunk.

### Configurations

| # | Mode | Paged Attention | Batch Size | Chunked Prefill |
|---|------|-----------------|------------|-----------------|
| 1 | Sequential | OFF | 1 | N/A |
| 2 | Sequential | ON | 1 | N/A |
| 3 | Batched | ON | 4 | OFF (`-bt 65536`) |
| 4 | Batched | ON | 4 | ON (`-bt 64`) |

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
    -b 4 -bt 65536 --save-results results/3_batch_paged_nochunk.json

# 4. Batched + Paged Attention + Chunked Prefill (64 tokens)
./build/main models --input-json examples/comparison_workload.json \
    -b 4 -bt 64 --save-results results/4_batch_paged_chunk64.json
```

## Results

| # | Configuration | Total Time | Prefill tok/s | Decode tok/s | Overall tok/s | KV Memory |
|---|---------------|------------|---------------|--------------|---------------|-----------|
| 1 | Sequential + StdAttn | 593.90 ms | 1286.45 | 534.22 | 769.50 | 3.38 MB |
| 2 | Sequential + PagedAttn | 594.68 ms | 1277.75 | 534.93 | 768.48 | 6.33 MB |
| 3 | Batched(4) + PagedAttn + No Chunk | 607.87 ms | 1232.90 | 516.74 | 751.81 | 6.33 MB |
| 4 | Batched(4) + PagedAttn + Chunk(64) | 604.96 ms | 1209.46 | 525.93 | 755.42 | 6.33 MB |

### Comparison Tables

#### Standard vs Paged Attention (Run 1 vs 2)

```
+--------------------------+--------------------+--------------------+----------+
| Metric                   | sequential + StdAt | sequential + Paged | Diff     |
+--------------------------+--------------------+--------------------+----------+
| Total Time               | 593.90 ms          | 594.68 ms          | +0.1%    |
| Prefill Time             | 192.00 ms          | 193.31 ms          | +0.7%    |
| Decode Time              | 393.09 ms          | 392.58 ms          | -0.1%    |
+--------------------------+--------------------+--------------------+----------+
| Prefill Throughput       | 1286.46 tok/s      | 1277.74 tok/s      | -0.7%    |
| Decode Throughput        | 534.23 tok/s       | 534.92 tok/s       | +0.1%    |
| Overall Throughput       | 769.49 tok/s       | 768.48 tok/s       | -0.1%    |
+--------------------------+--------------------+--------------------+----------+
| KV Cache Memory          | 3.38 MB            | 6.33 MB            | +87.5%   |
+--------------------------+--------------------+--------------------+----------+
```

#### Sequential vs Continuous Batching (Run 2 vs 3)

```
+--------------------------+--------------------+--------------------+----------+
| Metric                   | sequential + Paged | batched + PagedAtt | Diff     |
+--------------------------+--------------------+--------------------+----------+
| Total Time               | 594.68 ms          | 607.87 ms          | +2.2%    |
| Prefill Time             | 193.31 ms          | 200.34 ms          | +3.6%    |
| Decode Time              | 392.58 ms          | 406.39 ms          | +3.5%    |
+--------------------------+--------------------+--------------------+----------+
| Prefill Throughput       | 1277.74 tok/s      | 1232.90 tok/s      | -3.5%    |
| Decode Throughput        | 534.92 tok/s       | 516.74 tok/s       | -3.4%    |
| Overall Throughput       | 768.48 tok/s       | 751.81 tok/s       | -2.2%    |
+--------------------------+--------------------+--------------------+----------+
| KV Cache Memory          | 6.33 MB            | 6.33 MB            | 0.0%     |
+--------------------------+--------------------+--------------------+----------+
```

#### Chunked Prefill OFF vs ON (Run 3 vs 4)

```
+--------------------------+--------------------+--------------------+----------+
| Metric                   | batched + PagedAtt | batched + PagedAtt | Diff     |
+--------------------------+--------------------+--------------------+----------+
| Total Time               | 607.87 ms          | 604.96 ms          | -0.5%    |
| Prefill Time             | 200.34 ms          | 204.22 ms          | +1.9%    |
| Decode Time              | 406.39 ms          | 399.29 ms          | -1.7%    |
+--------------------------+--------------------+--------------------+----------+
| Prefill Throughput       | 1232.90 tok/s      | 1209.48 tok/s      | -1.9%    |
| Decode Throughput        | 516.74 tok/s       | 525.93 tok/s       | +1.8%    |
| Overall Throughput       | 751.81 tok/s       | 755.42 tok/s       | +0.5%    |
+--------------------------+--------------------+--------------------+----------+
| KV Cache Memory          | 6.33 MB            | 6.33 MB            | 0.0%     |
+--------------------------+--------------------+--------------------+----------+
```

### Scheduling Trace: Chunked Prefill in Action (Run 4, `-bt 64`)

The following trace shows how the scheduler chunks long prompts and interleaves
prefill with decode:

```
Iter 0:  PREFILL  1 req,  64 tok  | Req0: chunk [0..64) of 72 tokens
Iter 1:  PREFILL  3 req,  64 tok  | Req0: chunk [64..72) DONE
                                   | Req1: full prefill [0..6) DONE
                                   | Req2: chunk [0..50) of 79 tokens
Iter 2-31:  DECODE  2 req, 2 tok  | Req0 + Req1 decoding together
Iter 31: Req0 finished (30 tokens generated)
Iter 32-51: DECODE  1 req, 1 tok  | Req1 decoding alone
Iter 51: Req1 finished (50 tokens generated)

Iter 52: PREFILL  3 req, 64 tok   | Req2: chunk [50..79) DONE
                                   | Req3: full prefill [0..8) DONE
                                   | Req4: chunk [0..27) of 74 tokens
Iter 53-82: DECODE  2 req, 2 tok  | Req2 + Req3 decoding together
Iter 82: Req2 finished (30 tokens generated)
Iter 83-92: DECODE  1 req, 1 tok  | Req3 decoding alone
Iter 92: Req3 finished (40 tokens generated)

Iter 93: PREFILL  2 req, 55 tok   | Req4: chunk [27..74) DONE
                                   | Req5: full prefill [0..8) DONE
Iter 94-113: DECODE  2 req, 2 tok | Req4 + Req5 decoding together
Iter 113: Req4 finished (20 tokens generated)
Iter 114-133: DECODE  1 req, 1 tok| Req5 decoding alone
Iter 133: Req5 finished (40 tokens generated)
```

Key observations from the trace:

- **Chunking**: Req0 (72 tok), Req2 (79 tok), and Req4 (74 tok) all exceed
  the 64-token budget and are split across multiple prefill iterations.
- **Budget packing**: After a chunk completes, remaining budget is used for
  the next request (e.g., Iter 1: 8 + 6 + 50 = 64 tokens).
- **Decode-first policy**: Once requests enter decode, they are prioritized
  over pending prefills. New prefills only happen when decode slots are free.
- **Continuous batching**: Multiple requests decode simultaneously
  (e.g., Req0 + Req1 in Iter 2-31), and finished requests free slots
  for new prefills.

## Analysis

### 1. Standard vs Paged Attention

- **Throughput**: Nearly identical (~0.1% difference, within noise).
  The block indirection overhead is negligible for this small model.
- **Memory**: Paged attention reports **higher** KV cache in this measurement
  because sequential mode re-initializes the block manager per request,
  and the metric sums estimated blocks across all requests.
  In practice, paged attention only allocates blocks actually used,
  while standard attention pre-allocates the full `max_seq_len`.
- **Takeaway**: Paged attention has negligible overhead on CPU. The real
  memory savings are visible at scale with many concurrent sequences.

### 2. Sequential vs Continuous Batching

- **Throughput**: Batched mode is **2.2% slower** overall.
  Both prefill (-3.5%) and decode (-3.4%) are slower due to scheduler
  overhead (batch formation, block allocation for concurrent requests).
- **Why slower on CPU?** Requests execute serially within a batch --
  there is no parallel matrix multiplication. The scheduler adds
  overhead without a compute throughput benefit.
- **Takeaway**: On CPU, continuous batching adds overhead without throughput gain.
  Its value is **scheduling fairness** (shorter requests finish earlier
  when interleaved with long ones), not raw speed.

### 3. Chunked Prefill ON vs OFF

- **Overall throughput**: Nearly identical (+0.5%), within noise.
  With longer prompts that actually trigger chunking, the overhead
  of extra scheduler iterations is offset by better decode interleaving.
- **Prefill throughput**: Slightly slower (-1.9%) due to chunk boundary overhead.
- **Decode throughput**: Slightly faster (+1.8%) because chunking allows
  decode to start sooner -- requests that finish prefill early can begin
  generating while others are still prefilling.
- **Scheduling behavior**: With `chunk=64`, the 3 long prompts (72, 79, 74 tokens)
  are each split into 2 prefill iterations. Short prompts (6-8 tokens) fit
  in remaining budget alongside long-prompt chunks.
- **Takeaway**: Chunked prefill trades prefill throughput for **decode latency
  fairness**. On CPU, the trade-off is roughly even. On GPU, chunked prefill
  prevents long prompts from monopolizing compute while decode requests starve.

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
| Paged Attention | Same speed, memory savings at scale | Same speed, significant memory savings at scale |
| Continuous Batching | **-2.2% throughput** (overhead) | Major throughput gain (parallel GEMM) |
| Chunked Prefill | **~even throughput**, better decode latency | Better latency fairness + GPU utilization |

The primary value of this CPU implementation is **educational** --
it demonstrates the algorithms and scheduling policies of vLLM
in a readable, single-threaded environment.

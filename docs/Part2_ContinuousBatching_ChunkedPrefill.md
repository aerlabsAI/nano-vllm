# Building an LLM Inference Engine from Scratch (Part 2)

> **How Continuous Batching and Chunked Prefill Maximize Throughput**

---

## 1. Introduction

[Part 1](./Part1_CoreArchitecture_PagedAttention.md) covered PagedAttention: block-based KV cache management that eliminates memory fragmentation. But efficient memory for individual requests is only half the story — real serving must handle **multiple concurrent requests**. This part covers:

1. **Continuous Batching**: Schedule at iteration-level, not request-level
2. **Chunked Prefill**: Prevent long prompts from blocking decode

---

## 2. The Problem: Request-Level Batching

In traditional batching, we group requests and process them as a unit. But requests have different lengths:

```
Request A: 10 prompt tokens → 50 generated tokens
Request B: 100 prompt tokens → 20 generated tokens
Request C: 5 prompt tokens → 200 generated tokens

Static Batch: Process all → Generate until ALL finish → Return all results
```

**Problems:**

1. **Head-of-line blocking**: Fast requests (B) wait for slow requests (C)
2. **Underutilization**: After A and B finish, their compute slots sit idle
3. **Latency**: New requests wait for entire batch to complete

```
Iteration  1   10   20   30   40   50   ...  200
           ├────┼────┼────┼────┼────┼────...──┤

Request A: [████████████████████████]            Finishes at 60
Request B: [███████████████]                     Finishes at 120
Request C: [████████████████████████████████████████████████████]  Finishes at 205

Wasted:    [                        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  A's slot idle
           [               ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  B's slot idle
```

---

## 3. Continuous Batching

### 3.1. Core Idea

**Iteration-level scheduling**: Instead of waiting for an entire batch to complete, evaluate which requests to process at **each iteration**. Requests can **join** and **leave** the batch at any iteration.

### 3.2. The Effect

```
Request-Level Batching:
                           ↓ Batch 1 must finish before Batch 2 starts
Batch 1: [A████████] [B██████████████] [C████]     ← Wait for longest (B)
Batch 2:                                [D████████] [E██████]  ← D, E wait
         ├──────── Batch 1 (wasted slots) ────────├── Batch 2 ──┤

Continuous Batching:
Slot 1:  [A████████] [D████████████]                ← D joins when A finishes
Slot 2:  [B██████████████] [E██████]                ← E joins when B finishes
Slot 3:  [C████] [F████████] [G████████████]        ← Slot reused multiple times
         ├──────────── all slots stay full ──────────────┤
```

No wasted slots — when a request finishes, the next pending request takes its place immediately.

### 3.2.1. Experiment Summary

**GPU** (Qwen3-0.6B, RTX 3060, batch_size=8, 20 mixed-length requests):

| Mode | Output tok/s | Wall Time | TTFT p50 | Wasted Decode Slots |
|------|-------------|-----------|----------|---------------------|
| Request-Level | 3.6 | 253.9 s | 170.4 s | 1,528 |
| Continuous | 16.7 | 54.0 s | 10.8 s | 0 |

**4.7x throughput**, **15.7x TTFT improvement**, zero wasted slots. Batched requests share parallel matrix operations (fused GEMM), so more requests per batch means better hardware utilization.

**CPU** (stories15M, 6 mixed-length requests): Continuous batching is **2.2% slower** (768 → 752 tok/s) — the "batch" is a scheduling abstraction with no compute parallelism, so the scheduler adds pure overhead. Despite this, it still provides **scheduling fairness**: shorter requests finish earlier instead of waiting for the entire batch. See [Appendix A](#appendix-a-benchmark-results) for full results.

### 3.3. Request States

```cpp
enum class RequestStatus {
    PENDING,     // In queue, waiting to be scheduled
    PREFILLING,  // Processing prompt tokens
    DECODING,    // Generating tokens
    FINISHED,    // Completed
    FAILED       // Error occurred
};
```

```
PENDING → PREFILLING → DECODING → FINISHED
   │           │            │
   │           └────────────┴─→ FAILED (on error)
   └─→ Waiting in queue until scheduler picks it up
```

### 3.4. Scheduler Design

The scheduler builds a `ScheduledBatch` each iteration — a list of prefill and decode requests that fit within the token budget.

**Key design choice**: Decode requests get priority over prefill because decode costs 1 token per request while prefill costs many. This keeps in-progress requests moving and minimizes time-to-completion.

```cpp
ScheduledBatch schedule() {
    ScheduledBatch batch;

    // Priority 1: Decode requests (already in progress, 1 token each)
    for (auto* req : running_requests_) {
        if (req->status == RequestStatus::DECODING) {
            if (batch.total_requests() >= config_.max_batch_size) break;
            batch.decode_requests.push_back(req);
        }
    }

    // Priority 2: Prefill requests (new from queue, many tokens each)
    int remaining_slots = config_.max_batch_size - batch.total_requests();
    int current_tokens = batch.total_prefill_tokens() + batch.total_decode_tokens();

    while (!pending_queue_.empty() && remaining_slots > 0) {
        Request* req = pending_queue_.front();
        int req_tokens = req->num_prompt_tokens();

        if (current_tokens + req_tokens > config_.max_tokens_per_batch) break;

        pending_queue_.pop();
        req->status = RequestStatus::PREFILLING;
        running_requests_.push_back(req);
        batch.prefill_requests.push_back(req);

        current_tokens += req_tokens;
        remaining_slots--;
    }

    return batch;
}
```

---

## 4. Chunked Prefill

### 4.1. The Problem: Prefill Blocks Decode

Prefill is compute-intensive — a 2048-token prompt monopolizes the forward pass, stalling all decode requests:

```
Prefill:  [████████████████████████████████████████████████████] 2048 tokens
Decode:   [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] BLOCKED!

Users streaming tokens from decode requests see their output stall.
```

### 4.2. The Solution: Chunk the Prompt

Split prefill into smaller pieces and **interleave** with decode:

```
Without Chunking:
  Prefill: [████████████████████████████████████████] 2048 tokens at once
  Decode:  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] blocked entire time

With Chunking (chunk_size=256):
  Iter 1:  [████] 256 prefill + [●●●] decode tokens
  Iter 2:  [████] 256 prefill + [●●●] decode tokens  ← decode continues!
  Iter 3:  [████] 256 prefill + [●●●] decode tokens
  ...
  Iter 8:  [████] 256 prefill + [●●●] decode tokens  ← prefill complete
           └────┘               └───┘
           prefill chunk        decode (not blocked!)
```

Decode requests produce tokens every iteration instead of stalling for the entire prefill.

### 4.2.1. Experiment Summary

**GPU** (Qwen3-0.6B, RTX 3060, vLLM, 20 prompts, ~4096 input tokens, max_num_seqs=16):

| Config | Output tok/s | Mean TTFT | Mean TPOT |
|--------|-------------|-----------|-----------|
| No chunk | 278.3 | 1,048 ms | 32.6 ms |
| Chunk 512 | 277.9 | 1,288 ms | 31.8 ms |
| Chunk 1024 | 282.2 | 1,128 ms | 31.0 ms |
| Chunk 2048 | 285.9 | 959 ms | 30.6 ms |

Chunked prefill maintains throughput while improving **TPOT consistency** — decode requests are no longer blocked by long prefills. Larger chunks favor throughput; smaller chunks bound worst-case decode latency.

**CPU** (stories15M, 6 mixed-length requests): Chunked prefill is **roughly throughput-neutral** (752 → 755 tok/s) while improving **decode throughput by +1.8%** (517 → 526 tok/s). The slight prefill slowdown (-1.9%) comes from chunk boundary overhead. See [Appendix A](#appendix-a-benchmark-results) for full results.

### 4.3. Implementation

Chunked prefill requires three changes:

#### Request: Track prefill progress (`include/scheduler/request.hpp`)

```cpp
struct Request {
    int prefill_cursor = 0;  // How many prompt tokens processed so far

    bool is_prefill() const { return prefill_cursor < num_prompt_tokens(); }
    int  remaining_prompt() const { return num_prompt_tokens() - prefill_cursor; }
};
```

#### Scheduler: Allocate chunk sizes from token budget (`include/scheduler/scheduler.hpp`)

```cpp
// Continue prefill for requests already running (chunked)
for (auto *req : running_requests_) {
    if (req->status != RequestStatus::PREFILLING) continue;

    int remaining   = req->remaining_prompt();
    int budget_left = config_.max_tokens_per_batch - batch.total_scheduled_tokens;
    int chunk_size  = std::min(remaining, budget_left);

    if (chunk_size <= 0) break;
    batch.add(req, chunk_size);
}

// Admit new prefill requests from pending queue
while (!pending_queue_.empty()) {
    Request *req    = pending_queue_.front();
    int budget_left = config_.max_tokens_per_batch - batch.total_scheduled_tokens;
    int chunk_size  = std::min(req->remaining_prompt(), budget_left);

    if (chunk_size <= 0) break;

    pending_queue_.pop();
    req->status = RequestStatus::PREFILLING;
    running_requests_.push_back(req);
    batch.add(req, chunk_size);
}
```

#### Runner: Process only the scheduled chunk (`include/scheduler/batched_runner.hpp`)

```cpp
void run_prefill_batch(ScheduledBatch &batch, Scheduler &scheduler) {
    for (size_t i = 0; i < batch.requests.size(); i++) {
        Request *req          = batch.requests[i];
        int      tokens_to_do = batch.scheduled_tokens[i];  // Chunk size

        for (int t = 0; t < tokens_to_do; t++) {
            int token_idx = req->prefill_cursor + t;
            model_.forward_with_request(req->prompt_tokens[token_idx], req->current_pos, req);
            req->current_pos++;
        }
        req->prefill_cursor += tokens_to_do;

        if (!req->is_prefill()) {
            req->status = RequestStatus::DECODING;  // Entire prompt processed
        }
    }
}
```

### 4.4. Chunk Size Trade-off

| Chunk Size           | Prefill Efficiency          | Decode Latency              |
| -------------------- | --------------------------- | --------------------------- |
| Very Small (16)      | Poor: overhead dominates    | Excellent: minimal blocking |
| Very Large (2048)    | Excellent: like no chunking | Poor: blocks decode         |
| Sweet Spot (256-512) | Good                        | Good                        |

---

## 5. Putting It All Together

### 5.1. Complete Request Flow

```
1. Request arrives → Add to pending queue
                          ↓
2. Scheduler.schedule() → Build ScheduledBatch (decode first, then prefill)
                          ↓
3. Prefill Phase:
   - Allocate KV cache blocks (BlockManager)
   - Process prompt tokens (possibly chunked)
   - Store K, V in allocated blocks
   - Transition to DECODING when complete
                          ↓
4. Decode Phase (per iteration):
   - Forward pass for 1 token
   - Sample next token
   - Allocate new block if needed
   - Check termination (EOS, max_tokens)
                          ↓
5. Completion:
   - Free KV cache blocks
   - Remove from running list → slot available for next request
```

### 5.2. Component Interaction

```
┌─────────────────────────────────────────────────────────────────┐
│                        BatchedRunner                            │
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌────────────────┐  │
│  │   Scheduler  │────▶│ScheduledBatch│────▶│RequestProcessor│  │
│  │              │     │              │     │                │  │
│  │ - pending    │     │ - prefill[]  │     │ - forward()    │  │
│  │ - running    │     │ - decode[]   │     │ - sample()     │  │
│  └──────────────┘     └──────────────┘     └────────────────┘  │
│         │                                          │           │
│         │                                          ▼           │
│         │                                 ┌────────────────┐   │
│         └──── free on completion ────────▶│  BlockManager  │   │
│                                           │ - allocate()   │   │
│                                           │ - free()       │   │
│                                           └────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Summary

| Technique           | Problem Solved                     | Trade-off               |
| ------------------- | ---------------------------------- | ----------------------- |
| PagedAttention      | Memory fragmentation               | Extra indirection cost  |
| Continuous Batching | Request blocking, underutilization | Scheduling overhead     |
| Chunked Prefill     | Prefill blocking decode            | Slightly slower prefill |

### Further Reading

- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)

---

## Appendix A: Benchmark Results

### A.1. GPU: Continuous Batching (HF Transformers, Qwen3-0.6B)

**Setup**: Qwen3-0.6B on RTX 3060 12GB, batch_size=8, 20 requests with mixed prompt lengths (128–512 tokens) and mixed output lengths (24–128 tokens). All requests arrive at t=0. The static mode keeps finished requests in the batch until the longest request ends; continuous mode reuses slots immediately.

| Mode | Output tok/s | Wall Time | TTFT p50 | E2EL p50 | Wasted Decode Slots |
|------|-------------|-----------|----------|----------|---------------------|
| Request-Level Static | 3.6 | 253.9 s | 170.4 s | 181.0 s | 1,528 |
| Continuous Slot Reuse | 16.7 | 54.0 s | 10.8 s | 21.1 s | 0 |

**4.7x throughput improvement.** Request-level batching wastes 1,528 decode slots because finished requests sit idle until the longest request in their micro-batch completes. Continuous batching eliminates this entirely.

The same experiment on Qwen2.5-3B-Instruct shows **2.7x throughput improvement** (1.9 → 5.2 tok/s) and **6.0x TTFT improvement**.

### A.2. GPU: Chunked Prefill (vLLM, Qwen3-0.6B)

**Setup**: Qwen3-0.6B on RTX 3060 12GB via vLLM, 20 prompts with random input ~4096 tokens and random output ~512 tokens, request_rate=1.0 QPS.

| Config | Output tok/s | Mean TTFT | Mean TPOT | p99 ITL |
|--------|-------------|-----------|-----------|---------|
| No chunk (max_num_seqs=8) | 256.2 | 4,012 ms | 23.5 ms | 103.5 ms |
| No chunk (max_num_seqs=16) | 278.3 | 1,048 ms | 32.6 ms | 126.8 ms |
| Chunk 512 (max_num_seqs=16) | 277.9 | 1,288 ms | 31.8 ms | 91.5 ms |
| Chunk 1024 (max_num_seqs=16) | 282.2 | 1,128 ms | 31.0 ms | 123.7 ms |
| Chunk 2048 (max_num_seqs=16) | 285.9 | 959 ms | 30.6 ms | 178.8 ms |

Increasing `max_num_seqs` from 8 to 16 boosts throughput (+8.6%) by keeping more requests active. With chunking enabled, **p99 inter-token latency (ITL) drops** (126.8 → 91.5 ms at chunk 512) — long prefills no longer create decode stalls. Larger chunk sizes trade better throughput for higher tail latency.

### A.3. CPU: nano-vLLM Results

#### Key Metrics

The `BenchmarkMetrics` struct (`include/scheduler/benchmark.hpp`) collects per-request timing data:

```cpp
struct BenchmarkMetrics {
    int total_requests = 0;
    int total_prompt_tokens = 0;
    int total_generated_tokens = 0;
    double total_prefill_time_ms = 0.0;
    double total_decode_time_ms = 0.0;
    double total_time_ms = 0.0;

    double prefill_tokens_per_sec() const;
    double decode_tokens_per_sec() const;
    double overall_tokens_per_sec() const;

    void add_request(const Request &request);
    void print() const;
};
```

**What to measure:**

1. **TTFT (Time to First Token)**: User-perceived latency
2. **TPOT (Time Per Output Token)**: Streaming smoothness
3. **Throughput**: Tokens generated per second system-wide
4. **Memory Utilization**: KV cache efficiency (via BlockManager)

#### Benchmark Scenarios

nano-vLLM includes test scenarios in `examples/`:

| Scenario                | Description            | Focus                     |
| ----------------------- | ---------------------- | ------------------------- |
| `simple.json`           | Single short request   | Baseline                  |
| `short_burst.json`      | Many short requests    | Throughput                |
| `long_context.json`     | Long prompts           | Prefill efficiency        |
| `mixed_length.json`     | Varied prompt lengths  | Scheduling fairness       |
| `stress_test.json`      | High concurrency       | System limits             |
| `code_generation.json`  | Code generation tasks  | Long-form output          |
| `conversation.json`     | Multi-turn dialogue    | Conversational workloads  |
| `creative_writing.json` | Creative writing tasks | Open-ended generation     |
| `technical_qa.json`     | Technical Q&A          | Short output, long input  |
| `temperature_test.json` | Sampling variations    | Temperature/top-p effects |

#### Results

Using the stories15M model on Apple Silicon with 6 mixed-length requests:

| # | Configuration | Total Time | Prefill tok/s | Decode tok/s | Overall tok/s |
|---|---------------|------------|---------------|--------------|---------------|
| 1 | Sequential + StdAttn | 593.90 ms | 1286.45 | 534.22 | 769.50 |
| 2 | Sequential + PagedAttn | 594.68 ms | 1277.75 | 534.93 | 768.48 |
| 3 | Batched(4) + PagedAttn + No Chunk | 607.87 ms | 1232.90 | 516.74 | 751.81 |
| 4 | Batched(4) + PagedAttn + Chunk(64) | 604.96 ms | 1209.46 | 525.93 | 755.42 |

**Key observation**: Continuous batching (run 3) is **2.2% slower** than sequential (run 2), and adding chunked prefill (run 4) is roughly even with unchunked batching. This is the opposite of what happens on GPU.

### A.4. Why CPU and GPU Diverge

On **GPU**, batched requests share parallel matrix operations (GEMM) — more requests per batch means better utilization.

On **CPU**, each request calls `model.forward()` **sequentially** — the "batch" is a scheduling abstraction with no compute parallelism:

```
CPU (sequential within "batch"):
  Iteration N:  [Req A forward] → [Req B forward] → [Req C forward] → overhead
                 \_____________/   \_____________/   \_____________/   \________/
                  same speed as     same speed as     same speed as     pure cost
                  sequential        sequential        sequential

GPU (parallel within batch):
  Iteration N:  [Req A ─┐
                 Req B ──┤ fused GEMM  ] → overhead
                 Req C ──┘              /   \________/
                 \____________________/      amortized
                  faster than 3x sequential
```

The overhead comes from:

1. **Scheduler overhead**: Batch formation, priority evaluation, token budget accounting each iteration
2. **Block allocation cost**: Managing blocks for multiple concurrent sequences
3. **Chunk boundary cost**: More iterations for the same work when chunking

Despite the throughput penalty, continuous batching on CPU still provides **scheduling fairness** — shorter requests finish earlier. Chunked prefill improves **decode latency fairness** by allowing decode tokens between prefill chunks.

| Feature | CPU Impact | GPU Impact |
|---------|-----------|------------|
| Continuous Batching | -2.2% throughput (scheduling overhead) | Major throughput gain (parallel GEMM) |
| Chunked Prefill | ~even throughput, better decode latency | Better latency + GPU utilization |

### A.5. Testing Chunked Prefill

Use the `--max-tokens-per-batch` (or `-bt`) CLI option to control the token budget:

```bash
# With default model (max_seq_len=256), use -bt 64 to trigger chunking
./build/main models/model.bin --input-json examples/chunked_prefill_test.json -b 4 -bt 64
```

Example output showing a 72-token prompt split into chunks of 64 + 8:

```
Running in batched mode with max_batch_size=4, max_tokens_per_batch=64
Iteration 0: 1 requests (prefill), 64 tokens   # First chunk
Iteration 1: 3 requests (prefill), 28 tokens   # Remaining 8 + other requests
Request 0 prefill complete: 72 tokens
```

### A.6. Current Limitation: Scheduling Simulation

The current implementation processes each request completely before moving to the next. True continuous batching requires:

1. Per-request KV cache isolation
2. Batched forward pass with multiple sequences
3. Model architecture changes for concurrent execution

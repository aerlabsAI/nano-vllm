# HF Batching Baseline (Request-Level vs Continuous)

This directory contains a vanilla Hugging Face control experiment used to address the PR feedback that `max-num-seqs` sweeps alone do not represent request-level batching.

## What this script compares

- `request_level_static`: fixed micro-batches. Finished requests stay in the batch until the longest one completes.
- `continuous_slot_reuse`: when a request finishes, its slot is immediately reused by the next waiting request.

Both modes use the same model, request set, and decode rule (greedy), and run on vanilla `transformers` (not vLLM scheduler).

## Script

- `request_level_vs_continuous_hf.py`

## Example runs

```bash
# Qwen3-0.6B
docker run --rm --gpus all \
  -v "$(pwd)":/workspace -w /workspace \
  -e HF_HOME=/workspace/.bench-cache/hf \
  --entrypoint python3 \
  vllm/vllm-openai:latest \
  docs/data/hf-batching/request_level_vs_continuous_hf.py \
  --model Qwen/Qwen3-0.6B \
  --batch-size 8 \
  --num-requests 20 \
  --output docs/data/hf-batching/qwen3-0.6b-request-vs-continuous-20260226.json

# Qwen2.5-3B-Instruct
docker run --rm --gpus all \
  -v "$(pwd)":/workspace -w /workspace \
  -e HF_HOME=/workspace/.bench-cache/hf \
  --entrypoint python3 \
  vllm/vllm-openai:latest \
  docs/data/hf-batching/request_level_vs_continuous_hf.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --batch-size 8 \
  --num-requests 20 \
  --output docs/data/hf-batching/qwen2.5-3b-request-vs-continuous-20260226.json
```

The output JSON includes per-mode aggregate metrics and per-request latencies.

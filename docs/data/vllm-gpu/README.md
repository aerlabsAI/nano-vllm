# vLLM GPU Raw Benchmark JSON

This directory stores the raw `vllm bench serve --save-result --save-detailed` JSON files used by the GPU benchmark tables.

## Workload

- `num-prompts=20`
- `dataset-name=random`
- `random-input-len=4096`
- `random-output-len=512`
- `random-range-ratio=0.3334`
- `request-rate=1`

## Qwen3-0.6B (6 cases)

- `chunk_512_s16` -> `qwen3-0.6b/openai-1.0qps-Qwen3-0.6B-20260223-211942.json`
- `chunk_1024_s16` -> `qwen3-0.6b/openai-1.0qps-Qwen3-0.6B-20260223-212417.json`
- `chunk_2048_s16` -> `qwen3-0.6b/openai-1.0qps-Qwen3-0.6B-20260223-212826.json`
- `batch_8_t1024` -> `qwen3-0.6b/openai-1.0qps-Qwen3-0.6B-20260223-213235.json`
- `batch_16_t1024` -> `qwen3-0.6b/openai-1.0qps-Qwen3-0.6B-20260223-213652.json`
- `batch_24_t1024` -> `qwen3-0.6b/openai-1.0qps-Qwen3-0.6B-20260223-214050.json`

## Qwen2.5-3B-Instruct (6 cases, matched matrix)

- `chunk_512_s16` -> `qwen2.5-3b/openai-1.0qps-Qwen2.5-3B-Instruct-20260223-034033.json`
- `chunk_1024_s16` -> `qwen2.5-3b/openai-1.0qps-Qwen2.5-3B-Instruct-20260223-034348.json`
- `chunk_2048_s16` -> `qwen2.5-3b/openai-1.0qps-Qwen2.5-3B-Instruct-20260223-034651.json`
- `batch_8_t1024` -> `qwen2.5-3b/openai-1.0qps-Qwen2.5-3B-Instruct-20260223-035001.json`
- `batch_16_t1024` -> `qwen2.5-3b/openai-1.0qps-Qwen2.5-3B-Instruct-20260223-035302.json`
- `batch_24_t1024` -> `qwen2.5-3b/openai-1.0qps-Qwen2.5-3B-Instruct-20260223-035556.json`

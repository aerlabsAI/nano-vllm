<h1 align="center">nano-vllm</h1>

<h3 align="center">A minimalist, educational LLM inference engine built from scratch with C++.</h3>

## Project Structure

```
.
├── src/                   # Source code
│   └── main.cpp           # Main LLM inference engine
├── include/               # Header files
│   ├── core/              # Core components (model, tokenizer, attention, sampler)
│   ├── ops/               # Operations (activation, linear, normalization, positional)
│   ├── scheduler/         # Block manager for memory scheduling
│   └── utils/             # Utilities (logger, argparser, path, benchmark, comparison)
├── models/                # Model checkpoints and tokenizer
├── docs/                  # Documentation
├── CMakeLists.txt         # CMake configuration
└── Makefile               # Development commands
```

## Quick Start

1. **Initialize & Download Model**:

   ```bash
   make init
   ```

2. **Build**:

   ```bash
   make clang
   cmake --build build
   ```

3. **Run**:

   ```bash
   ./build/main models -i "Hello"
   ```

4. **Benchmark with JSON workload**:

   ```bash
   # Sequential
   ./build/main models --input-json examples/comparison_workload.json

   # Batched with continuous batching
   ./build/main models --input-json examples/comparison_workload.json -b 4

   # Async with dynamic arrivals
   ./build/main models --input-json examples/comparison_workload.json -b 4 --async
   ```

5. **Save & Compare results**:

   ```bash
   # Save results from two different configurations
   ./build/main models --input-json examples/comparison_workload.json --save-results result_a.json
   ./build/main models --input-json examples/comparison_workload.json -b 4 --save-results result_b.json

   # Compare side-by-side (no model needed)
   ./build/main --compare-a result_a.json --compare-b result_b.json
   ```

## Requirements

- CMake 3.20+
- C++20 compliant compiler (Clang, GCC)

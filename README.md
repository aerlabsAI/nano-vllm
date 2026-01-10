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
│   └── utils/             # Utilities (logger, argparser, path handler)
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
   ```

3. **Run**:

   ```bash
   cd build
   make main
   ./main ../models -i hi
   ```

## Requirements

- CMake 3.20+
- C++20 compliant compiler (Clang, GCC)

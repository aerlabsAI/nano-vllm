<h1 align="center">nano-vllm</h1>

<h3 align="center">A minimalist, educational LLM inference engine built from scratch with C++.</h3>

## Project Structure

```
.
├── src/             # Source code
│   └── main.cpp     # Single-file LLM inference engine
├── models/          # Model checkpoints
├── python_ref/      # Python reference and data
├── CMakeLists.txt   # CMake configuration
└── Makefile         # Development commands
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
   ./build/src_main models/model.bin -i "Once upon a time"
   ```

## Requirements

- CMake 3.20+
- C++20 compliant compiler (Clang, GCC)

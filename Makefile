SHELL := /bin/bash

# Compiler configuration
CXX := clang++
CXX_FLAGS := -std=c++20 -I include
NVCC := nvcc
NVCC_FLAGS := -x cu -std c++20 --gpu-architecture compute_90 --gpu-code sm_90

# Check if NVIDIA GPU is available
HAS_GPU := $(shell nvidia-smi > /dev/null 2>&1 && echo 1 || echo 0)

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	OS := macos
	LLVM_PATH := $(shell brew --prefix $$(brew list 2>/dev/null | grep "^llvm" | head -n 1) 2>/dev/null)
	ifneq ($(LLVM_PATH),)
		export PATH := $(LLVM_PATH)/bin:$(PATH)
	endif
else ifeq ($(UNAME_S),Linux)
	OS := linux
else
	OS := unknown
endif

# Get file from command line arguments (everything after first target)
FILE_ARG = $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
FILE_PATH = $(firstword $(FILE_ARG))

.DEFAULT_GOAL := help
.PHONY: init lint format clang nvcc run clean help

# Initialize development environment
init:
	@if [ "$(OS)" = "macos" ]; then \
		echo "Initializing development environment for macOS..."; \
		if ! command -v brew &> /dev/null; then \
			echo "Error: Homebrew is not installed. Please install it from https://brew.sh"; \
			exit 1; \
		fi; \
		brew install llvm; \
		if ! command -v uv &> /dev/null; then \
			echo "Installing uv..."; \
			curl -LsSf https://astral.sh/uv/install.sh | sh; \
		fi; \
		pip install pre-commit;\
		pre-commit install; \
		echo "Note: CUDA Toolkit is not officially supported on macOS."; \
		echo "For CUDA development, please use Docker or a Linux environment."; \
	elif [ "$(OS)" = "linux" ]; then \
		echo "Initializing development environment for Linux..."; \
		echo 'export PATH=$${PATH}:/usr/local/cuda/bin' > ~/.env; \
		curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-archive-keyring.gpg \
			-o /usr/share/keyrings/cuda-archive-keyring.gpg; \
		sed -i '/signed-by/!s@^deb @deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] @' \
			/etc/apt/sources.list.d/cuda.list; \
		apt-get update && \
		apt-get install -y \
		gdb cuda-toolkit-12-6 \
		clang-tidy clang-format; \
		pip install pre-commit; \
		pre-commit install; \
	else \
		echo "Error: Unsupported OS ($(UNAME_S))"; \
		exit 1; \
	fi
	@$(MAKE) download_models

download_models:
	@echo "Downloading models..."
	@mkdir -p models
	@if command -v wget >/dev/null 2>&1; then \
		echo "Downloading model with wget..."; \
		wget -O models/model.bin "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin"; \
		echo "Downloading tokenizer with wget..."; \
		wget -O models/tokenizer.bin "https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin"; \
	elif command -v curl >/dev/null 2>&1; then \
		echo "wget not found, using curl instead..."; \
		echo "Downloading model..."; \
		curl -L -o models/model.bin "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin"; \
		echo "Downloading tokenizer..."; \
		curl -L -o models/tokenizer.bin "https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin"; \
	else \
		echo "Error: Neither wget nor curl found. Please install wget."; \
		exit 1; \
	fi
	@echo "Download complete!"

# Code quality tools
lint:
	@if [ ! -f build/compile_commands.json ]; then \
		echo "Error: build/compile_commands.json not found. Run 'make clang' or 'make nvcc' first."; \
		exit 1; \
	fi
	find src \
	-name "*.cpp" \
	-o -name "*.cu" \
	| xargs clang-tidy \
	-p build \
	--header-filter='.*' \
	--fix

format:
	find src \
	-name "*.cpp" \
	-o -name "*.h" \
	-o -name "*.hpp" \
	-o -name "*.cu" \
	-o -name "*.cuh" \
	| xargs clang-format -i

# Generate CMake build directory with Clang++
# Usage: make clang [DEBUG=1]
clang:
	rm -rf build && \
	cmake -S . -B build \
	-DUSE_NVCC=OFF \
	$(if $(DEBUG),-DUSE_DEBUG=ON)

# Generate CMake build directory with NVCC
# Usage: make nvcc [DEBUG=1]
nvcc:
	@if [ "$(HAS_GPU)" = "0" ]; then \
		echo "Error: No NVIDIA GPU detected. Use 'make clang' instead."; \
		exit 1; \
	fi
	rm -rf build && \
	cmake -S . -B build \
	-DUSE_NVCC=ON \
	$(if $(DEBUG),-DUSE_DEBUG=ON)

# Compile, run, and clean up CUDA/C++ file
# Usage: make run example/test.cu
# Usage: make run example/test.cpp
run:
	@if [ -z "$(FILE_PATH)" ]; then echo "Usage: make run <path/to/file.cu|cpp>"; exit 1; fi
	@if [[ "$(FILE_PATH)" == *.cu ]]; then \
		if [ "$(HAS_GPU)" = "0" ]; then \
			echo "Error: No NVIDIA GPU detected. Cannot run CUDA code."; \
			exit 1; \
		fi; \
		$(NVCC) $(NVCC_FLAGS) -o $(basename $(FILE_PATH)) $(FILE_PATH); \
	else \
		$(CXX) $(CXX_FLAGS) -o $(basename $(FILE_PATH)) $(FILE_PATH); \
	fi
	./$(basename $(FILE_PATH))
	rm $(basename $(FILE_PATH))

# Clean up all compiled binaries
# Usage: make clean
clean:
	find ./src -type f -executable \
	! -name "*.c" \
	! -name "*.cpp" \
	! -name "*.h" \
	! -name "*.cu" \
	! -name "*.cuh" \
	! -name "*.sh" \
	! -name "*.py" \
	-delete

# Show help
help:
	@echo "Available targets:"
	@echo "  Development environment:"
	@echo "    init     - Initialize development environment with clang tools and pre-commit"
	@echo "    lint     - Run clang-tidy to fix code issues"
	@echo "    format   - Format code with clang-format"
	@echo "    clang    - Generate CMake build directory (Clang++)"
	@echo "    nvcc     - Generate CMake build directory (NVCC)"
	@echo ""
	@echo "  Debug mode: Add DEBUG=1 to any build command"
	@echo "    make clang DEBUG=1    - Clang++ with debug"
	@echo "    make nvcc DEBUG=1     - NVCC with debug"
	@echo ""
	@echo "  CUDA/C++ compilation:"
	@echo "    run      - Compile, run, and clean up CUDA/C++ file"
	@echo "    clean    - Remove all compiled binaries"
	@echo ""
	@echo "Usage: make <target> <path/to/file.cu|cpp>"
	@echo "Example: make run example/test.cu"
	@echo "Example: make run example/test.cpp"

# Catch-all rule to prevent file arguments from being treated as targets
%:
	@:

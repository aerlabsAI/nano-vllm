SHELL := /bin/bash

# Compiler configuration
CXX := clang++
CXX_FLAGS := -std=c++20 -I include

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
.PHONY: init lint format clang clean help

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
	elif [ "$(OS)" = "linux" ]; then \
		echo "Initializing development environment for Linux..."; \
		apt-get update && \
		apt-get install -y \
		gdb clang-tidy clang-format; \
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
		echo "Error: build/compile_commands.json not found. Run 'make clang' first."; \
		exit 1; \
	fi
	find src \
	-name "*.cpp" \
	| xargs clang-tidy \
	-p build \
	--header-filter='.*' \
	--fix

format:
	find src \
	-name "*.cpp" \
	-o -name "*.h" \
	-o -name "*.hpp" \
	| xargs clang-format -i

# Generate CMake build directory with Clang++
# Usage: make clang [DEBUG=1]
clang:
	rm -rf build && \
	cmake -S . -B build \
	$(if $(DEBUG),-DUSE_DEBUG=ON)

# Clean up all compiled binaries
# Usage: make clean
clean:
	find ./src -type f -executable \
	! -name "*.c" \
	! -name "*.cpp" \
	! -name "*.h" \
	! -name "*.sh" \
	! -name "*.py" \
	-delete

# Show help
help:
	@echo "Available targets:"
	@echo "  Development environment:"
	@echo "    init             - Initialize development environment with clang tools and pre-commit"
	@echo "    download_models  - Download model and tokenizer files"
	@echo "    lint             - Run clang-tidy to fix code issues"
	@echo "    format           - Format code with clang-format"
	@echo "    clang            - Generate CMake build directory (Clang++)"
	@echo "    clean    - Remove all compiled binaries"
	@echo ""
	@echo "  Debug mode: Add DEBUG=1 to any build command"
	@echo "    make clang DEBUG=1    - Clang++ with debug"

# Catch-all rule to prevent file arguments from being treated as targets
%:
	@:

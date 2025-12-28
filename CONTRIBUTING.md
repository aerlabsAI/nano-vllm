# Contributing to nano-vllm

Thank you for your interest in contributing to nano-vllm! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)
- [Pull Request Process](#pull-request-process)
- [PR Title Convention](#pr-title-convention)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Commit Messages](#commit-messages)

## Getting Started

1. Fork the repository
2. Clone your fork:

   ```bash
   git clone https://github.com/YOUR_USERNAME/nano-vllm.git
   cd nano-vllm
   ```

3. Add the upstream repository:

   ```bash
   git remote add upstream https://github.com/aerlabsAI/nano-vllm.git
   ```

4. Create a new branch for your changes:

   ```bash
   git checkout -b your-feature-branch
   ```

## Developer Certificate of Origin (DCO)

This project uses the Developer Certificate of Origin (DCO) to ensure that contributors have the right to submit their code. All commits must be signed off to indicate that you agree to the DCO.

### How to Sign Off Commits

Add the `-s` flag when committing:

```bash
git commit -m "Your commit message" -s
```

This adds a "Signed-off-by" line to your commit message:

```
Your commit message

Signed-off-by: Your Name <your.email@example.com>
```

### What is DCO?

By signing off your commits, you certify that:

- You have the right to submit the code under the project's license
- You created the contribution, or it's based on previous work under a compatible license
- You understand the contribution is public and may be redistributed

For more information, see [developercertificate.org](https://developercertificate.org/).

### Configure Git to Always Sign Off

To automatically sign off all your commits, configure git:

```bash
git config --global format.signoff true
```

**Note:** All commits in your PR must be signed off. PRs with unsigned commits will not be merged.

## Pull Request Process

### Requirements

- All PRs must have **at least 1 approval** from a reviewer before merging
- PRs are merged using **squash merge** to maintain a clean commit history
- All CI checks must pass before merging
- Follow the PR template when creating a pull request

### Steps

1. Make your changes in your feature branch
2. Test your changes thoroughly (see [Testing](#testing))
3. Push your changes to your fork:

   ```bash
   git push origin your-feature-branch
   ```

4. Create a Pull Request from your fork to the main repository
5. Fill out the PR template completely
6. Wait for review and address any feedback
7. Once approved and all checks pass, your PR will be squash merged

## PR Title Convention

**All PR titles MUST start with a category tag in square brackets.** The title should be concise and descriptive of the changes.

### Available Categories

- **[Bug]** - For bug fixes

  ```
  Example: [Bug] Fix memory leak in PagedAttention
  ```

- **[CI]** - For build system or continuous integration improvements

  ```
  Example: [CI] Add automated testing workflow
  ```

- **[Doc]** - For documentation fixes and improvements

  ```
  Example: [Doc] Update installation instructions for CUDA 12.6
  ```

- **[Model]** - For adding a new model or improving an existing model
  - Model name should appear in the title

  ```
  Example: [Model] Add support for LLaMA 3.1
  ```

- **[Frontend]** - For changes on the nano-vllm frontend (e.g., API interface, inference class, etc.)

  ```
  Example: [Frontend] Add streaming response support
  ```

- **[Kernel]** - For changes affecting CUDA kernels or other compute kernels

  ```
  Example: [Kernel] Optimize attention kernel for A100 GPUs
  ```

- **[Core]** - For changes in the core nano-vllm logic (e.g., scheduler, memory management, etc.)

  ```
  Example: [Core] Implement continuous batching scheduler
  ```

- **`[Hardware][Vendor]`** - For hardware-specific changes
  - Vendor name should appear in the prefix

  ```
  Example: [Hardware][AMD] Add ROCm support for AMD GPUs
  Example: [Hardware][Intel] Optimize for Intel Arc GPUs
  ```

- **[Misc]** - For PRs that do not fit the above categories
  - Use this sparingly

  ```
  Example: [Misc] Update project logo
  ```

### Multiple Categories

If your PR touches multiple areas, use the primary category and mention others in the PR description.

## Code Style Guidelines

### C++ Code Style

- Follow modern C++ best practices (C++20)
- Use consistent indentation (4 spaces)
- Add comments for complex logic
- Keep functions focused and modular
- Use meaningful variable and function names
- Leverage C++20 features where appropriate (concepts, ranges, coroutines, etc.)

### CUDA Code Style

- Follow CUDA best practices
- Document kernel launch configurations
- Add performance notes for optimization decisions
- Use appropriate memory types (shared, global, constant)

### File Organization

- Header files (.h, .cuh) go in `include/`
- Implementation files (.cpp, .cu) go in `src/`
- Example files go in `example/`

## Testing

Before submitting a PR, ensure your changes are tested:

```bash
# Build with clang
make clang

# Or build with CUDA
make cuda

# Run example tests
./build/example_test

# Run your specific tests
# Add test commands here
```

Include test results in your PR description.

## Commit Messages

While individual commits in your branch can have any format (since we use squash merge), it's good practice to write clear commit messages using the following format:

```
<type>: <description>
```

### Commit Types

- `add:` - Add new features or functionality
- `fix:` - Fix bugs
- `docs:` - Documentation changes
- `style:` - Code formatting, whitespace changes (no logic changes)
- `refactor:` - Code refactoring without changing functionality
- `test:` - Add or modify tests
- `perf:` - Performance improvements
- `chore:` - Build system, CI, or tooling changes
- `revert:` - Revert a previous commit

### Commit Guidelines

- **Always use the `-s` flag** to sign off your commits (see [DCO](#developer-certificate-of-origin-dco))
- Use lowercase for the type
- Use imperative mood in description ("add feature" not "added feature")
- Keep the first line under 72 characters
- Reference issues and PRs when relevant (#123)

### Examples

```bash
git commit -m "add: PagedAttention kernel optimization" -s
git commit -m "fix: memory leak in continuous batching" -s
git commit -m "docs: update CUDA installation guide" -s
git commit -m "refactor: simplify scheduler logic" -s
git commit -m "perf: optimize attention kernel for A100" -s
```

## Questions?

If you have questions about contributing, feel free to:

- Open an issue for discussion
- Ask in your PR
- Check existing PRs and issues for similar topics

Thank you for contributing to nano-vllm!

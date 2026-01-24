#pragma once

#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "utils/logger.hpp"

// ============================================================================
// Block Manager - Physical Memory Block Allocation
// ============================================================================

class BlockManager
{
public:
    BlockManager(int num_blocks, int block_size)
        : num_blocks_(num_blocks)
        , block_size_(block_size)
        , free_blocks_(num_blocks, true)
        , num_free_blocks_(num_blocks)
    {
        LOG_INFO("BlockManager initialized: ", num_blocks, " blocks of size ", block_size);
    }

    // Allocate a single block
    // Returns: physical block ID, or -1 if no free blocks
    int allocate_block()
    {
        if (num_free_blocks_ == 0) {
            LOG_WARNING("No free blocks available");
            return -1;
        }

        // Find first free block
        for (int i = 0; i < num_blocks_; i++) {
            if (free_blocks_[i]) {
                free_blocks_[i] = false;
                num_free_blocks_--;
                return i;
            }
        }
        return -1;
    }

    // TODO: Implement proper memory cleanup to prevent leaks
    // Free a single block
    void free_block(int block_id)
    {
        if (block_id < 0 || block_id >= num_blocks_) {
            LOG_ERROR("Invalid block_id: ", block_id);
            throw std::runtime_error("Invalid block_id");
        }

        if (free_blocks_[block_id]) {
            LOG_WARNING("Block ", block_id, " is already free");
            return;
        }

        free_blocks_[block_id] = true;
        num_free_blocks_++;
    }

    // TODO: Use for batch processing or preallocation optimization
    // Allocate multiple blocks for a sequence
    // Returns: vector of physical block IDs
    std::vector<int> allocate_sequence(int num_tokens)
    {
        int num_blocks_needed = (num_tokens + block_size_ - 1) / block_size_; // Ceiling division

        if (num_blocks_needed > num_free_blocks_) {
            LOG_ERROR("Not enough free blocks: need ", num_blocks_needed, ", have ", num_free_blocks_);
            throw std::runtime_error("Out of memory");
        }

        std::vector<int> allocated_blocks;
        allocated_blocks.reserve(num_blocks_needed);

        for (int i = 0; i < num_blocks_needed; i++) {
            int block_id = allocate_block();
            if (block_id == -1) {
                // Rollback: free all previously allocated blocks
                for (int freed_block : allocated_blocks) {
                    free_block(freed_block);
                }
                throw std::runtime_error("Failed to allocate sequence");
            }
            allocated_blocks.push_back(block_id);
        }

        return allocated_blocks;
    }

    // TODO: Use for memory cleanup in batch processing
    // Free all blocks in a sequence
    void free_sequence(const std::vector<int> &block_ids)
    {
        for (int block_id : block_ids) {
            free_block(block_id);
        }
    }

    // TODO: Expose metrics for monitoring (KV cache stats, memory pressure, etc.)
    // Similar to vLLM's KV cache activation and hit rate logging

    // Get number of free blocks
    int get_num_free_blocks() const { return num_free_blocks_; }

    // Get total number of blocks
    int get_num_blocks() const { return num_blocks_; }

    // Get block size
    int get_block_size() const { return block_size_; }

    // Check if a block is free
    bool is_free(int block_id) const
    {
        if (block_id < 0 || block_id >= num_blocks_)
            return false;
        return free_blocks_[block_id];
    }

    // Get memory utilization (0.0 to 1.0)
    float get_utilization() const { return 1.0f - (static_cast<float>(num_free_blocks_) / num_blocks_); }

    // ========================================================================
    // Per-Request Block Management (Thread-Safe)
    // ========================================================================

    // Allocate blocks for a specific request
    // Returns: vector of allocated block IDs
    std::vector<int> allocate_for_request(int request_id, int num_tokens)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<int> blocks = allocate_sequence_internal(num_tokens);
        if (!blocks.empty()) {
            request_blocks_[request_id].insert(request_blocks_[request_id].end(), blocks.begin(), blocks.end());
        }
        return blocks;
    }

    // Allocate a single block for a specific request
    // Returns: block ID, or -1 if no free blocks
    int allocate_block_for_request(int request_id)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        int block_id = allocate_block_internal();
        if (block_id >= 0) {
            request_blocks_[request_id].push_back(block_id);
        }
        return block_id;
    }

    // Free all blocks allocated to a request
    void free_request(int request_id)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = request_blocks_.find(request_id);
        if (it == request_blocks_.end()) {
            return;
        }

        for (int block_id : it->second) {
            free_block_internal(block_id);
        }
        request_blocks_.erase(it);
        LOG_INFO("Freed all blocks for request ", request_id);
    }

    // Get blocks allocated to a request
    std::vector<int> get_request_blocks(int request_id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = request_blocks_.find(request_id);
        if (it == request_blocks_.end()) {
            return {};
        }
        return it->second;
    }

    // Get number of blocks allocated to a request
    int get_request_block_count(int request_id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = request_blocks_.find(request_id);
        if (it == request_blocks_.end()) {
            return 0;
        }
        return static_cast<int>(it->second.size());
    }

    // Get number of active requests
    int get_num_active_requests() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(request_blocks_.size());
    }

    // Thread-safe version of allocate_block
    int allocate_block_safe()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return allocate_block_internal();
    }

    // Thread-safe version of free_block
    void free_block_safe(int block_id)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        free_block_internal(block_id);
    }

private:
    int               num_blocks_;      // Total number of blocks
    int               block_size_;      // Size of each block (in tokens)
    std::vector<bool> free_blocks_;     // Track which blocks are free
    int               num_free_blocks_; // Count of free blocks

    // Thread safety
    mutable std::mutex mutex_;

    // Per-request block tracking
    std::unordered_map<int, std::vector<int>> request_blocks_;

    // Internal allocation (no locking)
    int allocate_block_internal()
    {
        if (num_free_blocks_ == 0) {
            return -1;
        }
        for (int i = 0; i < num_blocks_; i++) {
            if (free_blocks_[i]) {
                free_blocks_[i] = false;
                num_free_blocks_--;
                return i;
            }
        }
        return -1;
    }

    // Internal free (no locking)
    void free_block_internal(int block_id)
    {
        if (block_id < 0 || block_id >= num_blocks_) {
            return;
        }
        if (!free_blocks_[block_id]) {
            free_blocks_[block_id] = true;
            num_free_blocks_++;
        }
    }

    // Internal sequence allocation (no locking)
    std::vector<int> allocate_sequence_internal(int num_tokens)
    {
        int num_blocks_needed = (num_tokens + block_size_ - 1) / block_size_;

        if (num_blocks_needed > num_free_blocks_) {
            return {};
        }

        std::vector<int> allocated_blocks;
        allocated_blocks.reserve(num_blocks_needed);

        for (int i = 0; i < num_blocks_needed; i++) {
            int block_id = allocate_block_internal();
            if (block_id == -1) {
                // Rollback
                for (int freed_block : allocated_blocks) {
                    free_block_internal(freed_block);
                }
                return {};
            }
            allocated_blocks.push_back(block_id);
        }
        return allocated_blocks;
    }
};

#pragma once

#include <stdexcept>
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

    // Free all blocks in a sequence
    void free_sequence(const std::vector<int> &block_ids)
    {
        for (int block_id : block_ids) {
            free_block(block_id);
        }
    }

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

private:
    int               num_blocks_;      // Total number of blocks
    int               block_size_;      // Size of each block (in tokens)
    std::vector<bool> free_blocks_;     // Track which blocks are free
    int               num_free_blocks_; // Count of free blocks
};

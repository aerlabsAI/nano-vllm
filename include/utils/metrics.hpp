#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// Forward declaration
struct Config;

// ============================================================================
// KV Cache Metrics - Compare Standard vs PagedAttention Memory Usage
// ============================================================================

class KVCacheMetrics
{
public:
    void set_sequence_length(int len) { sequence_length_ = len; }
    void set_blocks_used(int blocks) { blocks_used_ = blocks; }

    // Calculate KV cache memory size in bytes
    // KV Cache = n_layers × seq_tokens × n_kv_heads × head_dim × sizeof(float) × 2 (key + value)
    static size_t calculate_kv_cache_bytes(int n_layers, int seq_tokens, int n_kv_heads, int head_dim)
    {
        return static_cast<size_t>(n_layers) * static_cast<size_t>(seq_tokens) * static_cast<size_t>(n_kv_heads)
             * static_cast<size_t>(head_dim) * sizeof(float) * 2; // key + value
    }

    // Format bytes to human-readable string (KB, MB, GB)
    static std::string format_bytes(size_t bytes)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);

        if (bytes >= 1024ULL * 1024 * 1024) {
            oss << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) << " GB";
        }
        else if (bytes >= 1024ULL * 1024) {
            oss << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MB";
        }
        else if (bytes >= 1024ULL) {
            oss << (static_cast<double>(bytes) / 1024.0) << " KB";
        }
        else {
            oss << bytes << " B";
        }
        return oss.str();
    }

    // Print comparison between Standard Attention and PagedAttention
    void print_comparison(int n_layers, int n_kv_heads, int head_dim, int max_seq_len, int block_size) const
    {
        // Calculate memory for Standard Attention (reserves full max_seq_len)
        size_t standard_memory = calculate_kv_cache_bytes(n_layers, max_seq_len, n_kv_heads, head_dim);

        // Calculate memory for PagedAttention (only blocks actually used)
        int    paged_tokens = blocks_used_ * block_size;
        size_t paged_memory = calculate_kv_cache_bytes(n_layers, paged_tokens, n_kv_heads, head_dim);

        // Calculate savings
        size_t savings_bytes   = standard_memory - paged_memory;
        double savings_percent = (static_cast<double>(savings_bytes) / static_cast<double>(standard_memory)) * 100.0;

        // Print formatted comparison
        std::cout << "\n";
        std::cout << "┌─────────────────────────────────────────────────────────────────┐\n";
        std::cout << "│                  KV Cache Memory Comparison                     │\n";
        std::cout << "├─────────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Sequence Length:              " << std::setw(6) << sequence_length_ << " tokens"
                  << std::string(24, ' ') << "│\n";
        std::cout << "├─────────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Standard Attention:                                             │\n";
        std::cout << "│   KV Cache Size:              " << std::setw(10) << format_bytes(standard_memory)
                  << " (reserved for " << max_seq_len << " seq)" << std::string(4, ' ') << "│\n";
        std::cout << "│                                                                 │\n";
        std::cout << "│ PagedAttention:                                                 │\n";
        std::cout << "│   Blocks Used:                " << std::setw(6) << blocks_used_ << " blocks (" << paged_tokens
                  << " token capacity)" << std::string(10, ' ') << "│\n";
        std::cout << "│   KV Cache Size:              " << std::setw(10) << format_bytes(paged_memory)
                  << " (actually used)" << std::string(12, ' ') << "│\n";
        std::cout << "├─────────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Memory Savings:               " << std::setw(10) << format_bytes(savings_bytes) << " ("
                  << std::fixed << std::setprecision(1) << savings_percent << "%)" << std::string(18, ' ') << "│\n";
        std::cout << "└─────────────────────────────────────────────────────────────────┘\n";
        std::cout << std::endl;
    }

private:
    int sequence_length_ = 0;
    int blocks_used_     = 0;
};

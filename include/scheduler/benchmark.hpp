#pragma once

#include <iomanip>
#include <iostream>

// ============================================================================
// Benchmark Metrics - Performance measurement for request processing
// ============================================================================

struct BenchmarkMetrics
{
    int    total_requests         = 0;
    int    total_prompt_tokens    = 0;
    int    total_generated_tokens = 0;
    double total_prefill_time_ms  = 0.0;
    double total_decode_time_ms   = 0.0;
    double total_time_ms          = 0.0;

    double prefill_tokens_per_sec() const
    {
        return total_prefill_time_ms > 0 ? (total_prompt_tokens * 1000.0 / total_prefill_time_ms) : 0.0;
    }

    double decode_tokens_per_sec() const
    {
        return total_decode_time_ms > 0 ? (total_generated_tokens * 1000.0 / total_decode_time_ms) : 0.0;
    }

    double overall_tokens_per_sec() const
    {
        int total_tokens = total_prompt_tokens + total_generated_tokens;
        return total_time_ms > 0 ? (total_tokens * 1000.0 / total_time_ms) : 0.0;
    }

    void print() const
    {
        std::cout << "\n========================================\n";
        std::cout << "         BENCHMARK RESULTS\n";
        std::cout << "========================================\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Total requests:         " << total_requests << "\n";
        std::cout << "Total prompt tokens:    " << total_prompt_tokens << "\n";
        std::cout << "Total generated tokens: " << total_generated_tokens << "\n";
        std::cout << "----------------------------------------\n";
        std::cout << "Prefill time:           " << total_prefill_time_ms << " ms\n";
        std::cout << "Decode time:            " << total_decode_time_ms << " ms\n";
        std::cout << "Total time:             " << total_time_ms << " ms\n";
        std::cout << "----------------------------------------\n";
        std::cout << "Prefill throughput:     " << prefill_tokens_per_sec() << " tokens/sec\n";
        std::cout << "Decode throughput:      " << decode_tokens_per_sec() << " tokens/sec\n";
        std::cout << "Overall throughput:     " << overall_tokens_per_sec() << " tokens/sec\n";
        std::cout << "========================================\n";
    }

    void add_request(const struct Request &request);
};

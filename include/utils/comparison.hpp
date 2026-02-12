#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "utils/benchmark_result.hpp"
#include "utils/metrics.hpp"

// ============================================================================
// Comparison - Side-by-side display of two benchmark results
// ============================================================================

class Comparison
{
public:
    Comparison(const BenchmarkResult &a, const BenchmarkResult &b)
        : a_(a), b_(b)
    {
    }

    void print_table() const
    {
        std::string label_a = describe(a_);
        std::string label_b = describe(b_);

        std::cout << "\n";
        print_header(label_a, label_b);
        print_separator();

        // Timing (lower is better)
        print_row("Total Time", a_.total_time_ms, b_.total_time_ms, "ms", false);
        print_row("Prefill Time", a_.total_prefill_time_ms, b_.total_prefill_time_ms, "ms", false);
        print_row("Decode Time", a_.total_decode_time_ms, b_.total_decode_time_ms, "ms", false);
        print_separator();

        // Throughput (higher is better)
        print_row("Prefill Throughput", a_.prefill_tokens_per_sec(), b_.prefill_tokens_per_sec(), "tok/s", true);
        print_row("Decode Throughput", a_.decode_tokens_per_sec(), b_.decode_tokens_per_sec(), "tok/s", true);
        print_row("Overall Throughput", a_.overall_tokens_per_sec(), b_.overall_tokens_per_sec(), "tok/s", true);
        print_separator();

        // Memory
        std::string mem_a = KVCacheMetrics::format_bytes(a_.memory.kv_cache_bytes);
        std::string mem_b = KVCacheMetrics::format_bytes(b_.memory.kv_cache_bytes);
        print_row_str("KV Cache Memory", mem_a, mem_b,
                      format_diff(static_cast<double>(a_.memory.kv_cache_bytes),
                                  static_cast<double>(b_.memory.kv_cache_bytes)));

        if (a_.memory.blocks_used > 0 || b_.memory.blocks_used > 0) {
            print_row("Blocks Used", a_.memory.blocks_used, b_.memory.blocks_used, "", false);
        }

        print_separator();

        // Summary
        print_row("Total Requests", a_.total_requests, b_.total_requests, "", false);
        print_row("Prompt Tokens", a_.total_prompt_tokens, b_.total_prompt_tokens, "", false);
        print_row("Generated Tokens", a_.total_generated_tokens, b_.total_generated_tokens, "", false);
        print_footer();
    }

    std::string to_json() const
    {
        std::ostringstream os;
        os << std::fixed << std::setprecision(2);

        os << "{\n";
        os << "  \"label_a\": \"" << describe(a_) << "\",\n";
        os << "  \"label_b\": \"" << describe(b_) << "\",\n";
        os << "  \"comparison\": {\n";
        os << "    \"total_time_ms\": { \"a\": " << a_.total_time_ms
           << ", \"b\": " << b_.total_time_ms
           << ", \"diff_percent\": " << calc_diff(a_.total_time_ms, b_.total_time_ms) << " },\n";
        os << "    \"prefill_throughput\": { \"a\": " << a_.prefill_tokens_per_sec()
           << ", \"b\": " << b_.prefill_tokens_per_sec()
           << ", \"diff_percent\": " << calc_diff(a_.prefill_tokens_per_sec(), b_.prefill_tokens_per_sec()) << " },\n";
        os << "    \"decode_throughput\": { \"a\": " << a_.decode_tokens_per_sec()
           << ", \"b\": " << b_.decode_tokens_per_sec()
           << ", \"diff_percent\": " << calc_diff(a_.decode_tokens_per_sec(), b_.decode_tokens_per_sec()) << " },\n";
        os << "    \"overall_throughput\": { \"a\": " << a_.overall_tokens_per_sec()
           << ", \"b\": " << b_.overall_tokens_per_sec()
           << ", \"diff_percent\": " << calc_diff(a_.overall_tokens_per_sec(), b_.overall_tokens_per_sec()) << " },\n";
        os << "    \"kv_cache_bytes\": { \"a\": " << a_.memory.kv_cache_bytes
           << ", \"b\": " << b_.memory.kv_cache_bytes
           << ", \"diff_percent\": " << calc_diff(static_cast<double>(a_.memory.kv_cache_bytes),
                                                   static_cast<double>(b_.memory.kv_cache_bytes)) << " }\n";
        os << "  },\n";
        os << "  \"result_a\": " << a_.to_json() << ",\n";
        os << "  \"result_b\": " << b_.to_json() << "\n";
        os << "}\n";

        return os.str();
    }

private:
    const BenchmarkResult &a_;
    const BenchmarkResult &b_;

    static std::string describe(const BenchmarkResult &r)
    {
        std::string label = r.config.mode;
        if (r.config.paged_attention) label += " + PagedAttn";
        else                          label += " + StdAttn";
        if (r.config.max_batch_size > 1) label += " (b=" + std::to_string(r.config.max_batch_size) + ")";
        return label;
    }

    static double calc_diff(double a, double b)
    {
        if (a == 0.0) return 0.0;
        return ((b - a) / a) * 100.0;
    }

    static std::string format_diff(double a, double b)
    {
        double diff = calc_diff(a, b);
        std::ostringstream os;
        os << std::fixed << std::setprecision(1);
        if (diff > 0) os << "+";
        os << diff << "%";
        return os.str();
    }

    static void print_header(const std::string &label_a, const std::string &label_b)
    {
        std::cout << std::left;
        std::cout << "+--------------------------+--------------------+--------------------+----------+\n";
        std::cout << "| " << std::setw(25) << "Metric"
                  << "| " << std::setw(19) << label_a.substr(0, 18)
                  << "| " << std::setw(19) << label_b.substr(0, 18)
                  << "| " << std::setw(9) << "Diff"
                  << "|\n";
        std::cout << "+--------------------------+--------------------+--------------------+----------+\n";
    }

    static void print_separator()
    {
        std::cout << "+--------------------------+--------------------+--------------------+----------+\n";
    }

    static void print_footer()
    {
        std::cout << "+--------------------------+--------------------+--------------------+----------+\n";
        std::cout << std::endl;
    }

    static void print_row(const std::string &label, double a, double b,
                          const std::string &unit, bool /*higher_is_better*/)
    {
        std::ostringstream val_a, val_b;
        val_a << std::fixed << std::setprecision(2) << a;
        if (!unit.empty()) val_a << " " << unit;
        val_b << std::fixed << std::setprecision(2) << b;
        if (!unit.empty()) val_b << " " << unit;

        print_row_str(label, val_a.str(), val_b.str(), format_diff(a, b));
    }

    static void print_row(const std::string &label, int a, int b,
                          const std::string &unit, bool higher_is_better)
    {
        print_row(label, static_cast<double>(a), static_cast<double>(b), unit, higher_is_better);
    }

    static void print_row_str(const std::string &label, const std::string &a,
                              const std::string &b, const std::string &diff)
    {
        std::cout << std::left;
        std::cout << "| " << std::setw(25) << label
                  << "| " << std::setw(19) << a
                  << "| " << std::setw(19) << b
                  << "| " << std::setw(9) << diff
                  << "|\n";
    }
};

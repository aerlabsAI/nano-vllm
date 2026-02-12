#pragma once

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// Run Configuration - Execution mode and model parameters
// ============================================================================

struct RunConfig
{
    // Execution mode
    std::string mode = "single"; // "single", "sequential", "batched", "async"
    bool        paged_attention       = false;
    int         max_batch_size        = 1;
    int         max_tokens_per_batch  = 512;

    // Model config fields
    int dim         = 0;
    int n_layers    = 0;
    int n_heads     = 0;
    int n_kv_heads  = 0;
    int head_dim    = 0;
    int vocab_size  = 0;
    int max_seq_len = 0;
    int block_size  = 16;
    int num_blocks  = 256;
};

// ============================================================================
// Memory Metrics - KV cache memory usage
// ============================================================================

struct MemoryMetrics
{
    size_t kv_cache_bytes = 0;
    int    blocks_used    = 0;
    int    blocks_total   = 0;
};

// ============================================================================
// Request Result - Per-request performance data
// ============================================================================

struct RequestResult
{
    int    id               = 0;
    int    prompt_tokens    = 0;
    int    generated_tokens = 0;
    double prefill_time_ms  = 0.0;
    double decode_time_ms   = 0.0;
};

// ============================================================================
// Benchmark Result - Complete benchmark output with JSON serialization
// ============================================================================

struct BenchmarkResult
{
    // Configuration
    RunConfig config;

    // Memory
    MemoryMetrics memory;

    // Aggregate metrics
    int    total_requests         = 0;
    int    total_prompt_tokens    = 0;
    int    total_generated_tokens = 0;
    double total_prefill_time_ms  = 0.0;
    double total_decode_time_ms   = 0.0;
    double total_time_ms          = 0.0;

    // Per-request breakdown
    std::vector<RequestResult> per_request;

    // ---- Throughput calculations ----

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

    // ---- JSON serialization ----

    std::string to_json() const
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);

        oss << "{\n";

        // run_config section
        oss << "  \"run_config\": {\n";
        oss << "    \"mode\": \"" << config.mode << "\",\n";
        oss << "    \"paged_attention\": " << (config.paged_attention ? "true" : "false") << ",\n";
        oss << "    \"max_batch_size\": " << config.max_batch_size << ",\n";
        oss << "    \"max_tokens_per_batch\": " << config.max_tokens_per_batch << ",\n";
        oss << "    \"dim\": " << config.dim << ",\n";
        oss << "    \"n_layers\": " << config.n_layers << ",\n";
        oss << "    \"n_heads\": " << config.n_heads << ",\n";
        oss << "    \"n_kv_heads\": " << config.n_kv_heads << ",\n";
        oss << "    \"head_dim\": " << config.head_dim << ",\n";
        oss << "    \"vocab_size\": " << config.vocab_size << ",\n";
        oss << "    \"max_seq_len\": " << config.max_seq_len << ",\n";
        oss << "    \"block_size\": " << config.block_size << ",\n";
        oss << "    \"num_blocks\": " << config.num_blocks << "\n";
        oss << "  },\n";

        // memory section
        oss << "  \"memory\": {\n";
        oss << "    \"kv_cache_bytes\": " << memory.kv_cache_bytes << ",\n";
        oss << "    \"blocks_used\": " << memory.blocks_used << ",\n";
        oss << "    \"blocks_total\": " << memory.blocks_total << "\n";
        oss << "  },\n";

        // metrics section
        oss << "  \"metrics\": {\n";
        oss << "    \"total_requests\": " << total_requests << ",\n";
        oss << "    \"total_prompt_tokens\": " << total_prompt_tokens << ",\n";
        oss << "    \"total_generated_tokens\": " << total_generated_tokens << ",\n";
        oss << "    \"total_prefill_time_ms\": " << total_prefill_time_ms << ",\n";
        oss << "    \"total_decode_time_ms\": " << total_decode_time_ms << ",\n";
        oss << "    \"total_time_ms\": " << total_time_ms << ",\n";
        oss << "    \"prefill_tokens_per_sec\": " << prefill_tokens_per_sec() << ",\n";
        oss << "    \"decode_tokens_per_sec\": " << decode_tokens_per_sec() << ",\n";
        oss << "    \"overall_tokens_per_sec\": " << overall_tokens_per_sec() << "\n";
        oss << "  },\n";

        // per_request section
        oss << "  \"per_request\": [\n";
        for (size_t i = 0; i < per_request.size(); ++i) {
            const auto &r = per_request[i];
            oss << "    {\n";
            oss << "      \"id\": " << r.id << ",\n";
            oss << "      \"prompt_tokens\": " << r.prompt_tokens << ",\n";
            oss << "      \"generated_tokens\": " << r.generated_tokens << ",\n";
            oss << "      \"prefill_time_ms\": " << r.prefill_time_ms << ",\n";
            oss << "      \"decode_time_ms\": " << r.decode_time_ms << "\n";
            oss << "    }";
            if (i + 1 < per_request.size()) {
                oss << ",";
            }
            oss << "\n";
        }
        oss << "  ]\n";

        oss << "}\n";

        return oss.str();
    }

    // ---- File I/O ----

    bool save(const std::string &path) const
    {
        std::ofstream file(path);
        if (!file.is_open()) {
            return false;
        }
        file << to_json();
        return file.good();
    }

    static BenchmarkResult load(const std::string &path);
};

// ============================================================================
// JSON Deserialization - Load BenchmarkResult from file
// ============================================================================

#include "utils/json_parser.hpp"

inline BenchmarkResult BenchmarkResult::load(const std::string &path)
{
    json::JsonParser parser;
    json::JsonObject root = parser.parse_file(path);

    BenchmarkResult result;

    // Parse run_config
    const auto &cfg              = root.get_object("run_config");
    result.config.mode                  = cfg.get_string("mode", "single");
    result.config.paged_attention       = cfg.get_bool("paged_attention", false);
    result.config.max_batch_size        = cfg.get_int("max_batch_size", 1);
    result.config.max_tokens_per_batch  = cfg.get_int("max_tokens_per_batch", 512);
    result.config.dim                   = cfg.get_int("dim", 0);
    result.config.n_layers              = cfg.get_int("n_layers", 0);
    result.config.n_heads               = cfg.get_int("n_heads", 0);
    result.config.n_kv_heads            = cfg.get_int("n_kv_heads", 0);
    result.config.head_dim              = cfg.get_int("head_dim", 0);
    result.config.vocab_size            = cfg.get_int("vocab_size", 0);
    result.config.max_seq_len           = cfg.get_int("max_seq_len", 0);
    result.config.block_size            = cfg.get_int("block_size", 16);
    result.config.num_blocks            = cfg.get_int("num_blocks", 256);

    // Parse memory
    const auto &mem             = root.get_object("memory");
    result.memory.kv_cache_bytes = static_cast<size_t>(mem.get_number("kv_cache_bytes", 0.0));
    result.memory.blocks_used    = mem.get_int("blocks_used", 0);
    result.memory.blocks_total   = mem.get_int("blocks_total", 0);

    // Parse metrics
    const auto &metrics                = root.get_object("metrics");
    result.total_requests              = metrics.get_int("total_requests", 0);
    result.total_prompt_tokens         = metrics.get_int("total_prompt_tokens", 0);
    result.total_generated_tokens      = metrics.get_int("total_generated_tokens", 0);
    result.total_prefill_time_ms       = metrics.get_number("total_prefill_time_ms", 0.0);
    result.total_decode_time_ms        = metrics.get_number("total_decode_time_ms", 0.0);
    result.total_time_ms               = metrics.get_number("total_time_ms", 0.0);

    // Parse per_request
    const auto &requests = root.get_array("per_request");
    for (const auto &req_obj : requests) {
        RequestResult rr;
        rr.id               = req_obj.get_int("id", 0);
        rr.prompt_tokens    = req_obj.get_int("prompt_tokens", 0);
        rr.generated_tokens = req_obj.get_int("generated_tokens", 0);
        rr.prefill_time_ms  = req_obj.get_number("prefill_time_ms", 0.0);
        rr.decode_time_ms   = req_obj.get_number("decode_time_ms", 0.0);
        result.per_request.push_back(rr);
    }

    return result;
}

// ============================================================================
// Build Result - Construct BenchmarkResult from existing types
// ============================================================================

#include "core/model.hpp"
#include "scheduler/benchmark.hpp"
#include "utils/metrics.hpp"

inline BenchmarkResult build_result(const BenchmarkMetrics &metrics,
                                    const Config &model_config,
                                    const std::string &mode,
                                    int max_batch_size,
                                    int max_tokens_per_batch,
                                    const std::vector<Request> &requests)
{
    BenchmarkResult result;

    // Fill run config
    result.config.mode                 = mode;
    result.config.paged_attention      = model_config.use_paged_attention;
    result.config.max_batch_size       = max_batch_size;
    result.config.max_tokens_per_batch = max_tokens_per_batch;
    result.config.dim                  = model_config.dim;
    result.config.n_layers             = model_config.n_layers;
    result.config.n_heads              = model_config.n_heads;
    result.config.n_kv_heads           = model_config.n_kv_heads;
    result.config.head_dim             = model_config.head_dim;
    result.config.vocab_size           = model_config.vocab_size;
    result.config.max_seq_len          = model_config.max_seq_len;
    result.config.block_size           = model_config.block_size;
    result.config.num_blocks           = model_config.num_blocks;

    // Fill aggregate metrics
    result.total_requests         = metrics.total_requests;
    result.total_prompt_tokens    = metrics.total_prompt_tokens;
    result.total_generated_tokens = metrics.total_generated_tokens;
    result.total_prefill_time_ms  = metrics.total_prefill_time_ms;
    result.total_decode_time_ms   = metrics.total_decode_time_ms;
    result.total_time_ms          = metrics.total_time_ms;

    // Fill per-request results
    for (const auto &req : requests) {
        RequestResult rr;
        rr.id               = req.id;
        rr.prompt_tokens    = req.num_prompt_tokens();
        rr.generated_tokens = req.num_generated_tokens();
        rr.prefill_time_ms  = req.prefill_time_ms;
        rr.decode_time_ms   = req.decode_time_ms;
        result.per_request.push_back(rr);
    }

    // Calculate memory metrics
    if (model_config.use_paged_attention) {
        // Count total blocks used across all requests
        // In sequential mode, blocks are freed after each request, so block_tables may be empty.
        // In that case, estimate peak per-request block usage from sequence length.
        int total_blocks_used = 0;
        for (const auto &req : requests) {
            if (!req.block_tables.empty()) {
                total_blocks_used += static_cast<int>(req.block_tables[0].size());
            }
            else {
                int seq_len = req.num_prompt_tokens() + req.num_generated_tokens();
                total_blocks_used += (seq_len + model_config.block_size - 1) / model_config.block_size;
            }
        }
        int paged_tokens = total_blocks_used * model_config.block_size;
        result.memory.kv_cache_bytes = KVCacheMetrics::calculate_kv_cache_bytes(
            model_config.n_layers, paged_tokens, model_config.n_kv_heads, model_config.head_dim);
        result.memory.blocks_used  = total_blocks_used;
        result.memory.blocks_total = model_config.num_blocks;
    }
    else {
        // Standard attention: reserves full max_seq_len per request
        result.memory.kv_cache_bytes = KVCacheMetrics::calculate_kv_cache_bytes(
            model_config.n_layers, model_config.max_seq_len, model_config.n_kv_heads, model_config.head_dim);
        result.memory.blocks_used  = 0;
        result.memory.blocks_total = 0;
    }

    return result;
}

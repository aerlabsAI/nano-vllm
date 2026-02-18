#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "core/model.hpp"
#include "core/sampler.hpp"
#include "core/tokenizer.hpp"
#include "scheduler/async_request_queue.hpp"
#include "scheduler/batched_runner.hpp"
#include "scheduler/benchmark.hpp"
#include "scheduler/request.hpp"
#include "scheduler/request_processor.hpp"
#include "scheduler/request_submitter.hpp"
#include "scheduler/scheduler.hpp"
#include "utils/benchmark_result.hpp"
#include "utils/json_parser.hpp"
#include "utils/logger.hpp"

// ============================================================================
// Single Prompt Mode
// ============================================================================

inline BenchmarkResult run_single_prompt(LlamaModel        &model,
                                         Tokenizer         &tokenizer,
                                         const std::string &prompt,
                                         float              temperature,
                                         float              top_p,
                                         int                steps)
{
    Sampler sampler(model.config.vocab_size, temperature, top_p, std::time(nullptr));

    std::vector<int> tokens = tokenizer.encode(prompt, true, false);
    LOG_INFO("Encoded prompt into ", tokens.size(), " tokens");
    LOG_INFO("Starting generation with temperature=", temperature, " topp=", top_p, " steps=", steps);

    std::cout << "\n" << prompt;
    std::cout.flush();

    auto total_start   = std::chrono::high_resolution_clock::now();
    auto prefill_start = std::chrono::high_resolution_clock::now();

    int pos = 0;
    for (size_t i = 0; i < tokens.size() - 1; i++) {
        model.forward(tokens[i], pos);
        pos++;
    }
    int token = tokens.back();

    auto   prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_ms  = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();

    auto decode_start = std::chrono::high_resolution_clock::now();
    int  generated    = 0;

    for (int s = 0; s < steps; s++) {
        model.forward(token, pos);
        int next_token = sampler.sample(model.state.logits.data());
        std::cout << tokenizer.decode(next_token);
        std::cout.flush();
        token = next_token;
        pos++;
        generated++;
        if (pos >= model.config.max_seq_len)
            break;
    }

    auto   decode_end = std::chrono::high_resolution_clock::now();
    double decode_ms  = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
    auto   total_end  = std::chrono::high_resolution_clock::now();
    double total_ms   = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    std::cout << std::endl;
    LOG_SUCCESS("Generation completed in ", total_ms / 1000.0, " seconds");

    // Build result
    BenchmarkResult result;
    result.config.mode                 = "single";
    result.config.paged_attention      = model.config.use_paged_attention;
    result.config.max_batch_size       = 1;
    result.config.max_tokens_per_batch = 512;
    result.config.dim                  = model.config.dim;
    result.config.n_layers             = model.config.n_layers;
    result.config.n_heads              = model.config.n_heads;
    result.config.n_kv_heads           = model.config.n_kv_heads;
    result.config.head_dim             = model.config.head_dim;
    result.config.vocab_size           = model.config.vocab_size;
    result.config.max_seq_len          = model.config.max_seq_len;
    result.config.block_size           = model.config.block_size;
    result.config.num_blocks           = model.config.num_blocks;

    result.total_requests         = 1;
    result.total_prompt_tokens    = static_cast<int>(tokens.size());
    result.total_generated_tokens = generated;
    result.total_prefill_time_ms  = prefill_ms;
    result.total_decode_time_ms   = decode_ms;
    result.total_time_ms          = total_ms;

    if (model.config.use_paged_attention) {
        int blocks_used              = model.block_tables.empty() ? 0 : static_cast<int>(model.block_tables[0].size());
        int paged_tokens             = blocks_used * model.config.block_size;
        result.memory.kv_cache_bytes = KVCacheMetrics::calculate_kv_cache_bytes(
            model.config.n_layers, paged_tokens, model.config.n_kv_heads, model.config.head_dim);
        result.memory.blocks_used  = blocks_used;
        result.memory.blocks_total = model.config.num_blocks;
    }
    else {
        result.memory.kv_cache_bytes = KVCacheMetrics::calculate_kv_cache_bytes(
            model.config.n_layers, model.config.max_seq_len, model.config.n_kv_heads, model.config.head_dim);
    }

    RequestResult rr;
    rr.id               = 0;
    rr.prompt_tokens    = static_cast<int>(tokens.size());
    rr.generated_tokens = generated;
    rr.prefill_time_ms  = prefill_ms;
    rr.decode_time_ms   = decode_ms;
    result.per_request.push_back(rr);

    return result;
}

// ============================================================================
// JSON Benchmark Mode - Sequential
// ============================================================================

inline BenchmarkResult run_json_sequential(LlamaModel &model, Tokenizer &tokenizer, std::vector<Request> &requests)
{
    RequestProcessor processor(model, tokenizer);
    BenchmarkMetrics metrics;

    auto total_start = std::chrono::high_resolution_clock::now();

    for (auto &request : requests) {
        std::cout << "\n--- Request " << request.id << " ---\n";
        std::cout << "Prompt: " << request.prompt.substr(0, 50) << (request.prompt.size() > 50 ? "..." : "") << "\n";
        std::cout << "Output: ";

        processor.process(request);
        std::cout << "\n";

        metrics.add_request(request);
        processor.reset_state();
    }

    auto total_end        = std::chrono::high_resolution_clock::now();
    metrics.total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    metrics.print();
    return build_result(metrics, model.config, "sequential", 1, 512, requests);
}

// ============================================================================
// JSON Benchmark Mode - Batched (Continuous Batching)
// ============================================================================

inline BenchmarkResult run_json_batched(LlamaModel           &model,
                                        Tokenizer            &tokenizer,
                                        std::vector<Request> &requests,
                                        int                   max_batch_size,
                                        int                   max_tokens_per_batch = 512)
{
    if (!model.config.use_paged_attention && max_batch_size > 1) {
        LOG_WARNING("Non-paged attention uses a shared KV cache; interleaved batching is unsafe. "
                    "Falling back to sequential mode.");
        return run_json_sequential(model, tokenizer, requests);
    }

    SchedulerConfig config;
    config.max_batch_size       = max_batch_size;
    config.max_tokens_per_batch = max_tokens_per_batch;

    Scheduler     scheduler(config);
    BatchedRunner runner(model, tokenizer);

    LOG_INFO("Running in batched mode with max_batch_size=", max_batch_size);

    BenchmarkMetrics metrics = runner.run_all(requests, scheduler);

    metrics.print();
    return build_result(metrics, model.config, "batched", max_batch_size, max_tokens_per_batch, requests);
}

// ============================================================================
// JSON Benchmark Mode - Async (Dynamic Request Arrivals)
// ============================================================================

inline BenchmarkResult run_json_async(LlamaModel           &model,
                                      Tokenizer            &tokenizer,
                                      std::vector<Request> &requests,
                                      int                   max_batch_size,
                                      int                   max_tokens_per_batch = 512)
{
    if (!model.config.use_paged_attention) {
        LOG_WARNING("Async interleaved batching requires paged attention for KV isolation. "
                    "Falling back to sequential mode (arrival_delay_ms is ignored).");
        return run_json_sequential(model, tokenizer, requests);
    }

    SchedulerConfig config;
    config.max_batch_size       = max_batch_size;
    config.max_tokens_per_batch = max_tokens_per_batch;

    Scheduler         scheduler(config);
    BatchedRunner     runner(model, tokenizer);
    AsyncRequestQueue async_queue;

    LOG_INFO("Running in async mode with max_batch_size=", max_batch_size);

    // Start producer thread that submits requests with arrival delays
    RequestSubmitter submitter(requests, async_queue);
    std::thread      producer_thread = submitter.start();

    // Run consumer loop (main thread processes requests as they arrive)
    BenchmarkMetrics metrics = runner.run_async(requests, scheduler, async_queue);

    // Wait for producer to finish
    producer_thread.join();

    metrics.print();
    return build_result(metrics, model.config, "async", max_batch_size, max_tokens_per_batch, requests);
}

// ============================================================================
// JSON Benchmark Mode - Entry Point
// ============================================================================

inline BenchmarkResult run_json_benchmark(LlamaModel        &model,
                                          Tokenizer         &tokenizer,
                                          const std::string &json_path,
                                          int                max_batch_size       = 1,
                                          bool               async_mode           = false,
                                          int                max_tokens_per_batch = 512)
{
    std::vector<Request> requests = json::parse_benchmark_input(json_path);
    LOG_SUCCESS("Loaded ", requests.size(), " requests from JSON");

    BenchmarkResult result;
    if (async_mode && max_batch_size > 1) {
        result = run_json_async(model, tokenizer, requests, max_batch_size, max_tokens_per_batch);
    }
    else if (max_batch_size <= 1) {
        LOG_INFO("Running in sequential mode");
        result = run_json_sequential(model, tokenizer, requests);
    }
    else {
        result = run_json_batched(model, tokenizer, requests, max_batch_size, max_tokens_per_batch);
    }

    LOG_SUCCESS("Benchmark completed");
    return result;
}

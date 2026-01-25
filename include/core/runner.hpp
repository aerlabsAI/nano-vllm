#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "core/model.hpp"
#include "core/sampler.hpp"
#include "core/tokenizer.hpp"
#include "scheduler/batched_runner.hpp"
#include "scheduler/benchmark.hpp"
#include "scheduler/request.hpp"
#include "scheduler/request_processor.hpp"
#include "scheduler/scheduler.hpp"
#include "utils/json_parser.hpp"
#include "utils/logger.hpp"

// ============================================================================
// Single Prompt Mode
// ============================================================================

inline int run_single_prompt(LlamaModel        &model,
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

    int pos = 0;
    for (size_t i = 0; i < tokens.size() - 1; i++) {
        model.forward(tokens[i], pos);
        pos++;
    }
    int token = tokens.back();

    auto start = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < steps; s++) {
        model.forward(token, pos);
        int next_token = sampler.sample(model.state.logits.data());
        std::cout << tokenizer.decode(next_token);
        std::cout.flush();
        token = next_token;
        pos++;
        if (pos >= model.config.max_seq_len)
            break;
    }

    auto   end     = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << std::endl;
    LOG_SUCCESS("Generation completed in ", elapsed, " seconds");

    return 0;
}

// ============================================================================
// JSON Benchmark Mode - Sequential
// ============================================================================

inline int run_json_sequential(LlamaModel &model, Tokenizer &tokenizer, std::vector<Request> &requests)
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
    return 0;
}

// ============================================================================
// JSON Benchmark Mode - Batched (Continuous Batching)
// ============================================================================

inline int run_json_batched(LlamaModel &model, Tokenizer &tokenizer, std::vector<Request> &requests, int max_batch_size)
{
    SchedulerConfig config;
    config.max_batch_size = max_batch_size;

    Scheduler     scheduler(config);
    BatchedRunner runner(model, tokenizer);

    LOG_INFO("Running in batched mode with max_batch_size=", max_batch_size);

    BenchmarkMetrics metrics = runner.run_all(requests, scheduler);

    metrics.print();
    return 0;
}

// ============================================================================
// JSON Benchmark Mode - Entry Point
// ============================================================================

inline int
run_json_benchmark(LlamaModel &model, Tokenizer &tokenizer, const std::string &json_path, int max_batch_size = 1)
{
    std::vector<Request> requests;
    try {
        requests = json::parse_benchmark_input(json_path);
        LOG_SUCCESS("Loaded ", requests.size(), " requests from JSON");
    }
    catch (const std::exception &e) {
        LOG_ERROR("Failed to parse JSON: ", e.what());
        return 1;
    }

    int result;
    if (max_batch_size <= 1) {
        LOG_INFO("Running in sequential mode");
        result = run_json_sequential(model, tokenizer, requests);
    }
    else {
        result = run_json_batched(model, tokenizer, requests, max_batch_size);
    }

    LOG_SUCCESS("Benchmark completed");
    return result;
}

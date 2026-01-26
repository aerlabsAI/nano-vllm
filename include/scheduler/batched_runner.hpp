#pragma once

#include <chrono>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/model.hpp"
#include "core/sampler.hpp"
#include "core/tokenizer.hpp"
#include "scheduler/benchmark.hpp"
#include "scheduler/request.hpp"
#include "scheduler/scheduler.hpp"
#include "utils/logger.hpp"

// ============================================================================
// Batched Runner - Scheduling Simulation
//
// NOTE: This is a scheduling simulation, not true batched execution.
// Current model architecture supports single-sequence forward only.
// True continuous batching requires:
//   1. Per-request KV cache isolation
//   2. Batched forward pass with multiple sequences
//   3. Model architecture changes
//
// This implementation demonstrates the scheduler structure and
// request lifecycle management for educational purposes.
// ============================================================================

class BatchedRunner
{
public:
    BatchedRunner(LlamaModel &model, Tokenizer &tokenizer)
        : model_(model)
        , tokenizer_(tokenizer)
    {
    }

    // Run all requests with scheduling simulation
    BenchmarkMetrics run_all(std::vector<Request> &requests, Scheduler &scheduler)
    {
        BenchmarkMetrics metrics;

        LOG_WARNING("Batched mode is scheduling simulation only. "
                    "Requests are processed sequentially due to model architecture limits.");

        // Encode all prompts
        for (auto &req : requests) {
            req.prompt_tokens = tokenizer_.encode(req.prompt, true, false);
            // Pre-create sampler for each request (P1 fix)
            samplers_[req.id] = std::make_unique<Sampler>(model_.config.vocab_size,
                                                          req.sampling_params.temperature,
                                                          req.sampling_params.top_p,
                                                          static_cast<unsigned long long>(std::time(nullptr)) + req.id);
            scheduler.add_request(&req);
        }

        auto total_start = std::chrono::high_resolution_clock::now();

        // Process requests one at a time (sequential execution)
        int iteration = 0;
        while (scheduler.has_work()) {
            ScheduledBatch batch = scheduler.schedule();

            if (batch.empty()) {
                break;
            }

            LOG_INFO("Iteration ",
                     iteration,
                     ": ",
                     batch.size(),
                     " requests (",
                     (batch.is_prefill ? "prefill" : "decode"),
                     "), ",
                     batch.total_scheduled_tokens,
                     " tokens (simulated)");

            // Process requests completely (prefill + all decode)
            for (auto *req : batch.requests) {
                process_request_complete(req);
                scheduler.finish_request(req);
            }

            iteration++;
        }

        auto total_end        = std::chrono::high_resolution_clock::now();
        metrics.total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        // Collect metrics
        for (const auto &req : requests) {
            metrics.add_request(req);
        }

        // Cleanup samplers
        samplers_.clear();

        return metrics;
    }

private:
    // Process a single request completely (prefill + all decode steps)
    void process_request_complete(Request *req)
    {
        // Reset state only once per request (P0 fix)
        reset_model_state();

        // Prefill phase
        auto prefill_start = std::chrono::high_resolution_clock::now();

        int pos = 0;
        for (size_t i = 0; i < req->prompt_tokens.size() - 1; i++) {
            model_.forward(req->prompt_tokens[i], pos);
            pos++;
        }
        req->current_pos = pos;

        auto prefill_end     = std::chrono::high_resolution_clock::now();
        req->prefill_time_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();

        LOG_INFO("Request ", req->id, " prefill: ", req->num_prompt_tokens(), " tokens, ", req->prefill_time_ms, "ms");

        // Decode phase
        req->status = RequestStatus::DECODING;
        std::cout << "\n[" << req->id << "] ";

        auto decode_start = std::chrono::high_resolution_clock::now();

        int  token   = req->prompt_tokens.back();
        auto sampler = samplers_.find(req->id);

        while (req->can_generate_more()) {
            model_.forward(token, req->current_pos);

            int next_token = sampler->second->sample(model_.state.logits.data());
            req->generated_tokens.push_back(next_token);

            std::string piece = tokenizer_.decode(next_token);
            req->output_text += piece;
            std::cout << piece;
            std::cout.flush();

            token = next_token;
            req->current_pos++;

            // Check termination (P1 fix: use config instead of hardcoded 2)
            if (next_token == 2) { // TODO: model_.config.eos_token_id when available
                break;
            }
            if (req->current_pos >= model_.config.max_seq_len) {
                break;
            }
        }

        std::cout << "\n";

        auto decode_end     = std::chrono::high_resolution_clock::now();
        req->decode_time_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
        req->status         = RequestStatus::FINISHED;

        LOG_INFO("Request ", req->id, " decode: ", req->num_generated_tokens(), " tokens, ", req->decode_time_ms, "ms");
    }

    void reset_model_state()
    {
        if (model_.config.use_paged_attention) {
            model_.initialize_paged_attention();
        }
        else {
            std::fill(model_.state.key_cache.begin(), model_.state.key_cache.end(), 0.0f);
            std::fill(model_.state.value_cache.begin(), model_.state.value_cache.end(), 0.0f);
        }
    }

    LlamaModel &model_;
    Tokenizer  &tokenizer_;

    // Per-request samplers (P1 fix: avoid recreation every step)
    std::unordered_map<int, std::unique_ptr<Sampler>> samplers_;
};

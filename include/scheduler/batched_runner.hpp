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
// Batched Runner - Continuous Batching with Interleaved Execution
//
// Implements vLLM-style continuous batching on CPU:
// - Decode-first scheduling policy
// - Single-type batches (prefill OR decode, never mixed)
// - Per-request progress tracking (prefill_cursor, num_computed_tokens)
// - Interleaved multi-request stepping (not vectorized batched forward)
// ============================================================================

class BatchedRunner
{
public:
    BatchedRunner(LlamaModel &model, Tokenizer &tokenizer)
        : model_(model)
        , tokenizer_(tokenizer)
    {
    }

    BenchmarkMetrics run_all(std::vector<Request> &requests, Scheduler &scheduler)
    {
        BenchmarkMetrics metrics;

        for (auto &req : requests) {
            req.prompt_tokens = tokenizer_.encode(req.prompt, true, false);
            samplers_[req.id] = std::make_unique<Sampler>(model_.config.vocab_size,
                                                          req.sampling_params.temperature,
                                                          req.sampling_params.top_p,
                                                          static_cast<unsigned long long>(std::time(nullptr)) + req.id);
            scheduler.add_request(&req);
        }

        reset_model_state();
        auto total_start = std::chrono::high_resolution_clock::now();

        int iteration = 0;
        while (scheduler.has_work()) {
            ScheduledBatch batch = scheduler.schedule();
            if (batch.empty())
                break;

            LOG_INFO("Iteration ",
                     iteration,
                     ": ",
                     batch.size(),
                     " requests (",
                     (batch.is_prefill ? "prefill" : "decode"),
                     "), ",
                     batch.total_scheduled_tokens,
                     " tokens");

            if (batch.is_prefill) {
                run_prefill_batch(batch, scheduler);
            }
            else {
                run_decode_batch(batch, scheduler);
            }

            iteration++;
        }

        auto total_end        = std::chrono::high_resolution_clock::now();
        metrics.total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        for (const auto &req : requests) {
            metrics.add_request(req);
        }

        samplers_.clear();
        return metrics;
    }

private:
    // Process prefill phase for multiple requests (interleaved execution)
    // Each request processes scheduled_tokens[i] tokens in this iteration
    void run_prefill_batch(ScheduledBatch &batch, Scheduler &scheduler)
    {
        for (size_t i = 0; i < batch.requests.size(); i++) {
            Request *req          = batch.requests[i];
            int      tokens_to_do = batch.scheduled_tokens[i];

            auto prefill_start = std::chrono::high_resolution_clock::now();

            // Process chunk of prompt tokens (not full prompt at once)
            for (int t = 0; t < tokens_to_do; t++) {
                int token_idx = req->prefill_cursor + t;
                if (token_idx >= req->num_prompt_tokens())
                    break;

                if (model_.config.use_paged_attention) {
                    model_.forward_with_request(req->prompt_tokens[token_idx], req->current_pos, req);
                }
                else {
                    model_.forward(req->prompt_tokens[token_idx], req->current_pos);
                }
                req->current_pos++;
                req->num_computed_tokens++;
            }
            req->prefill_cursor += tokens_to_do;

            auto prefill_end = std::chrono::high_resolution_clock::now();
            req->prefill_time_ms += std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();

            // Transition to DECODING when entire prompt is processed
            if (!req->is_prefill()) {
                req->last_token = req->prompt_tokens.back();
                req->status     = RequestStatus::DECODING;
                LOG_INFO("Request ", req->id, " prefill complete: ", req->num_prompt_tokens(), " tokens");
                std::cout << "\n[" << req->id << "] ";
                std::cout.flush();
            }
        }
    }

    // Process decode phase for multiple requests (interleaved execution)
    // Each request generates exactly 1 token in this iteration
    void run_decode_batch(ScheduledBatch &batch, Scheduler &scheduler)
    {
        for (size_t i = 0; i < batch.requests.size(); i++) {
            Request *req = batch.requests[i];

            auto decode_start = std::chrono::high_resolution_clock::now();

            if (model_.config.use_paged_attention) {
                model_.forward_with_request(req->last_token, req->current_pos, req);
            }
            else {
                model_.forward(req->last_token, req->current_pos);
            }
            int next_token = samplers_[req->id]->sample(model_.state.logits.data());

            req->generated_tokens.push_back(next_token);
            req->current_pos++;
            req->num_computed_tokens++;
            req->last_token = next_token;

            std::string piece = tokenizer_.decode(next_token);
            req->output_text += piece;
            std::cout << piece;
            std::cout.flush();

            auto decode_end = std::chrono::high_resolution_clock::now();
            req->decode_time_ms += std::chrono::duration<double, std::milli>(decode_end - decode_start).count();

            // Check completion conditions
            if (next_token == 2) { // TODO: Make EOS token configurable
                req->finished_reason = FinishReason::Eos;
                finish_request(req, scheduler);
            }
            else if (!req->can_generate_more()) {
                req->finished_reason = FinishReason::MaxTokens;
                finish_request(req, scheduler);
            }
            else if (req->current_pos >= model_.config.max_seq_len) {
                req->finished_reason = FinishReason::MaxSeqLen;
                finish_request(req, scheduler);
            }
        }
    }

    void finish_request(Request *req, Scheduler &scheduler)
    {
        std::cout << "\n";
        LOG_INFO("Request ",
                 req->id,
                 " finished (",
                 finish_reason_to_string(req->finished_reason),
                 "): ",
                 req->num_generated_tokens(),
                 " tokens");

        if (model_.config.use_paged_attention && model_.block_manager != nullptr) {
            model_.block_manager->free_request(req->id);
        }

        scheduler.finish_request(req);
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

    LlamaModel                                       &model_;
    Tokenizer                                        &tokenizer_;
    std::unordered_map<int, std::unique_ptr<Sampler>> samplers_;
};

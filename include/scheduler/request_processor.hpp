#pragma once

#include <chrono>
#include <iostream>

#include "core/model.hpp"
#include "core/sampler.hpp"
#include "core/tokenizer.hpp"
#include "scheduler/benchmark.hpp"
#include "scheduler/request.hpp"
#include "utils/logger.hpp"

// ============================================================================
// Request Processor - Handles individual request execution
// ============================================================================

class RequestProcessor
{
public:
    RequestProcessor(LlamaModel &model, Tokenizer &tokenizer)
        : model_(model)
        , tokenizer_(tokenizer)
    {
    }

    void process(Request &request, bool stream_output = true)
    {
        request.prompt_tokens = tokenizer_.encode(request.prompt, true, false);
        request.status        = RequestStatus::PREFILLING;

        Sampler sampler(model_.config.vocab_size,
                        request.sampling_params.temperature,
                        request.sampling_params.top_p,
                        std::time(nullptr) + request.id);

        // Prefill phase
        auto prefill_start = std::chrono::high_resolution_clock::now();

        int pos = 0;
        for (size_t i = 0; i < request.prompt_tokens.size() - 1; i++) {
            model_.forward(request.prompt_tokens[i], pos);
            pos++;
        }

        auto prefill_end        = std::chrono::high_resolution_clock::now();
        request.prefill_time_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();

        // Decode phase
        request.status      = RequestStatus::DECODING;
        int token           = request.prompt_tokens.back();
        request.current_pos = pos;

        auto decode_start = std::chrono::high_resolution_clock::now();

        while (request.can_generate_more()) {
            model_.forward(token, request.current_pos);

            int next_token = sampler.sample(model_.state.logits.data());
            request.generated_tokens.push_back(next_token);

            std::string piece = tokenizer_.decode(next_token);
            request.output_text += piece;

            if (stream_output) {
                std::cout << piece;
                std::cout.flush();
            }

            token = next_token;
            request.current_pos++;

            if (request.current_pos >= model_.config.max_seq_len)
                break;

            // Check for EOS (token 2 for Llama models)
            if (next_token == 2)
                break;
        }

        auto decode_end        = std::chrono::high_resolution_clock::now();
        request.decode_time_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();

        request.status = RequestStatus::FINISHED;

        if (model_.config.use_paged_attention && model_.block_manager) {
            model_.block_manager->free_request(request.id);
        }
    }

    void reset_state()
    {
        if (model_.config.use_paged_attention) {
            model_.initialize_paged_attention();
        }
        else {
            std::fill(model_.state.key_cache.begin(), model_.state.key_cache.end(), 0.0f);
            std::fill(model_.state.value_cache.begin(), model_.state.value_cache.end(), 0.0f);
        }
    }

private:
    LlamaModel &model_;
    Tokenizer  &tokenizer_;
};

// ============================================================================
// Inline implementation for BenchmarkMetrics::add_request
// ============================================================================

inline void BenchmarkMetrics::add_request(const Request &request)
{
    total_requests++;
    total_prompt_tokens += request.num_prompt_tokens();
    total_generated_tokens += request.num_generated_tokens();
    total_prefill_time_ms += request.prefill_time_ms;
    total_decode_time_ms += request.decode_time_ms;
}

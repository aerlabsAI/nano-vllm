#pragma once

#include <string>
#include <vector>

// ============================================================================
// Request Status - Lifecycle states for request processing
// ============================================================================

enum class RequestStatus {
    PENDING,    // Waiting in queue
    PREFILLING, // Processing prompt tokens
    DECODING,   // Generating output tokens
    FINISHED,   // Completed successfully
    FAILED      // Failed with error
};

inline const char *request_status_to_string(RequestStatus status)
{
    switch (status) {
    case RequestStatus::PENDING:
        return "PENDING";
    case RequestStatus::PREFILLING:
        return "PREFILLING";
    case RequestStatus::DECODING:
        return "DECODING";
    case RequestStatus::FINISHED:
        return "FINISHED";
    case RequestStatus::FAILED:
        return "FAILED";
    default:
        return "UNKNOWN";
    }
}

// ============================================================================
// Sampling Parameters - Per-request generation configuration
// ============================================================================

struct SamplingParams
{
    float temperature = 1.0f;
    float top_p       = 0.9f;
    int   max_tokens  = 256;

    SamplingParams() = default;
    SamplingParams(float temp, float topp, int max_tok)
        : temperature(temp)
        , top_p(topp)
        , max_tokens(max_tok)
    {
    }
};

// ============================================================================
// Finish Reason - Why a request completed
// ============================================================================

enum class FinishReason {
    None,      // Not finished yet
    Eos,       // Hit end-of-sequence token
    MaxTokens, // Hit max_tokens limit
    MaxSeqLen, // Hit model's max sequence length
    OOM        // Out of memory (no blocks available)
};

inline const char *finish_reason_to_string(FinishReason reason)
{
    switch (reason) {
    case FinishReason::None:
        return "NONE";
    case FinishReason::Eos:
        return "EOS";
    case FinishReason::MaxTokens:
        return "MAX_TOKENS";
    case FinishReason::MaxSeqLen:
        return "MAX_SEQ_LEN";
    case FinishReason::OOM:
        return "OOM";
    default:
        return "UNKNOWN";
    }
}

// ============================================================================
// Request - Represents a single inference request
// ============================================================================

struct Request
{
    int id = -1;

    // Input
    std::string      prompt;
    std::vector<int> prompt_tokens;
    SamplingParams   sampling_params;

    // State
    RequestStatus    status              = RequestStatus::PENDING;
    int              current_pos         = 0;  // Current position in sequence (for positional encoding)
    int              num_computed_tokens = 0;  // Total tokens processed (prompt + generated)
    int              prefill_cursor      = 0;  // Progress tracker for chunked prefill
    int              last_token          = -1; // Last generated token (for decode phase)
    FinishReason     finished_reason     = FinishReason::None;
    std::vector<int> generated_tokens;

    // Memory management (for PagedAttention)
    // Per-request block tables: [n_layers][logical_blocks] -> physical_block_id
    // Each request owns its own block tables for KV cache isolation
    std::vector<std::vector<int>> block_tables;

    // Output
    std::string output_text;

    // Metrics
    double prefill_time_ms  = 0.0;
    double decode_time_ms   = 0.0;
    int    arrival_delay_ms = 0; // Delay before this request "arrives" (for async simulation)
    int    num_prompt_tokens() const { return static_cast<int>(prompt_tokens.size()); }
    int    num_generated_tokens() const { return static_cast<int>(generated_tokens.size()); }
    int    total_tokens() const { return num_prompt_tokens() + num_generated_tokens(); }

    Request() = default;
    Request(int req_id, const std::string &prompt_text, const SamplingParams &params)
        : id(req_id)
        , prompt(prompt_text)
        , sampling_params(params)
    {
    }

    bool is_finished() const { return status == RequestStatus::FINISHED || status == RequestStatus::FAILED; }

    bool can_generate_more() const { return num_generated_tokens() < sampling_params.max_tokens; }

    // Helper methods for continuous batching scheduler
    bool is_prefill() const { return prefill_cursor < num_prompt_tokens(); }
    int  remaining_prompt() const { return num_prompt_tokens() - prefill_cursor; }
    int  remaining_total() const { return total_tokens() - num_computed_tokens; }
};

// ============================================================================
// Request Batch - Collection of requests for batch processing
// ============================================================================

struct RequestBatch
{
    std::vector<Request *> requests;

    int  size() const { return static_cast<int>(requests.size()); }
    bool empty() const { return requests.empty(); }

    void add(Request *req) { requests.push_back(req); }
    void clear() { requests.clear(); }

    // Get all requests in a specific status
    std::vector<Request *> get_by_status(RequestStatus status) const
    {
        std::vector<Request *> result;
        for (auto *req : requests) {
            if (req->status == status) {
                result.push_back(req);
            }
        }
        return result;
    }
};

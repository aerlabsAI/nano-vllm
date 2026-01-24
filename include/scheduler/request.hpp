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
    RequestStatus    status      = RequestStatus::PENDING;
    int              current_pos = 0;
    std::vector<int> generated_tokens;

    // Memory management (for PagedAttention)
    std::vector<int> block_ids;

    // Output
    std::string output_text;

    // Metrics
    double prefill_time_ms = 0.0;
    double decode_time_ms  = 0.0;
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

#pragma once

#include <algorithm>
#include <queue>
#include <vector>

#include "scheduler/request.hpp"
#include "utils/logger.hpp"

// ============================================================================
// Scheduler Configuration
// ============================================================================

struct SchedulerConfig
{
    int max_batch_size       = 8;   // Maximum requests per batch
    int max_tokens_per_batch = 512; // Maximum total tokens per batch
};

// ============================================================================
// Scheduled Batch - Output of scheduler for execution
// ============================================================================

struct ScheduledBatch
{
    std::vector<Request *> prefill_requests; // Requests in prefill phase
    std::vector<Request *> decode_requests;  // Requests in decode phase

    int total_prefill_tokens() const
    {
        int total = 0;
        for (auto *req : prefill_requests) {
            total += req->num_prompt_tokens();
        }
        return total;
    }

    int total_decode_tokens() const { return static_cast<int>(decode_requests.size()); }

    int total_requests() const { return static_cast<int>(prefill_requests.size() + decode_requests.size()); }

    bool empty() const { return prefill_requests.empty() && decode_requests.empty(); }
};

// ============================================================================
// Scheduler - Manages request queue and batch formation
// ============================================================================

class Scheduler
{
public:
    explicit Scheduler(const SchedulerConfig &config = SchedulerConfig())
        : config_(config)
    {
    }

    // Add a new request to the queue
    void add_request(Request *request)
    {
        request->status = RequestStatus::PENDING;
        pending_queue_.push(request);
        LOG_INFO("Scheduler: Added request ", request->id, " to queue");
    }

    // Schedule next batch for execution
    ScheduledBatch schedule()
    {
        ScheduledBatch batch;

        // First, add decode requests (they have priority - shorter)
        for (auto *req : running_requests_) {
            if (req->status == RequestStatus::DECODING) {
                if (batch.total_requests() >= config_.max_batch_size)
                    break;
                batch.decode_requests.push_back(req);
            }
        }

        // Then, add prefill requests from pending queue
        int remaining_slots = config_.max_batch_size - batch.total_requests();
        // P2 fix: Include decode tokens in budget calculation
        int current_tokens = batch.total_prefill_tokens() + batch.total_decode_tokens();

        while (!pending_queue_.empty() && remaining_slots > 0) {
            Request *req        = pending_queue_.front();
            int      req_tokens = req->num_prompt_tokens();

            // Check token budget (prefill + decode)
            if (current_tokens + req_tokens > config_.max_tokens_per_batch) {
                break;
            }

            pending_queue_.pop();
            req->status = RequestStatus::PREFILLING;
            running_requests_.push_back(req);
            batch.prefill_requests.push_back(req);

            current_tokens += req_tokens;
            remaining_slots--;
        }

        return batch;
    }

    // Update request status after batch execution
    void update_after_prefill(Request *request) { request->status = RequestStatus::DECODING; }

    // Mark request as finished and remove from running
    void finish_request(Request *request)
    {
        request->status = RequestStatus::FINISHED;
        running_requests_.erase(std::remove(running_requests_.begin(), running_requests_.end(), request),
                                running_requests_.end());
        LOG_INFO("Scheduler: Request ", request->id, " finished");
    }

    // Check if there's more work to do
    bool has_pending() const { return !pending_queue_.empty(); }
    bool has_running() const { return !running_requests_.empty(); }
    bool has_work() const { return has_pending() || has_running(); }

    // Get counts
    int num_pending() const { return static_cast<int>(pending_queue_.size()); }
    int num_running() const { return static_cast<int>(running_requests_.size()); }

private:
    SchedulerConfig        config_;
    std::queue<Request *>  pending_queue_;
    std::vector<Request *> running_requests_;
};

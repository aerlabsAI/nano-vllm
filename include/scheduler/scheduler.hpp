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
    std::vector<Request *> requests;
    std::vector<int>       scheduled_tokens;
    bool                   is_prefill             = false;
    int                    total_scheduled_tokens = 0;

    int  size() const { return static_cast<int>(requests.size()); }
    bool empty() const { return requests.empty(); }

    void add(Request *req, int tokens)
    {
        requests.push_back(req);
        scheduled_tokens.push_back(tokens);
        total_scheduled_tokens += tokens;
    }

    void clear()
    {
        requests.clear();
        scheduled_tokens.clear();
        is_prefill             = false;
        total_scheduled_tokens = 0;
    }
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

    // Schedule next batch for execution (decode-first policy, single-type batches)
    ScheduledBatch schedule()
    {
        ScheduledBatch batch;

        // First priority: decode requests (1 token each)
        for (auto *req : running_requests_) {
            if (req->status == RequestStatus::DECODING) {
                if (batch.size() >= config_.max_batch_size)
                    break;
                if (batch.total_scheduled_tokens + 1 > config_.max_tokens_per_batch)
                    break;
                batch.add(req, 1);
            }
        }

        // If we have decode requests, return decode batch
        if (!batch.empty()) {
            batch.is_prefill = false;
            return batch;
        }

        // Second priority: prefill requests from pending queue
        while (!pending_queue_.empty()) {
            if (batch.size() >= config_.max_batch_size)
                break;

            Request *req         = pending_queue_.front();
            int      remaining   = req->remaining_prompt();
            int      budget_left = config_.max_tokens_per_batch - batch.total_scheduled_tokens;
            int      chunk_size  = std::min(remaining, budget_left);

            if (chunk_size <= 0)
                break;

            pending_queue_.pop();
            req->status = RequestStatus::PREFILLING;
            running_requests_.push_back(req);
            batch.add(req, chunk_size);
        }

        if (!batch.empty()) {
            batch.is_prefill = true;
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

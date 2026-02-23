#pragma once

#include <chrono>
#include <thread>
#include <vector>

#include "scheduler/async_request_queue.hpp"
#include "scheduler/request.hpp"
#include "utils/logger.hpp"

// ============================================================================
// Request Submitter - Producer thread for async request simulation
//
// Simulates staggered request arrivals based on arrival_delay_ms.
// Runs in a separate thread and submits requests to AsyncRequestQueue
// at their scheduled arrival times.
// ============================================================================

class RequestSubmitter
{
public:
    /**
     * @brief Construct a RequestSubmitter.
     *
     * @param requests Vector of requests with arrival_delay_ms set
     * @param queue Async queue to submit requests to
     */
    RequestSubmitter(std::vector<Request> &requests, AsyncRequestQueue &queue)
        : requests_(requests)
        , queue_(queue)
    {
    }

    /**
     * @brief Run the submission loop (blocking).
     *
     * Submits each request after its arrival_delay_ms has elapsed.
     * Call this in a separate thread.
     */
    void run()
    {
        auto start_time = std::chrono::steady_clock::now();

        for (auto &req : requests_) {
            // Wait until this request's arrival time
            auto target_time = start_time + std::chrono::milliseconds(req.arrival_delay_ms);
            std::this_thread::sleep_until(target_time);

            // Submit request to queue
            queue_.submit_request(&req);
            LOG_INFO("Request ",
                     req.id,
                     " arrived (delay=",
                     req.arrival_delay_ms,
                     "ms, prompt=\"",
                     req.prompt.substr(0, 20),
                     (req.prompt.size() > 20 ? "..." : ""),
                     "\")");
        }

        // Signal that all requests have been submitted
        queue_.mark_all_submitted();
        LOG_INFO("All ", requests_.size(), " requests submitted");
    }

    /**
     * @brief Start the submitter in a background thread.
     *
     * @return Thread handle for joining later
     */
    std::thread start()
    {
        return std::thread([this]() { run(); });
    }

private:
    std::vector<Request> &requests_;
    AsyncRequestQueue    &queue_;
};

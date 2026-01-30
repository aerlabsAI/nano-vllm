#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <vector>

#include "scheduler/request.hpp"

// ============================================================================
// Async Request Queue - Thread-safe queue for dynamic request arrivals
//
// Enables simulation of real serving scenarios where requests arrive
// dynamically while the model is processing other requests.
// ============================================================================

class AsyncRequestQueue
{
public:
    AsyncRequestQueue() = default;

    /**
     * @brief Submit a request to the queue (non-blocking).
     *
     * Called by producer thread to add a new request that has "arrived".
     *
     * @param request Pointer to request to submit
     */
    void submit_request(Request *request)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_requests_.push_back(request);
        }
        cv_.notify_one();
    }

    /**
     * @brief Get all pending requests and clear the queue.
     *
     * Called by consumer thread to retrieve newly arrived requests.
     *
     * @return Vector of pending request pointers
     */
    std::vector<Request *> get_pending()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<Request *>      result = std::move(pending_requests_);
        pending_requests_.clear();
        return result;
    }

    /**
     * @brief Wait for requests with timeout.
     *
     * Efficient waiting using condition variable. Returns when:
     * - New requests arrive
     * - All submissions are complete (is_done())
     * - Timeout expires
     *
     * @param timeout_ms Maximum wait time in milliseconds
     * @return true if woken by new requests or completion, false on timeout
     */
    bool wait_for_requests(int timeout_ms)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
            return !pending_requests_.empty() || all_submitted_;
        });
    }

    /**
     * @brief Mark all requests as submitted (producer complete).
     *
     * Called by producer thread when all requests have been submitted.
     */
    void mark_all_submitted()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            all_submitted_ = true;
        }
        cv_.notify_all();
    }

    /**
     * @brief Check if producer has finished submitting all requests.
     *
     * @return true if all requests have been submitted
     */
    bool is_done() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return all_submitted_;
    }

    /**
     * @brief Check if there are pending requests.
     *
     * @return true if queue has pending requests
     */
    bool has_pending() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return !pending_requests_.empty();
    }

    /**
     * @brief Get number of pending requests.
     *
     * @return Number of requests in queue
     */
    size_t num_pending() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return pending_requests_.size();
    }

    /**
     * @brief Reset the queue state for reuse.
     */
    void reset()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_requests_.clear();
        all_submitted_ = false;
    }

private:
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    std::vector<Request *>  pending_requests_;
    bool                    all_submitted_ = false;
};

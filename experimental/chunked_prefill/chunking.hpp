#pragma once

#include <algorithm>
#include <vector>

namespace ChunkedPrefill {

struct ChunkInfo
{
    std::vector<int> tokens;
    int              start_pos;
    int              chunk_id;
};

inline std::vector<ChunkInfo> create_chunks(const std::vector<int> &tokens, int chunk_size)
{
    std::vector<ChunkInfo> chunks;
    chunks.reserve((tokens.size() + chunk_size - 1) / chunk_size);

    for (size_t i = 0; i < tokens.size(); i += chunk_size) {
        ChunkInfo chunk;
        chunk.chunk_id  = static_cast<int>(chunks.size());
        chunk.start_pos = static_cast<int>(i);

        size_t end = std::min(i + chunk_size, tokens.size());
        chunk.tokens.assign(tokens.begin() + i, tokens.begin() + end);

        chunks.push_back(std::move(chunk));
    }

    return chunks;
}

struct PrefillMetrics
{
    double total_time_ms;
    double avg_chunk_time_ms;
    int    num_chunks;
    int    total_tokens;
    int    chunk_size;

    double tokens_per_second() const { return (total_time_ms > 0) ? (total_tokens * 1000.0 / total_time_ms) : 0.0; }
};

} // namespace ChunkedPrefill

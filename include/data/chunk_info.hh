#ifndef CHUNK_INFO_HH
#define CHUNK_INFO_HH

#include <set>
#include <algorithm>

using namespace std;

namespace neuroh5
{
  namespace data
  {

    // Constants for chunking
    constexpr size_t CHUNK_SIZE = 1ULL << 30; // 1GB chunks for safety margin

    template<typename T>
    struct ChunkInfo {
        std::vector<int> sendcounts;
        std::vector<int> sdispls;
        std::vector<int> recvcounts;
        std::vector<int> rdispls;
        size_t total_send_size;
        size_t total_recv_size;
    };

    
    template<typename T>
    ChunkInfo<T> calculate_chunk_sizes(
        const std::vector<size_t>& full_sendcounts,
        const std::vector<size_t>& full_sdispls,
        const std::vector<size_t>& full_recvcounts,
        const std::vector<size_t>& full_rdispls,
        size_t chunk_start,
        size_t chunk_size)
    {
        const size_t size = full_sendcounts.size();
        ChunkInfo<T> chunk;
        chunk.sendcounts.resize(size);
        chunk.sdispls.resize(size);
        chunk.recvcounts.resize(size);
        chunk.rdispls.resize(size);
        
        chunk.total_send_size = 0;
        chunk.total_recv_size = 0;

        for (size_t i = 0; i < size; ++i) {
            // Calculate how much data to send in this chunk
            size_t send_remaining = (chunk_start < full_sendcounts[i]) ? 
                full_sendcounts[i] - chunk_start : 0;
            chunk.sendcounts[i] = static_cast<int>(std::min(send_remaining, chunk_size));
            chunk.sdispls[i] = static_cast<int>(full_sdispls[i] + chunk_start);
            chunk.total_send_size += chunk.sendcounts[i];

            // Calculate how much data to receive in this chunk
            size_t recv_remaining = (chunk_start < full_recvcounts[i]) ?
                full_recvcounts[i] - chunk_start : 0;
            chunk.recvcounts[i] = static_cast<int>(std::min(recv_remaining, chunk_size));
            chunk.rdispls[i] = static_cast<int>(full_rdispls[i] + chunk_start);
            chunk.total_recv_size += chunk.recvcounts[i];
        }

        return chunk;
    }


  }
}
#endif

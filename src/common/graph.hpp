#pragma once
#include <cstdint>
#include <vector>

namespace dense_subgraph {
struct Graph {
    uint32_t n = 0;
    uint64_t m = 0;
    std::vector<uint16_t> groups, color_ids;
    std::vector<uint64_t> counts, offsets;
    std::vector<uint32_t> neighbors;
};
} // namespace dense_subgraph

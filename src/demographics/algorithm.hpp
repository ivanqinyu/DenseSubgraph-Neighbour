#pragma once
#include "graph_io.hpp"
#include <cstddef>
struct Ranking {
    std::vector<std::vector<uint32_t>> group;
    uint32_t densest_size;
};
struct Result {
    std::vector<uint8_t> in;
    uint64_t edges = 0;
    uint32_t size = 0;
    size_t candidates = 0;
};
// Deterministic peeling order, partitioned by demographic group.
Ranking rank_vertices(const Graph &graph);
// Search rounded population quotas over a fixed set of candidate core sizes.
Result solve(const Graph &graph, const Ranking &ranking);

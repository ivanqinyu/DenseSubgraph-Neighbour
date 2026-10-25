#pragma once
#include "../common/graph.hpp"
#include "../common/peeling_result.hpp"
#include <cstddef>
#include <string>
namespace dense_subgraph {

struct PeelingSnapshot {
    uint64_t step = 0, core_size = 0, edges = 0;
    double density = 0;
    std::vector<uint64_t> coverage, internal;
};
struct PeelingBest {
    int64_t step = -1;
    double density = -1;
    uint64_t edges = 0;
    std::vector<uint64_t> coverage, internal;
};
struct PeelingResult {
    std::vector<uint32_t> order;
    std::vector<PeelingBest> best;
    std::vector<PeelingSnapshot> snapshots;
    double elapsed = 0;
};
PeelingResult run_peeling(const Graph &g, const std::vector<double> &alphas = {0, .1, .2, .3, .4},
                          size_t max_snapshots = 1000);
} // namespace dense_subgraph

#pragma once
#include "../common/graph.hpp"
#include <string>

namespace dense_subgraph {
struct QADResult {
    std::vector<uint32_t> core;
    std::vector<uint64_t> coverage, internal;
    double density = 0, raw_density = 0, elapsed_seconds = 0;
    uint64_t edges = 0, quota_deletions = 0;
    double upper_bound = 0;
    bool feasible = false, reachability_impossible = false;
    std::string status = "no_feasible_candidate";
};

// Find a dense core satisfying integer external coverage demands.
QADResult run_quota_active(const Graph &g, const std::vector<uint64_t> &demand, uint64_t seed = 0);
} // namespace dense_subgraph

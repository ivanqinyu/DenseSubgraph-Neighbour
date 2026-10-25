#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace dense_subgraph {
struct ContinuousResult {
    std::vector<uint32_t> core;
    std::vector<uint64_t> coverage, internal;
    double density = 0, raw_density = 0, elapsed_seconds = 0;
    uint64_t edges = 0, moves = 0;
    int iterations = 0;
    bool feasible = false, reachability_impossible = false;
    std::string status = "no_feasible_candidate";
};
} // namespace dense_subgraph

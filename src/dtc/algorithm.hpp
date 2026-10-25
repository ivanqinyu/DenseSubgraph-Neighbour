#pragma once
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace dtc {
using Vertex = uint32_t;
struct Graph {
    std::vector<std::vector<Vertex>> adjacency;
    std::vector<uint32_t> colors;
};
struct Options {
    uint64_t seed = 0;
    double time_limit = 0, stop_factor = 6;
    // Optional explicit vertex order; an empty vector uses the seeded order.
    std::vector<Vertex> order;
};
struct Statistics {
    uint64_t oracle_calls = 0, completion_calls = 0, representative_search_nodes = 0;
    uint64_t recursion_nodes = 0, maximum_depth = 0, duplicate_nodes = 0, small_core_calls = 0;
    double oracle_seconds = 0, completion_seconds = 0;
    bool small_core_skipped_by_bound = false;
    std::string winning_stage;
};
struct Result {
    bool feasible = false;
    std::vector<Vertex> core;
    std::vector<uint64_t> coverage;
    uint64_t internal_edges = 0;
    double density = 0, search_seconds = 0;
    double first_feasible_seconds = std::numeric_limits<double>::quiet_NaN();
    int64_t winning_depth = -1;
    double density_upper_bound = std::numeric_limits<double>::quiet_NaN();
    double approximation_factor = std::numeric_limits<double>::quiet_NaN();
    bool target_met = false;
    std::string status = "completed", termination_reason = "exhausted";
    Statistics statistics;
};

Result solve(const Graph &graph, const std::vector<uint64_t> &demands, const Options &options);
} // namespace dtc

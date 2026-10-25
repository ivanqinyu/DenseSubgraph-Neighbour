#pragma once
#include "../common/graph.hpp"
#include <string>

namespace dense_subgraph {
// Defaults used by the command-line runner: penalty weight mu, Adam learning rate and
// number of steps.
inline constexpr double kDefaultMu = 256.0, kDefaultLearningRate = 0.02;
inline constexpr int kDefaultSteps = 1000;
struct ContinuousResult {
    std::vector<uint32_t> core;
    std::vector<uint64_t> coverage, internal;
    double density = 0;
    uint64_t edges = 0;
    bool feasible = false;
    std::string status = "no_feasible_candidate";
};

struct SimpleRun {
    ContinuousResult raw;
    double optimization_seconds = 0, rounding_seconds = 0;
    int iterations = 0;
};

// Optimize continuous membership scores, then round by exact prefix evaluation.
// Set optimize=false to evaluate the same seeded random initialization alone.
SimpleRun run_simple(const Graph &g, const std::vector<uint64_t> &d, uint64_t seed,
                     double mu = kDefaultMu, bool optimize = true,
                     double learning_rate = kDefaultLearningRate, int steps = kDefaultSteps);
} // namespace dense_subgraph

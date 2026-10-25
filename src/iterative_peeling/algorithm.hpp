#pragma once
#include "../common/graph.hpp"
#include "../common/peeling_result.hpp"
#include <cstddef>
#include <string>
namespace dense_subgraph {

struct IterativeResult {
    std::vector<ContinuousResult> best;
    std::vector<int> selected_round;
    std::vector<uint64_t> selected_step;
    int rounds_completed = 0;
    double elapsed = 0;
};
IterativeResult run_iterative_peeling(const Graph &g, int rounds = 5,
                                      const std::vector<double> &alphas = {.1, .2, .3, .4});

} // namespace dense_subgraph

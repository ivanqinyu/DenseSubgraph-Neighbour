#pragma once
#include "../common/graph.hpp"
#include "../common/peeling_result.hpp"
#include <cstddef>
#include <string>
namespace dense_subgraph {

struct BaselineFistaResult {
    std::vector<ContinuousResult> best;
    int iterations_completed = 0;
    double elapsed_seconds = 0;
};
BaselineFistaResult run_baseline_fista(const Graph &g, int iterations = 50,
                                       const std::vector<double> &alphas = {.1, .2, .3, .4});

} // namespace dense_subgraph

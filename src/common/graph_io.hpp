#pragma once
#include "graph.hpp"
#include <string>

namespace dense_subgraph {
Graph load_graph(const std::string &edge_path, const std::string &label_path, uint64_t n,
                 uint64_t m);
}

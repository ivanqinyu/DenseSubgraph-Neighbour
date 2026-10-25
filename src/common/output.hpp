#pragma once
#include "graph.hpp"
#include <ostream>

namespace dense_subgraph {
inline void write_group_counts(std::ostream &out, const Graph &g,
                               const std::vector<uint64_t> &values) {
    out << '{';
    for (size_t i = 0; i < g.counts.size(); ++i) {
        if (i)
            out << ',';
        out << '"' << g.color_ids[i] << '"' << ':' << (i < values.size() ? values[i] : 0);
    }
    out << '}';
}
} // namespace dense_subgraph

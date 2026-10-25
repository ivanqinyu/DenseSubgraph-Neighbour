#pragma once
#include "graph_io.hpp"

// Count induced edges in a byte mask (1 means selected).
uint64_t edges(const Graph &graph, const std::vector<uint8_t> &selected);
// Upper: no group may exceed max_group_share of the selected vertices.
std::vector<uint8_t> ddsp(const Graph &graph, double max_group_share);
// Lower: include at least population_percent percent of every group.
std::vector<uint8_t> dalks(const Graph &graph, int population_percent);

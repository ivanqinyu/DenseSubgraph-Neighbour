#pragma once
#include <cstdint>
#include <string>
#include <vector>
struct Graph {
    uint32_t n;
    uint64_t m = 0;
    std::vector<uint32_t> col, pop, adj;
    std::vector<uint64_t> off;
};
Graph load(const std::string &, uint64_t, uint64_t);

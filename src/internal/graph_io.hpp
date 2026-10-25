#pragma once
#include <cstdint>
#include <string>
#include <vector>
struct Graph {
    uint32_t n;
    uint64_t m;
    std::vector<uint16_t> c;
    std::vector<std::vector<uint32_t>> a;
    uint16_t C;
};
Graph load(const std::string &, uint64_t, uint64_t);

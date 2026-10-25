#pragma once
#include <cstdint>
#include <utility>
#include <vector>
using I = int64_t;
struct Iter {
    I p, q, value, total;
};
struct Result {
    I p, q;
    std::vector<int> core;
    std::vector<Iter> its;
};
// Input edges must be simple, undirected, and indexed in [0,n).
// Returns an exact density fraction and the verified flow iterations.
Result solve(int n, const std::vector<std::pair<int, int>> &edges);

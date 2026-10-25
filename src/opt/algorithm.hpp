#pragma once
#include <bitset>
#include <chrono>
#include <cstdint>
#include <vector>
constexpr int MAX_VERTICES = 1024;
using Mask = std::bitset<MAX_VERTICES>;
using Clock = std::chrono::steady_clock;
struct Solver {
    int n, groups, alpha;
    double seconds;
    std::vector<Mask> adj, groupMasks, suffix, suffixNeighbors;
    std::vector<int> demand, order;
    Mask best = 0;
    int bestEdges = 0, bestSize = 0;
    uint64_t nodes = 0, densityPrunes = 0, coveragePrunes = 0;
    bool completed = true;
    Clock::time_point start;

    bool expired() const;
    void consider(Mask chosen, Mask neighbors, int edges, int size);

    // For T subset U, edges(I union T) <= edges(I) +
    // sum_{v in T} (deg_I(v) + deg_U(v)/2). All bounds use doubled integers.
    // Selecting scores in descending order maximizes this upper bound at each
    // cardinality even with the necessary per-group inside-count limits.
    bool densityPossible(int depth, Mask chosen, int edges, int size);

    void dfs(int depth, Mask chosen, Mask neighbors, int edges, int size);

    void solve();
};

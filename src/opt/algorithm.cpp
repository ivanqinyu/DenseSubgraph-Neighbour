#include "algorithm.hpp"
#include <algorithm>
#include <numeric>
#include <random>

static int count(const Mask &x) { return int(x.count()); }
bool Solver::expired() const {
    return std::chrono::duration<double>(Clock::now() - start).count() >= seconds;
}
void Solver::consider(Mask chosen, Mask neighbors, int edges, int size) {
    if (!size || (bestSize && int64_t(edges) * bestSize <= int64_t(bestEdges) * size))
        return;
    Mask outside = neighbors & ~chosen;
    for (int c = 0; c < groups; ++c)
        if (count(outside & groupMasks[c]) < demand[c])
            return;
    best = chosen;
    bestEdges = edges;
    bestSize = size;
}
bool Solver::densityPossible(int depth, Mask chosen, int edges, int size) {
    if (!bestSize)
        return true;
    struct Score {
        int value, group;
    };
    std::vector<Score> scores;
    scores.reserve(n - depth);
    std::vector<int> caps(groups, 0);
    int total = 0;
    for (int c = 0; c < groups; ++c)
        caps[c] = count(groupMasks[c]) - demand[c] - count(chosen & groupMasks[c]);
    for (int j = depth; j < n; ++j) {
        int v = order[j], c = 0;
        while (!(groupMasks[c] & (Mask(1) << v)).any())
            ++c;
        scores.push_back({2 * count(adj[v] & chosen) + count(adj[v] & suffix[depth]), c});
        ++total;
    }
    std::sort(scores.begin(), scores.end(), [](Score a, Score b) { return a.value > b.value; });
    int bound = 2 * edges, k = 0;
    for (int j = 0; j < total; ++j) {
        int c = scores[j].group;
        if (caps[c] <= 0)
            continue;
        --caps[c];
        ++k;
        bound += scores[j].value;
        if (int64_t(bound) * bestSize > int64_t(2 * bestEdges) * (size + k))
            return true;
    }
    return false;
}
void Solver::dfs(int depth, Mask chosen, Mask neighbors, int edges, int size) {
    if (!completed)
        return;
    ++nodes;
    if ((nodes & 4095) == 1 && expired()) {
        completed = false;
        return;
    }
    Mask potential = (neighbors | suffixNeighbors[depth]) & ~chosen;
    for (int c = 0; c < groups; ++c) {
        if (count(potential & groupMasks[c]) < demand[c]) {
            ++coveragePrunes;
            return;
        }
    }
    consider(chosen, neighbors, edges, size);
    if (depth == n)
        return;
    if (!densityPossible(depth, chosen, edges, size)) {
        ++densityPrunes;
        return;
    }
    int v = order[depth];
    dfs(depth + 1, chosen | (Mask(1) << v), neighbors | adj[v], edges + count(adj[v] & chosen),
        size + 1);
    dfs(depth + 1, chosen, neighbors, edges, size);
}
void Solver::solve() {
    start = Clock::now();
    order.resize(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return count(adj[a]) > count(adj[b]); });
    // Cheap deterministic feasible incumbents improve pruning, but have no
    // effect on the certificate: only complete DFS proves optimality.
    std::mt19937 rng(20261012);
    std::vector<int> trial = order;
    for (int rep = 0; rep < 256 && !expired(); ++rep) {
        Mask chosen = 0, neighbors = 0;
        int edges = 0, size = 0;
        for (int v : trial) {
            edges += count(adj[v] & chosen);
            ++size;
            chosen |= Mask(1) << v;
            neighbors |= adj[v];
            consider(chosen, neighbors, edges, size);
        }
        std::shuffle(trial.begin(), trial.end(), rng);
    }
    suffix.assign(n + 1, 0);
    suffixNeighbors.assign(n + 1, 0);
    for (int j = n - 1; j >= 0; --j) {
        suffix[j] = suffix[j + 1] | (Mask(1) << order[j]);
        suffixNeighbors[j] = suffixNeighbors[j + 1] | adj[order[j]];
    }
    dfs(0, 0, 0, 0, 0);
}

#include "algorithm.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace dense_subgraph {
PeelingResult run_peeling(const Graph &g, const std::vector<double> &alphas, size_t max_snapshots) {
    const auto started = std::chrono::steady_clock::now();
    const uint32_t NIL = UINT32_MAX;
    const size_t k = g.counts.size();
    PeelingResult result;
    result.order.reserve(g.n);
    result.best.resize(alphas.size());
    std::vector<uint32_t> degree(g.n), next(g.n, NIL), prev(g.n, NIL);
    std::vector<uint8_t> live(g.n, 1);
    uint32_t maximum = 0;
    for (uint32_t v = 0; v < g.n; ++v) {
        degree[v] = uint32_t(g.offsets[v + 1] - g.offsets[v]);
        maximum = std::max(maximum, degree[v]);
    }
    std::vector<uint32_t> heads(size_t(maximum) + 1, NIL);
    auto insert = [&](uint32_t v, uint32_t d) {
        prev[v] = NIL;
        next[v] = heads[d];
        if (heads[d] != NIL)
            prev[heads[d]] = v;
        heads[d] = v;
    };
    auto erase = [&](uint32_t v, uint32_t d) {
        if (prev[v] != NIL)
            next[prev[v]] = next[v];
        else
            heads[d] = next[v];
        if (next[v] != NIL)
            prev[next[v]] = prev[v];
    };
    for (uint32_t v = 0; v < g.n; ++v)
        insert(v, degree[v]);
    std::vector<uint64_t> coverage(k, 0), internal = g.counts;
    std::vector<std::vector<uint64_t>> demand(alphas.size(), std::vector<uint64_t>(k));
    std::vector<uint32_t> violations(alphas.size(), 0);
    for (size_t a = 0; a < alphas.size(); ++a) {
        if (!std::isfinite(alphas[a]) || alphas[a] < 0 || alphas[a] > 1)
            throw std::runtime_error("Invalid alpha");
        for (size_t c = 0; c < k; ++c) {
            double percent = std::round(alphas[a] * 100);
            demand[a][c] = std::abs(alphas[a] * 100 - percent) < 1e-10
                               ? (g.counts[c] * uint64_t(percent) + 99) / 100
                               : uint64_t(std::ceil(alphas[a] * double(g.counts[c])));
            violations[a] += demand[a][c] > 0;
        }
    }
    auto change_coverage = [&](uint16_t c, int delta) {
        uint64_t old = coverage[c];
        coverage[c] = delta > 0 ? old + 1 : old - 1;
        for (size_t a = 0; a < alphas.size(); ++a) {
            bool before = old < demand[a][c], after = coverage[c] < demand[a][c];
            if (before && !after)
                --violations[a];
            else if (!before && after)
                ++violations[a];
        }
    };
    uint64_t edges = g.m;
    uint32_t minimum = 0;
    uint64_t stride =
        std::max<uint64_t>(1, (uint64_t(g.n) + std::max<size_t>(max_snapshots, 2) - 2) /
                                  (std::max<size_t>(max_snapshots, 2) - 1));
    auto record = [&](uint64_t step) {
        uint64_t size = uint64_t(g.n) - step;
        double rho = size ? double(edges) / double(size) : 0;
        for (size_t a = 0; a < alphas.size(); ++a)
            if (size && !violations[a] && rho > result.best[a].density) {
                auto &b = result.best[a];
                b.step = int64_t(step);
                b.density = rho;
                b.edges = edges;
                b.coverage = coverage;
                b.internal = internal;
            }
        if (step % stride == 0 || step == g.n)
            result.snapshots.push_back({step, size, edges, rho, coverage, internal});
    };
    record(0);
    for (uint64_t step = 1; step <= g.n; ++step) {
        while (minimum <= maximum && heads[minimum] == NIL)
            ++minimum;
        if (minimum > maximum)
            throw std::runtime_error("Broken degree buckets");
        uint32_t v = heads[minimum], dv = degree[v];
        erase(v, dv);
        live[v] = 0;
        result.order.push_back(v);
        edges -= dv;
        --internal[g.groups[v]];
        if (dv)
            change_coverage(g.groups[v], 1);
        for (uint64_t j = g.offsets[v]; j < g.offsets[v + 1]; ++j) {
            uint32_t u = g.neighbors[j], d = degree[u];
            if (!d)
                throw std::runtime_error("Invalid residual degree");
            if (live[u]) {
                erase(u, d);
                --degree[u];
                insert(u, d - 1);
                minimum = std::min(minimum, d - 1);
            } else {
                --degree[u];
                if (d == 1)
                    change_coverage(g.groups[u], -1);
            }
        }
        record(step);
        if (step % 10000000 == 0)
            std::cerr << "Peeling " << step << "/" << g.n << " vertices" << std::endl;
    }
    for (const auto &b : result.best)
        if (b.step >= 0)
            result.snapshots.push_back({uint64_t(b.step), uint64_t(g.n) - uint64_t(b.step), b.edges,
                                        b.density, b.coverage, b.internal});
    std::sort(result.snapshots.begin(), result.snapshots.end(),
              [](const PeelingSnapshot &a, const PeelingSnapshot &b) { return a.step < b.step; });
    result.snapshots.erase(std::unique(result.snapshots.begin(), result.snapshots.end(),
                                       [](const PeelingSnapshot &a, const PeelingSnapshot &b) {
                                           return a.step == b.step;
                                       }),
                           result.snapshots.end());
    result.elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}
} // namespace dense_subgraph

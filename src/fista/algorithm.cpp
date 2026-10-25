// Harb--Quanrud--Chekuri 2022, Algorithm 3: alpha=1/(2 Delta), beta_t=(t-1)/(t+2).
// Retain the existing feasible symmetric 0.5 initialization and external suffix adaptation.
/*
NetworkX formula-reference license, retained for attribution:
NetworkX is distributed with the 3-clause BSD license.

::

   Copyright (c) 2004-2026, NetworkX Developers
   Aric Hagberg <hagberg@lanl.gov>
   Dan Schult <dschult@colgate.edu>
   Pieter Swart <swart@lanl.gov>
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

     * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

     * Neither the name of the NetworkX Developers nor the names of its
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Independent C++17 reimplementation; no author source code is redistributed.
// Harb, Quanrud, and Chekuri, NeurIPS 2022.
// Faster and Scalable Algorithms for Densest Subgraph and Decomposition.
// Paper:
// https://papers.neurips.cc/paper_files/paper/2022/hash/ac8fbba029dadca99d6b8c3f913d3ed6-Abstract-Conference.html
// Author supplementary archive SHA256:
// 3b43dfb326b249ba9767b3b3d3cf6aa152df338779deec7d5531c86ac23a07f8 This implementation is not a
// byte-for-byte reproduction of the author program.

#include "algorithm.hpp"
#include "../common/peeling_state.hpp"
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

namespace baseline_fista_detail {
struct Allocation {
    std::vector<uint32_t> u, v;
    std::vector<uint64_t> arc_edge;
    std::vector<double> x, y, load;
    explicit Allocation(const Graph &g) : arc_edge(g.neighbors.size()), load(g.n, 0) {
        u.reserve(g.m);
        v.reserve(g.m);
        // load_graph produces sorted adjacency. Reverse positions are filled with
        // cursors, avoiding per-edge binary search and an additional reverse mapping.
        std::vector<uint64_t> cursor = g.offsets;
        for (uint32_t a = 0; a < g.n; ++a)
            for (uint64_t j = g.offsets[a]; j < g.offsets[a + 1]; ++j) {
                uint32_t b = g.neighbors[j];
                if (a < b) {
                    uint64_t e = u.size();
                    u.push_back(a);
                    v.push_back(b);
                    arc_edge[j] = e;
                    uint64_t reverse = cursor[b]++;
                    if (reverse >= g.offsets[b + 1] || g.neighbors[reverse] != a)
                        throw std::runtime_error("FISTA requires sorted symmetric CSR");
                    arc_edge[reverse] = e;
                }
            }
        if (u.size() != g.m)
            throw std::runtime_error("FISTA edge count mismatch");
        x.assign(g.m, .5);
        y = x;
    }
    void loads(const Graph &g, const std::vector<double> &allocation) {
#pragma omp parallel for schedule(static)
        for (uint64_t a = 0; a < g.n; ++a) {
            double sum = 0;
            for (uint64_t j = g.offsets[a]; j < g.offsets[a + 1]; ++j) {
                double z = allocation[arc_edge[j]];
                sum += a < g.neighbors[j] ? z : 1 - z;
            }
            load[a] = sum;
        }
    }
    void optimize(const Graph &g, int iterations) {
        uint64_t maximum = 0;
        for (uint32_t a = 0; a < g.n; ++a)
            maximum = std::max(maximum, g.offsets[a + 1] - g.offsets[a]);
        if (!maximum)
            return;
        const double rate = 1. / (2. * double(maximum));
        for (int it = 0; it < iterations; ++it) {
            loads(g, y);
            const double momentum = double(it) / (double(it) + 3.);
#pragma omp parallel for schedule(static)
            for (uint64_t e = 0; e < g.m; ++e) {
                double old = x[e], extrapolated = y[e];
                double z = std::clamp(extrapolated - rate * (load[u[e]] - load[v[e]]), 0., 1.);
                x[e] = z;
                y[e] = z + momentum * (z - old);
            }
        }
        loads(g, x); // Never peel using stale extrapolated loads.
    }
};
struct Heap {
    std::vector<uint32_t> items, position;
    const std::vector<double> &key;
    explicit Heap(const std::vector<double> &score)
        : items(score.size()), position(score.size()), key(score) {
        std::iota(items.begin(), items.end(), 0);
        std::iota(position.begin(), position.end(), 0);
        for (size_t i = items.size() / 2; i > 0; --i)
            down(i - 1);
    }
    bool before(uint32_t a, uint32_t b) const { return key[a] == key[b] ? a < b : key[a] < key[b]; }
    void exchange(size_t a, size_t b) {
        std::swap(items[a], items[b]);
        position[items[a]] = uint32_t(a);
        position[items[b]] = uint32_t(b);
    }
    void down(size_t i) {
        while (i * 2 + 1 < items.size()) {
            size_t j = i * 2 + 1;
            if (j + 1 < items.size() && before(items[j + 1], items[j]))
                ++j;
            if (!before(items[j], items[i]))
                break;
            exchange(i, j);
            i = j;
        }
    }
    uint32_t pop() {
        auto v = items.front();
        exchange(0, items.size() - 1);
        items.pop_back();
        position[v] = UINT32_MAX;
        if (!items.empty())
            down(0);
        return v;
    }
    void decrease(uint32_t v) {
        size_t i = position[v];
        while (i && before(items[i], items[(i - 1) / 2])) {
            exchange(i, (i - 1) / 2);
            i = (i - 1) / 2;
        }
    }
};
} // namespace baseline_fista_detail
BaselineFistaResult run_baseline_fista(const Graph &g, int iterations,
                                       const std::vector<double> &alphas) {
    if (iterations < 1)
        throw std::invalid_argument("FISTA iterations must be positive");
    auto started = std::chrono::steady_clock::now();
    BaselineFistaResult out;
    out.best.resize(alphas.size());
    const size_t k = g.counts.size();
    std::vector<std::vector<uint64_t>> demand(alphas.size(), std::vector<uint64_t>(k));
    std::vector<uint64_t> violations(alphas.size(), 0), coverage(k, 0),
        internal = g.counts, reachable(k, 0), best_size(alphas.size(), 0),
        best_step(alphas.size(), 0);
    for (size_t a = 0; a < alphas.size(); ++a) {
        if (!std::isfinite(alphas[a]) || alphas[a] < 0 || alphas[a] > 1)
            throw std::invalid_argument("Invalid FISTA alpha");
        for (size_t c = 0; c < k; ++c) {
            double percent = std::round(alphas[a] * 100);
            demand[a][c] = std::abs(alphas[a] * 100 - percent) < 1e-10
                               ? (g.counts[c] * uint64_t(percent) + 99) / 100
                               : uint64_t(std::ceil(alphas[a] * double(g.counts[c])));
            violations[a] += demand[a][c] > 0;
        }
        out.best[a].coverage.assign(k, 0);
        out.best[a].internal.assign(k, 0);
    }
    std::vector<uint32_t> degree(g.n), order;
    order.reserve(g.n);
    std::vector<uint8_t> in(g.n, 1);
    for (uint32_t v = 0; v < g.n; ++v) {
        degree[v] = uint32_t(g.offsets[v + 1] - g.offsets[v]);
        if (degree[v])
            ++reachable[g.groups[v]];
    }
    baseline_fista_detail::Allocation allocation(g);
    allocation.optimize(g, iterations);
    out.iterations_completed = g.m ? iterations : 0;
    // Iterations no longer need endpoint or extrapolation arrays; release before
    // heap construction to reduce peak resident memory on billion-edge inputs.
    std::vector<uint32_t>().swap(allocation.u);
    std::vector<uint32_t>().swap(allocation.v);
    std::vector<double>().swap(allocation.y);
    baseline_fista_detail::Heap heap(allocation.load);
    uint64_t edges = g.m, size = g.n;
    auto change = [&](uint16_t c, int delta) {
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
    while (size) {
        for (size_t a = 0; a < alphas.size(); ++a)
            if (!violations[a] &&
                (!best_size[a] || static_cast<unsigned __int128>(edges) * best_size[a] >
                                      static_cast<unsigned __int128>(out.best[a].edges) * size)) {
                auto &r = out.best[a];
                r.feasible = true;
                r.status = "feasible";
                r.edges = edges;
                r.density = double(edges) / size;
                r.raw_density = r.density;
                r.coverage = coverage;
                r.internal = internal;
                best_size[a] = size;
                best_step[a] = order.size();
            }
        uint32_t v = heap.pop();
        in[v] = 0;
        order.push_back(v);
        edges -= degree[v];
        --size;
        --internal[g.groups[v]];
        if (degree[v])
            change(g.groups[v], 1);
        for (uint64_t j = g.offsets[v]; j < g.offsets[v + 1]; ++j) {
            uint32_t u = g.neighbors[j];
            if (!in[u] && degree[u] == 1)
                change(g.groups[u], -1);
            --degree[u];
            if (in[u]) {
                double x = allocation.x[allocation.arc_edge[j]];
                allocation.load[u] -= u < v ? x : 1 - x;
                heap.decrease(u);
            }
        }
    }
    std::vector<uint32_t> rank(g.n);
    for (uint32_t i = 0; i < order.size(); ++i)
        rank[order[i]] = i;
    out.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    for (size_t a = 0; a < alphas.size(); ++a) {
        auto &r = out.best[a];
        r.elapsed_seconds = out.elapsed_seconds;
        r.iterations = out.iterations_completed;
        if (r.feasible) {
            r.core.reserve(best_size[a]);
            for (uint32_t v = 0; v < g.n; ++v)
                if (rank[v] >= best_step[a])
                    r.core.push_back(v);
        } else {
            if (!g.n)
                r.status = "empty_graph";
            for (size_t c = 0; c < k; ++c)
                if (reachable[c] < demand[a][c]) {
                    r.reachability_impossible = true;
                    r.status = "infeasible_reachability_bound";
                }
        }
    }
    return out;
}
} // namespace dense_subgraph

// Independent C++17 reimplementation; no author source code is redistributed.
// Greedy++: Boob et al., Flowless, WWW 2020, https://doi.org/10.1145/3366423.3380140
// Reports only the external-feasible suffix adaptation.

#include "algorithm.hpp"
#include "../common/peeling_state.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace dense_subgraph {

namespace baseline_iterative_detail {
// Indexed heap: exactly one entry per live vertex, bounded O(n) memory.
class LoadHeap {
    const std::vector<uint64_t> &load;
    const std::vector<uint32_t> &degree;
    std::vector<uint32_t> heap, position;
    bool less(uint32_t u, uint32_t v) const {
        uint64_t a = load[u] + degree[u], b = load[v] + degree[v];
        return a < b || (a == b && u < v);
    }
    void exchange(size_t a, size_t b) {
        std::swap(heap[a], heap[b]);
        position[heap[a]] = uint32_t(a);
        position[heap[b]] = uint32_t(b);
    }
    void down(size_t a) {
        for (;;) {
            size_t b = 2 * a + 1;
            if (b >= heap.size())
                return;
            if (b + 1 < heap.size() && less(heap[b + 1], heap[b]))
                ++b;
            if (!less(heap[b], heap[a]))
                return;
            exchange(a, b);
            a = b;
        }
    }

  public:
    LoadHeap(const std::vector<uint64_t> &l, const std::vector<uint32_t> &d)
        : load(l), degree(d), heap(d.size()), position(d.size()) {
        std::iota(heap.begin(), heap.end(), 0);
        std::iota(position.begin(), position.end(), 0);
        for (size_t i = heap.size() / 2; i > 0; --i)
            down(i - 1);
    }
    uint32_t pop() {
        uint32_t v = heap.front();
        exchange(0, heap.size() - 1);
        heap.pop_back();
        position[v] = UINT32_MAX;
        if (!heap.empty())
            down(0);
        return v;
    }
    void decrease(uint32_t v) {
        size_t a = position[v];
        while (a) {
            size_t b = (a - 1) / 2;
            if (!less(heap[a], heap[b]))
                break;
            exchange(a, b);
            a = b;
        }
    }
};

inline IterativeResult run(const Graph &g, int rounds, const std::vector<double> &alphas) {
    if (rounds < 1)
        throw std::invalid_argument("Iterative peeling requires positive rounds");
    if (alphas.empty())
        throw std::invalid_argument("At least one coverage target is required");
    const auto start = std::chrono::steady_clock::now();
    const size_t k = g.counts.size(), a_count = alphas.size();
    IterativeResult out;
    out.best.resize(a_count);
    out.selected_round.assign(a_count, -1);
    out.selected_step.assign(a_count, 0);
    std::vector<std::vector<uint64_t>> demand(a_count, std::vector<uint64_t>(k));
    for (size_t a = 0; a < a_count; ++a) {
        double alpha = alphas[a];
        if (!std::isfinite(alpha) || alpha < 0 || alpha > 1)
            throw std::invalid_argument("Invalid alpha");
        double percent = std::round(alpha * 100);
        for (size_t c = 0; c < k; ++c)
            demand[a][c] = std::abs(alpha * 100 - percent) < 1e-10
                               ? (g.counts[c] * uint64_t(percent) + 99) / 100
                               : uint64_t(std::ceil(alpha * double(g.counts[c])));
    }
    std::vector<uint64_t> load(g.n, 0), best_size(a_count, 0);
    for (int round = 0; round < rounds; ++round) {
        const bool select_candidates = round == rounds - 1;
        std::vector<uint32_t> degree(g.n), order;
        order.reserve(g.n);
        std::vector<uint8_t> live(g.n, 1);
        for (uint32_t v = 0; v < g.n; ++v)
            degree[v] = uint32_t(g.offsets[v + 1] - g.offsets[v]);
        LoadHeap heap(load, degree);
        std::vector<uint64_t> coverage(k, 0), internal = g.counts;
        std::vector<uint32_t> violations(a_count, 0);
        for (size_t a = 0; select_candidates && a < a_count; ++a)
            for (size_t c = 0; c < k; ++c)
                violations[a] += demand[a][c] > 0;
        auto change = [&](uint16_t c, int delta) {
            uint64_t old = coverage[c];
            coverage[c] = delta > 0 ? old + 1 : old - 1;
            for (size_t a = 0; a < a_count; ++a) {
                bool before = old < demand[a][c], after = coverage[c] < demand[a][c];
                if (before && !after)
                    --violations[a];
                else if (!before && after)
                    ++violations[a];
            }
        };
        uint64_t edges = g.m;
        for (uint64_t step = 0; step < g.n; ++step) {
            uint64_t size = uint64_t(g.n) - step;
            for (size_t a = 0; select_candidates && a < a_count; ++a)
                if (!violations[a] &&
                    (!out.best[a].feasible ||
                     static_cast<unsigned __int128>(edges) * best_size[a] >
                         static_cast<unsigned __int128>(out.best[a].edges) * size)) {
                    auto &b = out.best[a];
                    b.feasible = true;
                    b.status = "feasible";
                    b.edges = edges;
                    b.density = double(edges) / size;
                    b.raw_density = b.density;
                    b.coverage = coverage;
                    b.internal = internal;
                    best_size[a] = size;
                    out.selected_round[a] = round;
                    out.selected_step[a] = step;
                }
            uint32_t v = heap.pop(), dv = degree[v];
            load[v] += dv;
            live[v] = 0;
            order.push_back(v);
            edges -= dv;
            if (select_candidates)
                --internal[g.groups[v]];
            if (select_candidates && dv)
                change(g.groups[v], 1);
            for (uint64_t j = g.offsets[v]; j < g.offsets[v + 1]; ++j) {
                uint32_t u = g.neighbors[j];
                if (!degree[u])
                    throw std::runtime_error("Broken iterative residual degree");
                --degree[u];
                if (live[u])
                    heap.decrease(u);
                else if (select_candidates && !degree[u])
                    change(g.groups[u], -1);
            }
        }
        // Materialize at most one suffix per target from the final round, never copy a core
        // at every density improvement (which would take quadratic time).
        for (size_t a = 0; a < a_count; ++a)
            if (out.selected_round[a] == round) {
                auto &core = out.best[a].core;
                core.assign(order.begin() + out.selected_step[a], order.end());
                std::sort(core.begin(), core.end());
            }
        out.rounds_completed = round + 1;
        std::cerr << "Greedy++ round " << round + 1 << "/" << rounds << std::endl;
    }
    out.elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    for (auto &b : out.best) {
        b.iterations = out.rounds_completed;
        b.elapsed_seconds = out.elapsed;
    }
    return out;
}
} // namespace baseline_iterative_detail

IterativeResult run_iterative_peeling(const Graph &g, int rounds,
                                      const std::vector<double> &alphas) {
    return baseline_iterative_detail::run(g, rounds, alphas);
}
} // namespace dense_subgraph

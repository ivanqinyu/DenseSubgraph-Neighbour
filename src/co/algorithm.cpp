// Continuous Optimization (CO): Adam updates followed by exact prefix rounding.
#include "algorithm.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace dense_subgraph {
// Maintains exact external coverage while scanning a discrete prefix.
struct ContinuousDiscreteState {
    const Graph &g;
    const std::vector<uint64_t> &demand;
    std::vector<uint8_t> in;
    std::vector<uint32_t> t;
    std::vector<uint64_t> coverage, internal;
    std::vector<int64_t> delta;
    std::vector<uint8_t> touched_flag;
    std::vector<uint16_t> touched;
    uint64_t size = 0, edges = 0, violations = 0;
    double deficit = 0;
    ContinuousDiscreteState(const Graph &graph, const std::vector<uint64_t> &d)
        : g(graph), demand(d), in(g.n, 0), t(g.n, 0), coverage(d.size(), 0), internal(d.size(), 0),
          delta(d.size(), 0), touched_flag(d.size(), 0) {
        for (auto a : d)
            if (a) {
                deficit += 1;
                ++violations;
            }
    }
    void change(uint16_t c, int val) {
        if (!touched_flag[c]) {
            touched_flag[c] = 1;
            touched.push_back(c);
        }
        delta[c] += val;
    }
    void prepare(uint32_t v) {
        for (auto c : touched) {
            delta[c] = 0;
            touched_flag[c] = 0;
        }
        touched.clear();
        if (t[v])
            change(g.groups[v], -1);
        for (uint64_t j = g.offsets[v]; j < g.offsets[v + 1]; ++j) {
            auto u = g.neighbors[j];
            if (!in[u] && t[u] == 0)
                change(g.groups[u], 1);
        }
    }
    double next_deficit() const {
        double z = deficit;
        for (auto c : touched) {
            double den = std::max<uint64_t>(1, demand[c]);
            z += (std::max(0., double(demand[c]) - double(coverage[c]) - double(delta[c])) -
                  std::max(0., double(demand[c]) - double(coverage[c]))) /
                 den;
        }
        return std::max(0., z);
    }
    void apply(uint32_t v, double nd) {
        for (auto c : touched) {
            bool was = coverage[c] < demand[c];
            coverage[c] = uint64_t(int64_t(coverage[c]) + delta[c]);
            bool now = coverage[c] < demand[c];
            if (was && !now)
                --violations;
            else if (!was && now)
                ++violations;
        }
        edges += t[v];
        ++size;
        ++internal[g.groups[v]];
        in[v] = 1;
        for (uint64_t j = g.offsets[v]; j < g.offsets[v + 1]; ++j) {
            auto u = g.neighbors[j];
            ++t[u];
        }
        deficit = nd;
    }
    bool feasible() const { return size && violations == 0; }
    double density() const { return size ? double(edges) / size : 0; }
};

// Deterministic prefix selection.

struct UniformCandidate {
    std::shared_ptr<std::vector<uint32_t>> order;
    size_t prefix = 0;
    double density = 0, deficit = 0;
    bool feasible = false;
};
inline std::vector<uint32_t> uniform_candidate_core(const UniformCandidate &c) {
    if (!c.order)
        return {};
    return std::vector<uint32_t>(c.order->begin(), c.order->begin() + c.prefix);
}
inline double uniform_exact_deficit(const ContinuousDiscreteState &s) {
    double d = 0;
    for (size_t c = 0; c < s.demand.size(); ++c)
        if (s.coverage[c] < s.demand[c])
            d += double(s.demand[c] - s.coverage[c]) / std::max<uint64_t>(1, s.demand[c]);
    return d;
}
// Stable exact radix sorting of nonnegative finite IEEE doubles. Equal scores retain node ID order.
inline std::shared_ptr<std::vector<uint32_t>>
uniform_score_order(const std::vector<double> &scores) {
    auto order = std::make_shared<std::vector<uint32_t>>(scores.size());
    std::iota(order->begin(), order->end(), 0);
    for (double x : scores)
        if (!std::isfinite(x) || x < 0)
            throw std::invalid_argument("Scores must be finite and nonnegative");
    std::vector<uint32_t> tmp(scores.size());
    constexpr size_t bins = 65536;
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    std::vector<size_t> counts(size_t(threads) * bins);
    for (unsigned pass = 0; pass < 4; ++pass) {
        std::fill(counts.begin(), counts.end(), 0);
        auto bucket = [&](uint32_t v) {
            uint64_t bits;
            double x = scores[v] == 0 ? 0 : scores[v];
            std::memcpy(&bits, &x, 8);
            return size_t((~bits >> (16 * pass)) & 65535);
        };
#pragma omp parallel for schedule(static)
        for (int t = 0; t < threads; ++t) {
            size_t begin = scores.size() * t / threads, end = scores.size() * (t + 1) / threads;
            auto *hist = counts.data() + size_t(t) * bins;
            for (size_t i = begin; i < end; ++i)
                ++hist[bucket((*order)[i])];
        }
        size_t sum = 0;
        for (size_t b = 0; b < bins; ++b)
            for (int t = 0; t < threads; ++t) {
                auto &a = counts[size_t(t) * bins + b];
                size_t old = a;
                a = sum;
                sum += old;
            }
#pragma omp parallel for schedule(static)
        for (int t = 0; t < threads; ++t) {
            size_t begin = scores.size() * t / threads, end = scores.size() * (t + 1) / threads;
            auto *pos = counts.data() + size_t(t) * bins;
            for (size_t i = begin; i < end; ++i) {
                auto v = (*order)[i];
                tmp[pos[bucket(v)]++] = v;
            }
        }
        order->swap(tmp);
    }
    return order;
}

// A single fixed protocol for every graph size. All reductions use 32 fixed
// partitions, independent of the number of physical OpenMP workers.
constexpr int uniform_partitions = 32;
inline double uniform_clipped_sum(const std::vector<double> &x, double shift = 0) {
    std::array<double, uniform_partitions> partial{};
#pragma omp parallel for schedule(static)
    for (int p = 0; p < uniform_partitions; ++p) {
        double s = 0;
        size_t a = x.size() * p / uniform_partitions, b = x.size() * (p + 1) / uniform_partitions;
        for (size_t v = a; v < b; ++v)
            s += std::clamp(x[v] + shift, 1e-5, 1. - 1e-5);
        partial[p] = s;
    }
    return std::accumulate(partial.begin(), partial.end(), 0.);
}
inline void uniform_project(std::vector<double> &x) {
    if (x.size() < 2)
        throw std::invalid_argument("Projection requires at least two vertices");
    double shift = 0;
    if (uniform_clipped_sum(x) < 1) {
        double lo = 0, hi = std::max(1., 1. - *std::min_element(x.begin(), x.end()));
        for (int i = 0; i < 45; ++i) {
            double mid = (lo + hi) * .5;
            if (uniform_clipped_sum(x, mid) < 1)
                lo = mid;
            else
                hi = mid;
        }
        shift = hi;
    }
#pragma omp parallel for schedule(static)
    for (size_t v = 0; v < x.size(); ++v)
        x[v] = std::clamp(x[v] + shift, 1e-5, 1. - 1e-5);
}
// SplitMix64 maps each restart index to a reproducible, distinct RNG seed.
inline uint64_t uniform_start_seed(uint64_t seed, uint64_t start) {
    uint64_t z = seed + 0x9e3779b97f4a7c15ULL * (start + 1);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
// Scratch storage is reused across Adam steps; the same evaluator serves gradient tests.
struct UniformLossWorkspace {
    std::vector<double> product, ax, scratch, b, weight, local_b;
    std::array<double, uniform_partitions> local_mass{}, local_edge{};
    double rho = 0;
    UniformLossWorkspace(size_t n, size_t k)
        : product(n), ax(n), scratch(n), b(k), weight(k), local_b(uniform_partitions * k) {}
    // x must be strictly inside (0,1), as enforced by uniform_project.
    // Returns -rho + mu * sum_c max(0, (d_c-B_c)/max(1,d_c))^2.
    // scratch contains its gradient on return.
    double evaluate(const Graph &g, const std::vector<uint64_t> &demand,
                    const std::vector<double> &x, double mu) {
        const size_t n = g.n, k = g.counts.size();
#pragma omp parallel for schedule(static)
        for (size_t v = 0; v < n; ++v)
            scratch[v] = std::log1p(-x[v]);
        std::fill(local_b.begin(), local_b.end(), 0);
#pragma omp parallel for schedule(static)
        for (int part = 0; part < uniform_partitions; ++part) {
            double mass = 0, edge = 0;
            size_t lo = n * part / uniform_partitions, hi = n * (part + 1) / uniform_partitions;
            for (size_t v = lo; v < hi; ++v) {
                double sum = 0, logp = 0;
                for (uint64_t j = g.offsets[v]; j < g.offsets[v + 1]; ++j) {
                    auto u = g.neighbors[j];
                    sum += x[u];
                    logp += scratch[u];
                }
                ax[v] = sum;
                product[v] = std::exp(logp);
                mass += x[v];
                edge += .5 * x[v] * sum;
                local_b[size_t(part) * k + g.groups[v]] += (1 - x[v]) * (1 - product[v]);
            }
            local_mass[part] = mass;
            local_edge[part] = edge;
        }
        std::fill(b.begin(), b.end(), 0);
        for (int part = 0; part < uniform_partitions; ++part)
            for (size_t c = 0; c < k; ++c)
                b[c] += local_b[size_t(part) * k + c];
        double mass = std::accumulate(local_mass.begin(), local_mass.end(), 0.),
               edge = std::accumulate(local_edge.begin(), local_edge.end(), 0.);
        rho = edge / mass;
        double loss = -rho;
        for (size_t c = 0; c < k; ++c) {
            double den = std::max<uint64_t>(1, demand[c]), def = (double(demand[c]) - b[c]) / den;
            double positive = std::max(0., def);
            loss += mu * positive * positive;
            weight[c] = 2 * mu * positive / den;
        }
        // Freeze x until all gradient entries are ready. In particular, do not
        // accidentally turn this Jacobi gradient into an in-place neighbor update.
#pragma omp parallel for schedule(static)
        for (size_t v = 0; v < n; ++v) {
            double z = 0;
            for (uint64_t j = g.offsets[v]; j < g.offsets[v + 1]; ++j) {
                auto u = g.neighbors[j];
                z += weight[g.groups[u]] * (1 - x[u]) * product[u];
            }
            scratch[v] =
                -(ax[v] - rho) / mass + weight[g.groups[v]] * (1 - product[v]) - z / (1 - x[v]);
        }
        return loss;
    }
};

// Every nonempty prefix is checked exactly. Ties retain the shorter prefix.
inline UniformCandidate simple_round(const Graph &g, const std::vector<uint64_t> &d,
                                     const std::vector<double> &x) {
    auto order = uniform_score_order(x);
    ContinuousDiscreteState state(g, d);
    UniformCandidate best;
    for (size_t i = 0; i < order->size(); ++i) {
        auto v = (*order)[i];
        state.prepare(v);
        state.apply(v, state.next_deficit());
        // Use the exact integer counts for deficit ordering as well as feasibility.
        double deficit = uniform_exact_deficit(state), density = state.density();
        if (!best.order || deficit < best.deficit ||
            (deficit == best.deficit && density > best.density))
            best = {order, i + 1, density, deficit, state.feasible()};
    }
    return best;
}

inline ContinuousResult simple_capture(const Graph &g, const std::vector<uint64_t> &d,
                                       const UniformCandidate &candidate) {
    ContinuousDiscreteState state(g, d);
    auto core = uniform_candidate_core(candidate);
    for (auto v : core) {
        state.prepare(v);
        state.apply(v, state.next_deficit());
    }
    ContinuousResult result;
    result.core = std::move(core);
    result.edges = state.edges;
    result.density = state.density();
    result.coverage = state.coverage;
    result.internal = state.internal;
    result.feasible = state.feasible();
    result.status = result.feasible ? "feasible" : "no_feasible_candidate";
    return result;
}

SimpleRun run_simple(const Graph &g, const std::vector<uint64_t> &d, uint64_t seed, double mu,
                     bool optimize, double learning_rate, int steps) {
    using Clock = std::chrono::steady_clock;
    auto seconds = [](auto t) { return std::chrono::duration<double>(Clock::now() - t).count(); };
    if (!std::isfinite(mu) || mu < 0)
        throw std::invalid_argument("Invalid mu");
    if (!std::isfinite(learning_rate) || learning_rate <= 0 || steps < 0)
        throw std::invalid_argument("Invalid learning rate or step count");
    SimpleRun out;
    if (g.n < 2)
        throw std::invalid_argument("Continuous optimization requires at least two vertices");
    auto start = Clock::now();
    std::mt19937_64 rng(uniform_start_seed(seed, 0));
    std::vector<double> x(g.n), moment(g.n, 0), variance(g.n, 0);
    for (auto &v : x)
        v = .05 + .4 * double(rng() >> 11) * 0x1.0p-53;
    uniform_project(x);
    UniformLossWorkspace workspace(g.n, g.counts.size());
    for (int step = 1; optimize && step <= steps; ++step) {
        workspace.evaluate(g, d, x, mu);
        const double correction1 = 1 - std::pow(.9, step), correction2 = 1 - std::pow(.999, step);
        for (size_t v = 0; v < g.n; ++v) {
            const double grad = workspace.scratch[v];
            moment[v] = .9 * moment[v] + .1 * grad;
            variance[v] = .999 * variance[v] + .001 * grad * grad;
            x[v] -= learning_rate * (moment[v] / correction1) /
                    (std::sqrt(variance[v] / correction2) + 1e-8);
        }
        uniform_project(x);
        ++out.iterations;
    }
    out.optimization_seconds = seconds(start);
    start = Clock::now();
    auto candidate = simple_round(g, d, x);
    out.raw = simple_capture(g, d, candidate);
    out.rounding_seconds = seconds(start);
    return out;
}
} // namespace dense_subgraph

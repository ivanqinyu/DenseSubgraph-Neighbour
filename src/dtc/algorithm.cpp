// Densest-then-Complete recursive search and completion.
#include "algorithm.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dtc {
namespace {
using Wide = unsigned __int128;
class Solver {
    using Clock = std::chrono::steady_clock;
    using Mask = std::vector<uint8_t>;
    using Pair = std::pair<Vertex, Vertex>;
    struct Deadline {};
    struct Certified {};
    struct Timer {
        double &total;
        Clock::time_point start = Clock::now();
        ~Timer() { total += std::chrono::duration<double>(Clock::now() - start).count(); }
    };
    const Graph &g;
    std::vector<uint64_t> demands;
    Options options;
    size_t n;
    uint64_t d = 0, ticks = 0, best_edges = 0, upper_twice = 0;
    bool has_upper = false;
    int64_t current_depth = -1, winning_depth = -1;
    double first_feasible_seconds = std::numeric_limits<double>::quiet_NaN();
    Clock::time_point started;
    std::vector<Vertex> order, rank, best;
    Statistics stats;
    Clock::time_point deadline = Clock::time_point::max();
    std::set<std::vector<Vertex>> visited;
    void check(bool force = false) {
        if ((++ticks % 256 == 0 || force) && Clock::now() >= deadline)
            throw Deadline{};
    }
    Mask mask(const std::vector<Vertex> &core) const {
        Mask m(n, 0);
        for (auto v : core) {
            if (v >= n)
                throw std::invalid_argument("Core vertex outside graph");
            m[v] = 1;
        }
        return m;
    }
    std::vector<Vertex> vertices(const Mask &m) const {
        std::vector<Vertex> out;
        for (size_t v = 0; v < n; ++v)
            if (m[v])
                out.push_back(static_cast<Vertex>(v));
        return out;
    }
    std::pair<uint64_t, std::vector<uint64_t>> measure(const std::vector<Vertex> &core,
                                                       bool timed = true) {
        auto m = mask(core);
        Mask reached(n, 0);
        uint64_t edges = 0;
        for (auto v : core)
            for (auto u : g.adjacency[v]) {
                if (timed)
                    check();
                if (m[u])
                    ++edges;
                else
                    reached[u] = 1;
            }
        std::vector<uint64_t> coverage(demands.size(), 0);
        for (size_t v = 0; v < n; ++v) {
            if (timed)
                check();
            if (reached[v])
                ++coverage[g.colors[v]];
        }
        return {edges / 2, coverage};
    }
    bool feasible(const std::vector<uint64_t> &coverage) const {
        for (size_t c = 0; c < demands.size(); ++c)
            if (coverage[c] < demands[c])
                return false;
        return true;
    }
    void record(const std::optional<std::vector<Vertex>> &candidate, const std::string &stage) {
        if (!candidate || candidate->empty())
            return;
        auto measured = measure(*candidate);
        if (!feasible(measured.second))
            throw std::logic_error("Invalid completion core");
        if (best.empty() ||
            Wide(measured.first) * best.size() > Wide(best_edges) * candidate->size()) {
            if (best.empty())
                first_feasible_seconds =
                    std::chrono::duration<double>(Clock::now() - started).count();
            best = *candidate;
            best_edges = measured.first;
            stats.winning_stage = stage;
            winning_depth = current_depth;
        }
    }
    static unsigned bits(Wide x) {
        unsigned b = 0;
        while (x) {
            ++b;
            x >>= 1;
        }
        return b;
    }
    bool target_met() const {
        if (best.empty() || !has_upper || options.stop_factor == 0)
            return false;
        // Compare using the exact binary rational represented by the double.
        int exponent;
        double fraction = std::frexp(options.stop_factor, &exponent);
        uint64_t mantissa = static_cast<uint64_t>(std::ldexp(fraction, 53));
        Wide left = Wide(2) * best_edges * mantissa, right = Wide(upper_twice) * best.size();
        if (!right)
            return true;
        if (!left)
            return false;
        int shift = exponent - 53;
        int lb = static_cast<int>(bits(left)) + shift, rb = static_cast<int>(bits(right));
        if (lb != rb)
            return lb > rb;
        return shift >= 0 ? (left << shift) >= right : left >= (right << (-shift));
    }
    void stop_if_certified() {
        if (target_met()) {
            check(true);
            throw Certified{};
        }
    }
    struct Peel {
        std::vector<Vertex> core;
        std::vector<uint64_t> degree;
    };
    Peel peel(const std::vector<Vertex> &excluded) {
        Timer timer{stats.oracle_seconds};
        ++stats.oracle_calls;
        Mask active(n, 1);
        for (auto v : excluded)
            active[v] = 0;
        std::vector<uint64_t> degree(n, 0);
        uint64_t max_degree = 0, edges = 0;
        for (size_t v = 0; v < n; ++v) {
            check();
            if (active[v]) {
                for (auto u : g.adjacency[v]) {
                    check();
                    degree[v] += active[u];
                }
                max_degree = std::max(max_degree, degree[v]);
                edges += degree[v];
            }
        }
        edges /= 2;
        std::vector<int64_t> heads(max_degree + 1, -1), previous(n, -1), following(n, -1);
        for (auto v : order) {
            check();
            if (active[v]) {
                auto old = heads[degree[v]];
                following[v] = old;
                if (old != -1)
                    previous[old] = v;
                heads[degree[v]] = v;
            }
        }
        uint64_t count = n - excluded.size(), best_size = count, local_best_edges = edges,
                 minimum = 0, degeneracy = 0;
        size_t best_step = 0;
        std::vector<Vertex> removed;
        removed.reserve(count);
        while (count) {
            check();
            if (Wide(edges) * best_size > Wide(local_best_edges) * count) {
                best_size = count;
                local_best_edges = edges;
                best_step = removed.size();
            }
            while (heads[minimum] == -1) {
                check();
                ++minimum;
            }
            Vertex v = static_cast<Vertex>(heads[minimum]);
            heads[minimum] = following[v];
            if (following[v] != -1)
                previous[following[v]] = -1;
            active[v] = 0;
            degeneracy = std::max(degeneracy, degree[v]);
            edges -= degree[v];
            --count;
            removed.push_back(v);
            for (auto u : g.adjacency[v]) {
                check();
                if (active[u]) {
                    auto p = previous[u], f = following[u];
                    auto old_degree = degree[u];
                    if (p == -1)
                        heads[old_degree] = f;
                    else
                        following[p] = f;
                    if (f != -1)
                        previous[f] = p;
                    --degree[u];
                    auto old = heads[degree[u]];
                    previous[u] = -1;
                    following[u] = old;
                    if (old != -1)
                        previous[old] = u;
                    heads[degree[u]] = u;
                    minimum = std::min(minimum, degree[u]);
                }
            }
        }
        std::vector<Vertex> core(removed.begin() + best_step, removed.end());
        std::sort(core.begin(), core.end());
        active = mask(core);
        std::fill(degree.begin(), degree.end(), 0);
        for (auto v : core)
            for (auto u : g.adjacency[v]) {
                check();
                degree[v] += active[u];
            }
        if (excluded.empty()) {
            upper_twice = std::min(2 * degeneracy, max_degree);
            has_upper = true;
        }
        return {std::move(core), std::move(degree)};
    }
    std::vector<Pair> representatives(const Mask &core, size_t color) {
        Mask left(n, 0), right(n, 0);
        std::vector<Pair> matching;
        for (auto w : order) {
            check();
            if (core[w] || g.colors[w] != color)
                continue;
            for (auto a : g.adjacency[w]) {
                check();
                if (!right[a]) {
                    left[w] = 1;
                    right[a] = 1;
                    matching.emplace_back(w, a);
                    break;
                }
            }
            if (Wide(matching.size()) > Wide(3) * d)
                return matching;
        }
        std::set<Pair> pairs;
        for (size_t w = 0; w < n; ++w) {
            check();
            if (left[w]) {
                size_t limit =
                    static_cast<size_t>(std::min<Wide>(g.adjacency[w].size(), Wide(d) + 1));
                for (size_t i = 0; i < limit; ++i) {
                    check();
                    pairs.emplace(w, g.adjacency[w][i]);
                }
            }
        }
        for (size_t a = 0; a < n; ++a) {
            check();
            if (right[a]) {
                uint64_t retained = 0;
                for (auto w : g.adjacency[a]) {
                    check();
                    if (!core[w] && g.colors[w] == color) {
                        pairs.emplace(w, a);
                        ++retained;
                        if (Wide(retained) == Wide(2) * d + 1)
                            break;
                    }
                }
            }
        }
        std::vector<Pair> result(pairs.begin(), pairs.end());
        std::sort(result.begin(), result.end(), [&](auto a, auto b) {
            if (core[a.second] != core[b.second])
                return core[a.second] > core[b.second];
            return std::make_pair(rank[a.first], rank[a.second]) <
                   std::make_pair(rank[b.first], rank[b.second]);
        });
        return result;
    }
    void search(const std::vector<Vertex> &excluded, uint64_t depth) {
        struct RestoreDepth {
            int64_t &value;
            int64_t previous;
            ~RestoreDepth() { value = previous; }
        };
        RestoreDepth restore{current_depth, current_depth};
        current_depth = static_cast<int64_t>(depth);
        check(true);
        if (!visited.insert(excluded).second) {
            ++stats.duplicate_nodes;
            return;
        }
        ++stats.recursion_nodes;
        stats.maximum_depth = std::max(stats.maximum_depth, depth);
        auto peeled = peel(excluded);
        auto &core = peeled.core;
        if (core.empty())
            return;
        record(complete(core), "direct");
        stop_if_certified();
        std::vector<std::vector<Vertex>> by_color(demands.size());
        for (auto v : core) {
            check();
            by_color[g.colors[v]].push_back(v);
        }
        Mask pool(n, 0);
        bool has_pool = false;
        std::vector<Vertex> branch;
        for (size_t c = 0; c < demands.size(); ++c) {
            check();
            if (!demands[c])
                continue;
            auto &vs = by_color[c];
            if (Wide(vs.size()) >= Wide(8) * d) {
                std::sort(vs.begin(), vs.end(), [&](Vertex a, Vertex b) {
                    return std::make_pair(peeled.degree[a], rank[a]) <
                           std::make_pair(peeled.degree[b], rank[b]);
                });
                size_t limit = static_cast<size_t>(std::min<Wide>(vs.size(), Wide(demands[c]) + d));
                for (size_t i = 0; i < limit; ++i) {
                    check();
                    pool[vs[i]] = 1;
                    has_pool = true;
                }
            } else
                branch.insert(branch.end(), vs.begin(), vs.end());
        }
        if (has_pool) {
            std::vector<Vertex> reduced;
            for (auto v : core)
                if (!pool[v])
                    reduced.push_back(v);
            record(complete(reduced), "selective_deletion");
            stop_if_certified();
        }
        if (depth < d) {
            std::sort(branch.begin(), branch.end(),
                      [&](Vertex a, Vertex b) { return rank[a] < rank[b]; });
            for (auto v : branch) {
                auto next = excluded;
                next.insert(std::lower_bound(next.begin(), next.end(), v), v);
                search(next, depth + 1);
            }
        }
    }
    std::vector<Pair> edge_representatives() {
        Mask matched(n, 0);
        std::vector<Pair> matching;
        for (auto u : order) {
            check();
            if (matched[u])
                continue;
            for (auto v : g.adjacency[u]) {
                check();
                if (!matched[v]) {
                    matched[u] = matched[v] = 1;
                    matching.emplace_back(std::min(u, v), std::max(u, v));
                    break;
                }
            }
            if (matching.size() > d)
                return matching;
        }
        std::set<Pair> edges;
        for (size_t u = 0; u < n; ++u) {
            check();
            if (matched[u]) {
                size_t limit =
                    static_cast<size_t>(std::min<Wide>(g.adjacency[u].size(), Wide(d) + 1));
                for (size_t i = 0; i < limit; ++i) {
                    check();
                    auto v = g.adjacency[u][i];
                    edges.emplace(std::min<Vertex>(u, v), std::max<Vertex>(u, v));
                }
            }
        }
        return {edges.begin(), edges.end()};
    }

  public:
    Solver(const Graph &graph, std::vector<uint64_t> quotas, Options opts = {})
        : g(graph), demands(std::move(quotas)), options(std::move(opts)), n(g.colors.size()) {
        if (!n || n > std::numeric_limits<Vertex>::max() || g.adjacency.size() != n ||
            demands.empty())
            throw std::invalid_argument("Invalid graph dimensions");
        if (!std::isfinite(options.time_limit) || options.time_limit < 0 ||
            !std::isfinite(options.stop_factor) ||
            (options.stop_factor != 0 && options.stop_factor < 1))
            throw std::invalid_argument("Invalid time limit or stop factor");
        for (auto q : demands) {
            if (q > std::numeric_limits<uint64_t>::max() - d)
                throw std::overflow_error("Demand sum overflow");
            d += q;
        }
        for (size_t v = 0; v < n; ++v) {
            if (g.colors[v] >= demands.size())
                throw std::invalid_argument("Color outside demand vector");
            const auto &a = g.adjacency[v];
            if (!std::is_sorted(a.begin(), a.end()) ||
                std::adjacent_find(a.begin(), a.end()) != a.end())
                throw std::invalid_argument("Adjacency must be sorted and unique");
            for (auto u : a)
                if (u >= n || u == v)
                    throw std::invalid_argument("Invalid edge endpoint or self loop");
        }
        for (size_t v = 0; v < n; ++v)
            for (auto u : g.adjacency[v])
                if (!std::binary_search(g.adjacency[u].begin(), g.adjacency[u].end(),
                                        static_cast<Vertex>(v)))
                    throw std::invalid_argument("Graph must be undirected");
        order = options.order;
        if (order.empty()) {
            order.resize(n);
            std::iota(order.begin(), order.end(), 0);
            std::mt19937_64 rng(options.seed);
            std::shuffle(order.begin(), order.end(), rng);
        }
        auto sorted = order;
        std::sort(sorted.begin(), sorted.end());
        if (sorted.size() != n)
            throw std::invalid_argument("Order must be a vertex permutation");
        for (size_t i = 0; i < n; ++i)
            if (sorted[i] != i)
                throw std::invalid_argument("Order must be a vertex permutation");
        rank.resize(n);
        for (size_t i = 0; i < n; ++i)
            rank[order[i]] = static_cast<Vertex>(i);
    }
    std::optional<std::vector<Vertex>> complete(const std::vector<Vertex> &initial) {
        Timer timer{stats.completion_seconds};
        ++stats.completion_calls;
        check(true);
        auto core = mask(initial);
        auto canonical = vertices(core);
        if (!d)
            return canonical.empty() ? std::vector<Vertex>{order[0]} : canonical;
        if (!canonical.empty() && feasible(measure(canonical).second))
            return canonical;
        Mask witnesses(n, 0), anchors(n, 0), reached(n, 0);
        auto remaining = demands;
        uint64_t remaining_total = d;
        for (auto a : canonical)
            for (auto u : g.adjacency[a]) {
                check();
                reached[u] = 1;
            }
        for (bool reached_only : {true, false})
            for (auto w : order) {
                check();
                auto c = g.colors[w];
                if (!remaining[c] || core[w] || witnesses[w] || anchors[w] ||
                    (reached_only && !reached[w]))
                    continue;
                std::optional<Vertex> anchor;
                for (auto a : g.adjacency[w]) {
                    check();
                    if (core[a]) {
                        anchor = a;
                        break;
                    }
                }
                if (!anchor)
                    for (auto a : g.adjacency[w]) {
                        check();
                        if (!witnesses[a]) {
                            anchor = a;
                            break;
                        }
                    }
                if (anchor) {
                    witnesses[w] = 1;
                    anchors[*anchor] = 1;
                    --remaining[c];
                    --remaining_total;
                }
                if (!remaining_total) {
                    for (size_t v = 0; v < n; ++v)
                        core[v] |= anchors[v];
                    return vertices(core);
                }
            }
        std::vector<std::vector<Pair>> families(demands.size());
        std::vector<size_t> colors;
        for (size_t c = 0; c < demands.size(); ++c)
            if (demands[c]) {
                families[c] = representatives(core, c);
                Mask distinct(n, 0);
                uint64_t count = 0;
                for (auto p : families[c]) {
                    check();
                    if (!distinct[p.first]) {
                        distinct[p.first] = 1;
                        ++count;
                    }
                }
                if (count < demands[c])
                    return std::nullopt;
                colors.push_back(c);
            }
        std::sort(colors.begin(), colors.end(), [&](size_t a, size_t b) {
            return std::make_pair(families[a].size(), a) < std::make_pair(families[b].size(), b);
        });
        std::fill(witnesses.begin(), witnesses.end(), 0);
        std::vector<uint64_t> uses(n, 0);
        std::function<std::optional<std::vector<Vertex>>(size_t, uint64_t, size_t)> visit;
        visit = [&](size_t ci, uint64_t needed,
                    size_t first) -> std::optional<std::vector<Vertex>> {
            check();
            ++stats.representative_search_nodes;
            if (ci == colors.size() || !needed) {
                if (ci < colors.size())
                    ++ci;
                if (ci == colors.size()) {
                    auto result = core;
                    for (size_t v = 0; v < n; ++v)
                        result[v] |= (uses[v] != 0);
                    return vertices(result);
                }
                return visit(ci, demands[colors[ci]], 0);
            }
            auto &pairs = families[colors[ci]];
            for (size_t i = first; i < pairs.size(); ++i) {
                check();
                auto w = pairs[i].first, a = pairs[i].second;
                if (witnesses[w] || uses[w] || witnesses[a])
                    continue;
                witnesses[w] = 1;
                ++uses[a];
                auto result = visit(ci, needed - 1, i + 1);
                --uses[a];
                witnesses[w] = 0;
                if (result)
                    return result;
            }
            return std::nullopt;
        };
        return visit(0, demands[colors[0]], 0);
    }
    Result run() {
        stats = {};
        best.clear();
        best_edges = 0;
        visited.clear();
        has_upper = false;
        ticks = 0;
        auto start = Clock::now();
        started = start;
        deadline = Clock::time_point::max();
        current_depth = winning_depth = -1;
        first_feasible_seconds = std::numeric_limits<double>::quiet_NaN();
        if (options.time_limit > 0) {
            double max_seconds =
                std::chrono::duration<double>(Clock::time_point::max() - start).count();
            if (options.time_limit < max_seconds)
                deadline = start + std::chrono::duration_cast<Clock::duration>(
                                       std::chrono::duration<double>(options.time_limit));
        }
        Result result;
        try {
            std::vector<uint64_t> counts(demands.size(), 0);
            for (auto c : g.colors) {
                check();
                ++counts[c];
            }
            if (!feasible(counts))
                result.status = "infeasible";
            else if (!d) {
                auto p = peel({});
                record(p.core, "unconstrained");
                stop_if_certified();
            } else {
                search({}, 0);
                if (!best.empty() && Wide(2) * best_edges >= (Wide(d) + 1) * best.size())
                    stats.small_core_skipped_by_bound = true;
                else
                    for (auto edge : edge_representatives()) {
                        ++stats.small_core_calls;
                        record(complete({edge.first, edge.second}), "small_core");
                        stop_if_certified();
                    }
                if (best.empty()) {
                    record(complete({}), "feasibility_fallback");
                    stop_if_certified();
                    if (best.empty())
                        result.status = "infeasible";
                }
            }
            check(true);
        } catch (const Certified &) {
            result.termination_reason = "certified";
        } catch (const Deadline &) {
            result.status = "timeout";
        }
        if (result.status != "completed")
            result.termination_reason = result.status;
        result.search_seconds = std::chrono::duration<double>(Clock::now() - start).count();
        result.statistics = stats;
        result.first_feasible_seconds = first_feasible_seconds;
        result.winning_depth = winning_depth;
        if (has_upper)
            result.density_upper_bound = double(upper_twice) / 2;
        if (!best.empty()) {
            result.feasible = true;
            result.core = best;
            std::sort(result.core.begin(), result.core.end());
            result.internal_edges = best_edges;
            result.coverage = measure(best, false).second;
            result.density = double(best_edges) / best.size();
            if (has_upper) {
                result.density_upper_bound = double(upper_twice) / 2;
                if (best_edges)
                    result.approximation_factor =
                        std::max(1.0, double(upper_twice) * best.size() / (2.0 * best_edges));
                else if (!upper_twice)
                    result.approximation_factor = 1;
                result.target_met = target_met();
            }
        }
        deadline = Clock::time_point::max();
        return result;
    }
};

} // namespace

Result solve(const Graph &graph, const std::vector<uint64_t> &demands, const Options &options) {
    Solver solver(graph, demands, options);
    return solver.run();
}
} // namespace dtc

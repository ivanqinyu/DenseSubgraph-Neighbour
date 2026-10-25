// QAD (Quota-Aware Deletion).
#include "algorithm.hpp"
#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace dense_subgraph {
// Maintains exact external coverage during the deletion chain.
struct QADState {
    const Graph &g;
    const std::vector<uint64_t> &demand;
    std::vector<uint8_t> in;
    std::vector<uint32_t> t;
    std::vector<uint64_t> coverage, internal;
    std::vector<int64_t> delta;
    std::vector<uint8_t> touched_flag;
    std::vector<uint16_t> touched;
    uint64_t size = 0, edges = 0, violations = 0;
    QADState(const Graph &graph, const std::vector<uint64_t> &d)
        : g(graph), demand(d), in(g.n, 0), t(g.n, 0), coverage(d.size(), 0), internal(d.size(), 0),
          delta(d.size(), 0), touched_flag(d.size(), 0) {
        for (auto a : d)
            if (a) {
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
        if (!in[v]) {
            if (t[v])
                change(g.groups[v], -1);
            for (uint64_t j = g.offsets[v]; j < g.offsets[v + 1]; ++j) {
                auto u = g.neighbors[j];
                if (!in[u] && t[u] == 0)
                    change(g.groups[u], 1);
            }
        } else {
            if (t[v])
                change(g.groups[v], 1);
            for (uint64_t j = g.offsets[v]; j < g.offsets[v + 1]; ++j) {
                auto u = g.neighbors[j];
                if (!in[u] && t[u] == 1)
                    change(g.groups[u], -1);
            }
        }
    }
    void apply(uint32_t v) {
        bool add = !in[v];
        for (auto c : touched) {
            bool was = coverage[c] < demand[c];
            coverage[c] = uint64_t(int64_t(coverage[c]) + delta[c]);
            bool now = coverage[c] < demand[c];
            if (was && !now)
                --violations;
            else if (!was && now)
                ++violations;
        }
        if (add) {
            edges += t[v];
            ++size;
            ++internal[g.groups[v]];
        } else {
            edges -= t[v];
            --size;
            --internal[g.groups[v]];
        }
        in[v] = add;
        for (uint64_t j = g.offsets[v]; j < g.offsets[v + 1]; ++j) {
            auto u = g.neighbors[j];
            if (add)
                ++t[u];
            else
                --t[u];
        }
    }
    bool feasible() const { return size && violations == 0; }
    double density() const { return size ? double(edges) / size : 0; }
};

// QAD deletion chain

// Call only after prepare(v). Only touched groups need to be updated.
inline uint64_t next_shortfall(const QADState &s, uint64_t total) {
    for (auto c : s.touched) {
        const uint64_t before = s.coverage[c];
        const uint64_t after = uint64_t(int64_t(before) + s.delta[c]);
        if (before < s.demand[c])
            total -= s.demand[c] - before;
        if (after < s.demand[c])
            total += s.demand[c] - after;
    }
    return total;
}
struct QuotaPass {
    std::vector<uint32_t> core;
    uint64_t deletions = 0, seed_edges = 0, seed_shortfall = 0;
};

// Semantics-identical quota chain with sparse unique-witness loss caches.
inline QuotaPass quota_active_deletion_chain(const Graph &g, const std::vector<uint64_t> &demand,
                                             uint64_t seed = 0) {
    QuotaPass out;
    QADState s(g, demand);
    s.in.assign(g.n, 1);
    s.size = g.n;
    s.edges = g.m;
    s.internal = g.counts;
    uint64_t shortfall = std::accumulate(demand.begin(), demand.end(), uint64_t(0));
    using Key = std::tuple<uint32_t, uint64_t, uint32_t>;
    using Heap = std::priority_queue<Key, std::vector<Key>, std::greater<Key>>;
    Heap global;
    std::vector<Heap> groups(demand.size());
    std::vector<uint64_t> tie(g.n);
    std::mt19937_64 rng(seed);
    for (uint32_t v = 0; v < g.n; ++v) {
        s.t[v] = uint32_t(g.offsets[v + 1] - g.offsets[v]);
        tie[v] = rng();
        Key key{s.t[v], tie[v], v};
        global.push(key);
        groups[g.groups[v]].push(key);
    }
    // A witness is an outside vertex with exactly one surviving core neighbor.
    // During a deletion chain each outside vertex acquires an owner at most
    // once, then loses it only when that owner is deleted. Thus total witness
    // discovery scans are O(m), and the number of stored loss counts is O(n).
    const uint32_t no_owner = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> owner(g.n, no_owner);
    std::vector<std::vector<std::pair<uint16_t, uint32_t>>> loss(g.n);
    // Own-group witness losses only increase while a vertex remains inside.
    // Once one exists, deletion cannot strictly improve total shortfall.
    std::vector<uint8_t> no_gain(g.n, 0), blocked(g.n, 0);
    std::vector<uint64_t> generation(g.n, 0);
    using WaitingKey = std::tuple<uint64_t, uint64_t, uint32_t>;
    using WaitingHeap =
        std::priority_queue<WaitingKey, std::vector<WaitingKey>, std::greater<WaitingKey>>;
    std::vector<WaitingHeap> waiting(demand.size());
    std::vector<uint8_t> improving_blocked(g.n, 0);
    std::vector<uint64_t> improving_generation(g.n, 0);
    std::vector<WaitingHeap> improving_waiting(demand.size());
    auto assign_owner = [&](uint32_t witness) {
        for (auto j = g.offsets[witness]; j < g.offsets[witness + 1]; ++j) {
            auto u = g.neighbors[j];
            if (!s.in[u])
                continue;
            owner[witness] = u;
            auto color = g.groups[witness];
            for (auto &entry : loss[u])
                if (entry.first == color) {
                    ++entry.second;
                    return;
                }
            loss[u].push_back({color, 1});
            return;
        }
        throw std::logic_error("Unique witness has no surviving owner");
    };
    auto cached_next_shortfall = [&](uint32_t v) {
        uint64_t total = shortfall;
        bool own_group_seen = false;
        const auto own_group = g.groups[v];
        const bool own_gain = s.t[v] > 0;
        for (const auto &entry : loss[v]) {
            auto c = entry.first;
            uint64_t before = s.coverage[c];
            int64_t delta = -int64_t(entry.second);
            if (c == own_group) {
                own_group_seen = true;
                delta += own_gain;
            }
            uint64_t after = uint64_t(int64_t(before) + delta);
            if (before < demand[c])
                total -= demand[c] - before;
            if (after < demand[c])
                total += demand[c] - after;
        }
        if (own_gain && !own_group_seen && s.coverage[own_group] < demand[own_group])
            --total;
        return total;
    };
    std::vector<uint32_t> removed;
    size_t best_step = 0;
    bool found = s.feasible();
    uint64_t best_edges = s.edges, best_size = s.size;
    auto observe = [&]() {
        if (s.feasible() && (!found || (unsigned __int128)s.edges * best_size >
                                           (unsigned __int128)best_edges * s.size)) {
            found = true;
            best_step = removed.size();
            best_edges = s.edges;
            best_size = s.size;
        }
    };
    observe();
    // Searches the entire heap until its minimum admissible vertex is found.
    // Selected and rejected valid entries are restored; stale ones discarded.
    auto take = [&](Heap &heap, bool improving, Key &answer) {
        std::vector<Key> deferred;
        bool exists = false;
        while (!heap.empty()) {
            Key key = heap.top();
            heap.pop();
            auto v = std::get<2>(key);
            if (!s.in[v] || std::get<0>(key) != s.t[v])
                continue;
            // Active degree never increases during deletion. At zero, removing
            // v cannot gain external coverage, so it can never reduce shortfall.
            // Keep it in the global heap for neutral deletion, not this group heap.
            if (improving) {
                if (improving_blocked[v])
                    continue;
                if (!s.t[v])
                    no_gain[v] = 1;
                if (!no_gain[v])
                    for (const auto &entry : loss[v])
                        if (entry.first == g.groups[v] && entry.second) {
                            no_gain[v] = 1;
                            break;
                        }
                if (no_gain[v])
                    continue;
                for (const auto &entry : loss[v]) {
                    const auto c = entry.first;
                    if (c == g.groups[v])
                        continue;
                    const uint64_t slack =
                        s.coverage[c] > demand[c] ? s.coverage[c] - demand[c] : 0;
                    if (entry.second > slack) {
                        improving_blocked[v] = 1;
                        ++improving_generation[v];
                        improving_waiting[c].push(
                            {demand[c] + entry.second, improving_generation[v], v});
                        break;
                    }
                }
                if (improving_blocked[v])
                    continue;
            }
            if (!shortfall) {
                if (blocked[v])
                    continue;
                // Remember one violated coverage group. Until its slack rises
                // to this threshold the vertex cannot become admissible;
                // other groups and decreasing induced degree cannot fix it.
                for (const auto &entry : loss[v]) {
                    auto c = entry.first;
                    uint64_t bonus = (c == g.groups[v] && s.t[v] > 0) ? 1 : 0;
                    uint64_t threshold = entry.second - bonus;
                    if (threshold > s.coverage[c] - demand[c]) {
                        blocked[v] = 1;
                        ++generation[v];
                        waiting[c].push({threshold, generation[v], v});
                        break;
                    }
                }
                if (blocked[v])
                    continue;
            }
            deferred.push_back(key);
            uint64_t next = cached_next_shortfall(v);
            if (improving ? next < shortfall : next == shortfall) {
                answer = key;
                exists = true;
                break;
            }
        }
        for (const auto &key : deferred)
            heap.push(key);
        return exists;
    };
    while (s.size > 1) {
        Key chosen{};
        bool exists = false;
        if (shortfall)
            for (size_t c = 0; c < demand.size(); ++c)
                if (s.coverage[c] < demand[c]) {
                    Key candidate{};
                    if (take(groups[c], true, candidate) && (!exists || candidate < chosen)) {
                        chosen = candidate;
                        exists = true;
                    }
                }
        if (!exists)
            exists = take(global, false, chosen);
        if (!exists)
            break;
        auto v = std::get<2>(chosen);
        uint64_t cached = cached_next_shortfall(v);
        s.prepare(v);
        shortfall = next_shortfall(s, shortfall);
        if (cached != shortfall)
            throw std::logic_error("Quota witness cache differs from exact coverage delta");
        s.apply(v);
        loss[v].clear();
        removed.push_back(v);
        ++out.deletions;
        for (auto j = g.offsets[v]; j < g.offsets[v + 1]; ++j) {
            auto u = g.neighbors[j];
            if (s.in[u]) {
                Key key{s.t[u], tie[u], u};
                if (!blocked[u])
                    global.push(key);
                if (!no_gain[u] && !improving_blocked[u])
                    groups[g.groups[u]].push(key);
            } else if (!s.t[u])
                owner[u] = no_owner;
            else if (s.t[u] == 1 && owner[u] == no_owner)
                assign_owner(u);
        }
        if (s.t[v] == 1)
            assign_owner(v);
        for (auto c : s.touched)
            if (s.delta[c] > 0) {
                auto &heap = improving_waiting[c];
                while (!heap.empty()) {
                    auto entry = heap.top();
                    auto u = std::get<2>(entry);
                    if (!s.in[u] || !improving_blocked[u] ||
                        std::get<1>(entry) != improving_generation[u]) {
                        heap.pop();
                        continue;
                    }
                    if (std::get<0>(entry) > s.coverage[c])
                        break;
                    heap.pop();
                    improving_blocked[u] = 0;
                    if (!no_gain[u])
                        groups[g.groups[u]].push({s.t[u], tie[u], u});
                }
            }
        if (!shortfall)
            for (auto c : s.touched)
                if (s.delta[c] > 0) {
                    auto &heap = waiting[c];
                    uint64_t slack = s.coverage[c] - demand[c];
                    while (!heap.empty()) {
                        auto entry = heap.top();
                        auto u = std::get<2>(entry);
                        if (!s.in[u] || !blocked[u] || std::get<1>(entry) != generation[u]) {
                            heap.pop();
                            continue;
                        }
                        if (std::get<0>(entry) > slack)
                            break;
                        heap.pop();
                        blocked[u] = 0;
                        global.push({s.t[u], tie[u], u});
                    }
                }
        observe();
    }
    if (found) {
        std::vector<uint8_t> excluded(g.n, 0);
        for (size_t i = 0; i < best_step; ++i)
            excluded[removed[i]] = 1;
        for (uint32_t v = 0; v < g.n; ++v)
            if (!excluded[v])
                out.core.push_back(v);
        out.seed_edges = best_edges;
        out.seed_shortfall = 0;
    } else {
        for (uint32_t v = 0; v < g.n; ++v)
            if (s.in[v])
                out.core.push_back(v);
        out.seed_edges = s.edges;
        out.seed_shortfall = shortfall;
    }
    return out;
}

QADResult run_quota_active(const Graph &g, const std::vector<uint64_t> &demand, uint64_t seed) {
    if (demand.size() != g.counts.size())
        throw std::invalid_argument("Invalid QAD demand vector");
    auto start = std::chrono::steady_clock::now();
    QADResult out;
    out.coverage.assign(demand.size(), 0);
    out.internal.assign(demand.size(), 0);
    uint64_t required = 0, maxdegree = 0;
    std::vector<uint64_t> reachable(demand.size(), 0);
    for (uint32_t v = 0; v < g.n; ++v) {
        uint64_t degree = g.offsets[v + 1] - g.offsets[v];
        maxdegree = std::max(maxdegree, degree);
        if (degree)
            ++reachable[g.groups[v]];
    }
    for (size_t c = 0; c < demand.size(); ++c) {
        if (demand[c] > reachable[c]) {
            out.reachability_impossible = true;
            out.status = "infeasible_reachability_bound";
            return out;
        }
        required += demand[c];
    }
    if (!g.n || required >= g.n) {
        out.status = "infeasible_cardinality_bound";
        return out;
    }
    out.upper_bound = std::min(double(maxdegree) / 2, double(g.n - required - 1) / 2);
    std::mt19937_64 rng(seed);
    {
        auto chain = quota_active_deletion_chain(g, demand, rng());
        out.quota_deletions += chain.deletions;
        if (!chain.seed_shortfall && !chain.core.empty())
            out.raw_density =
                std::max(out.raw_density, double(chain.seed_edges) / chain.core.size());
        QADState state(g, demand);
        for (auto v : chain.core) {
            state.prepare(v);
            state.apply(v);
        }
        QADResult r;
        r.feasible = state.feasible();
        r.edges = state.edges;
        r.density = state.density();
        r.coverage = std::move(state.coverage);
        r.internal = std::move(state.internal);
        r.core = std::move(chain.core);
        if (r.feasible && (!out.feasible || (unsigned __int128)r.edges * out.core.size() >
                                                (unsigned __int128)out.edges * r.core.size())) {
            out.core = std::move(r.core);
            out.coverage = std::move(r.coverage);
            out.internal = std::move(r.internal);
            out.feasible = true;
            out.edges = r.edges;
            out.density = r.density;
            out.status = "feasible";
        }
    }
    out.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return out;
}
} // namespace dense_subgraph

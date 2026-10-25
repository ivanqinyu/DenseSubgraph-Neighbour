#pragma once
#include "graph.hpp"
#include "peeling_result.hpp"
#include <algorithm>
namespace dense_subgraph {
// Exact induced-edge and external-coverage updates for discrete vertex moves.
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
        deficit = nd;
    }
    bool feasible() const { return size && violations == 0; }
    double density() const { return size ? double(edges) / size : 0; }
};
inline ContinuousResult measure(const Graph &g, const std::vector<uint64_t> &d,
                                const std::vector<uint32_t> &core) {
    ContinuousDiscreteState st(g, d);
    for (auto v : core) {
        st.prepare(v);
        st.apply(v, 0);
    }
    ContinuousResult r;
    r.core = core;
    r.edges = st.edges;
    r.density = st.size ? double(st.edges) / st.size : 0;
    r.coverage = st.coverage;
    r.internal = st.internal;
    r.feasible = st.feasible();
    return r;
}
} // namespace dense_subgraph

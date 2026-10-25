#include "algorithm.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

Ranking rank_vertices(const Graph &g) {
    const uint32_t nil = UINT32_MAX;
    std::vector<uint32_t> d(g.n), next(g.n, nil), prev(g.n, nil);
    uint32_t md = 0;
    for (uint32_t u = 0; u < g.n; u++) {
        uint64_t degree = g.off[u + 1] - g.off[u];
        if (degree >= UINT32_MAX)
            throw std::runtime_error("degree overflow");
        d[u] = degree;
        md = std::max(md, d[u]);
    }
    std::vector<uint32_t> head(size_t(md) + 1, nil);
    std::vector<uint8_t> live(g.n, 1);
    std::vector<uint32_t> order;
    order.reserve(g.n);
    auto add = [&](uint32_t u) {
        auto h = head[d[u]];
        prev[u] = nil;
        next[u] = h;
        if (h != nil)
            prev[h] = u;
        head[d[u]] = u;
    };
    auto del = [&](uint32_t u) {
        if (prev[u] != nil)
            next[prev[u]] = next[u];
        else
            head[d[u]] = next[u];
        if (next[u] != nil)
            prev[next[u]] = prev[u];
    };
    for (uint64_t u = g.n; u-- > 0;)
        add(u);
    uint32_t cur = 0, bestsize = g.n;
    uint64_t edges = g.m, bestedges = g.m;
    for (uint32_t step = 0; step < g.n; step++) {
        while (cur <= md && head[cur] == nil)
            cur++;
        if (cur > md)
            throw std::runtime_error("empty degree buckets");
        uint32_t u = head[cur];
        del(u);
        live[u] = 0;
        order.push_back(u);
        edges -= d[u];
        for (uint64_t j = g.off[u]; j < g.off[u + 1]; j++) {
            auto v = g.adj[j];
            if (live[v]) {
                del(v);
                d[v]--;
                add(v);
                cur = std::min(cur, d[v]);
            }
        }
        uint32_t size = g.n - step - 1;
        if (size && static_cast<long double>(edges) * bestsize >
                        static_cast<long double>(bestedges) * size) {
            bestedges = edges;
            bestsize = size;
        }
    }
    Ranking r;
    r.group.resize(g.pop.size());
    r.densest_size = bestsize;
    for (auto it = order.rbegin(); it != order.rend(); ++it)
        r.group[g.col[*it]].push_back(*it);
    return r;
}
std::vector<uint32_t> quotas(const Graph &g, uint32_t k) {
    std::vector<uint32_t> q(g.pop.size()), idx(g.pop.size());
    std::vector<uint64_t> rem(g.pop.size());
    uint64_t sum = 0;
    for (size_t c = 0; c < q.size(); c++) {
        uint64_t x = uint64_t(k) * g.pop[c];
        q[c] = x / g.n;
        rem[c] = x % g.n;
        sum += q[c];
        idx[c] = c;
    }
    std::sort(idx.begin(), idx.end(),
              [&](auto a, auto b) { return rem[a] != rem[b] ? rem[a] > rem[b] : a < b; });
    for (uint64_t j = 0; j < k - sum; j++)
        q[idx[j]]++;
    return q;
}
Result solve(const Graph &g, const Ranking &r) {
    std::set<uint32_t> sizes;
    for (uint32_t k = 1; k <= std::min(g.n, uint32_t(64)); k++)
        sizes.insert(k);
    for (int j = 0; j < 128; j++)
        sizes.insert(std::max(
            uint32_t(1), std::min(g.n, uint32_t(std::llround(std::exp(std::log(double(g.n)) * j / 127))))));
    sizes.insert(g.n);
    sizes.insert(r.densest_size);
    std::vector<uint8_t> in(g.n);
    std::vector<uint32_t> counts(g.pop.size()), bestcounts;
    uint64_t edges = 0, bestedges = 0;
    uint32_t bestsize = 0;
    for (auto k : sizes) {
        auto q = quotas(g, k);
        for (size_t c = 0; c < q.size(); c++)
            while (counts[c] > q[c]) {
                auto u = r.group[c][--counts[c]];
                in[u] = 0;
                for (uint64_t j = g.off[u]; j < g.off[u + 1]; j++)
                    if (in[g.adj[j]])
                        edges--;
            }
        for (size_t c = 0; c < q.size(); c++)
            while (counts[c] < q[c]) {
                auto u = r.group[c][counts[c]++];
                for (uint64_t j = g.off[u]; j < g.off[u + 1]; j++)
                    if (in[g.adj[j]])
                        edges++;
                in[u] = 1;
            }
        if (!bestsize ||
            static_cast<long double>(edges) * bestsize > static_cast<long double>(bestedges) * k) {
            bestsize = k;
            bestedges = edges;
            bestcounts = q;
        }
    }
    std::fill(in.begin(), in.end(), 0);
    for (size_t c = 0; c < bestcounts.size(); c++)
        for (uint32_t j = 0; j < bestcounts[c]; j++)
            in[r.group[c][j]] = 1;
    return {std::move(in), bestedges, bestsize, sizes.size()};
}

#include "algorithm.hpp"
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_set>

uint64_t edges(const Graph &g, const std::vector<uint8_t> &in) {
    uint64_t e = 0;
    for (uint32_t u = 0; u < g.n; u++)
        if (in[u])
            for (auto v : g.a[u])
                if (v > u && in[v])
                    e++;
    return e;
}
std::vector<uint8_t> densest(const Graph &g) {
    uint64_t live = g.n;
    std::vector<uint8_t> in(g.n, 1), best = in;
    std::vector<int> d(g.n);
    for (uint32_t i = 0; i < g.n; i++)
        d[i] = g.a[i].size();
    double bd = g.n ? double(edges(g, in)) / g.n : 0;
    std::priority_queue<std::pair<int, uint32_t>, std::vector<std::pair<int, uint32_t>>,
                        std::greater<std::pair<int, uint32_t>>>
        q;
    for (uint32_t i = 0; i < g.n; i++)
        q.push({d[i], i});
    uint64_t ee = edges(g, in);
    for (uint32_t step = 0; step < g.n && q.size(); step++) {
        auto [du, u] = q.top();
        q.pop();
        if (!in[u] || du != d[u])
            continue;
        in[u] = 0;
        --live;
        ee -= du;
        if (!in.empty() && live > 0) {
            double x = double(ee) / live;
            if (x > bd) {
                bd = x;
                best = in;
            }
        }
        for (auto v : g.a[u])
            if (in[v]) {
                d[v]--;
                q.push({d[v], v});
            }
    }
    return best;
}
std::vector<uint8_t> ddsp(const Graph &g, double alpha) {
    auto in = densest(g);
    std::vector<int> cnt(g.C), deg(g.n);
    int sz = 0;
    for (uint32_t u = 0; u < g.n; u++) {
        if (in[u]) {
            cnt[g.c[u]]++;
            sz++;
        }
        for (auto v : g.a[u])
            if (in[v])
                deg[u]++;
    }
    for (;;) {
        if (!sz)
            break;
        int mx = *std::max_element(cnt.begin(), cnt.end());
        if (double(mx) / sz <= alpha + 1e-12)
            break;
        int col = std::max_element(cnt.begin(), cnt.end()) - cnt.begin();
        int rem = -1, best = 1e9;
        for (uint32_t u = 0; u < g.n; u++)
            if (in[u] && g.c[u] == col) {
                int k = deg[u];
                if (k < best)
                    best = k, rem = u;
            }
        if (rem < 0)
            break;
        in[rem] = 0;
        cnt[col]--;
        sz--;
        for (auto v : g.a[rem])
            deg[v]--;
    }
    for (;;) {
        if (!sz)
            break;
        int mx = *std::max_element(cnt.begin(), cnt.end());
        int col = -1;
        for (int c = 0; c < g.C; c++)
            if (cnt[c]<mx &&double(mx) / (sz + 1)> alpha) {
                if (col < 0 || cnt[c] < cnt[col])
                    col = c;
            }
        if (col < 0)
            break;
        int add = -1, bd = -1;
        for (uint32_t u = 0; u < g.n; u++)
            if (!in[u] && g.c[u] == col) {
                int k = deg[u];
                if (k > bd)
                    bd = k, add = u;
            }
        if (add < 0)
            break;
        in[add] = 1;
        cnt[col]++;
        sz++;
        for (auto v : g.a[add])
            deg[v]++;
    }
    return in;
}
// Degree ordering is exact when the conservative floating error guard passes;
// otherwise retain the original full vertex scan and comparisons.
std::vector<uint8_t> dalks(const Graph &g, int pct) {
    auto in = densest(g);
    std::vector<int> cnt(g.C), deg(g.n), q(g.C), pop(g.C);
    int sz = 0;
    for (uint32_t u = 0; u < g.n; u++) {
        pop[g.c[u]]++;
        if (in[u]) {
            cnt[g.c[u]]++;
            sz++;
        }
        for (auto v : g.a[u])
            if (in[v])
                deg[u]++;
    }
    for (int c = 0; c < g.C; c++)
        q[c] = std::max(1, int((uint64_t(pop[c]) * pct + 99) / 100));
    using Item = std::pair<int, uint32_t>;
    using Heap = std::priority_queue<Item, std::vector<Item>, std::greater<Item>>;
    std::vector<Heap> outside(g.C);
    for (uint32_t u = 0; u < g.n; u++)
        if (!in[u])
            outside[g.c[u]].push({-deg[u], u});
    uint64_t e = edges(g, in);
    for (int guard = 0; guard < 2 * (int)g.n; guard++) {
        int col = -1;
        for (int c = 0; c < g.C; c++)
            if (cnt[c] < q[c] && (col < 0 || cnt[c] < q[c]))
                col = c;
        if (col < 0)
            break;
        auto &h = outside[col];
        while (!h.empty() && (in[h.top().second] || -h.top().first != deg[h.top().second]))
            h.pop();
        if (h.empty())
            break;
        auto add = h.top().second;
        h.pop();
        e += deg[add];
        in[add] = 1;
        cnt[col]++;
        sz++;
        for (auto v : g.a[add]) {
            deg[v]++;
            if (!in[v])
                outside[g.c[v]].push({-deg[v], v});
        }
    }
    outside.clear();
    outside.shrink_to_fit();
    Heap removable;
    for (uint32_t u = 0; u < g.n; u++)
        if (in[u] && cnt[g.c[u]] > q[g.c[u]])
            removable.push({deg[u], u});
    for (;;) {
        double cur = sz ? double(e) / sz : 0;
        int rem = -1;
        const bool ordered =
            sz > 1 && 16 * 2.2204460492503131e-16 * double(e) + 1e-12 * (sz - 1) < .25;
        if (ordered) {
            while (!removable.empty()) {
                auto [d, u] = removable.top();
                if (!in[u] || d != deg[u] || cnt[g.c[u]] <= q[g.c[u]])
                    removable.pop();
                else
                    break;
            }
            if (!removable.empty()) {
                auto [d, u] = removable.top();
                double nd = double(e - d) / (sz - 1);
                if (nd > cur + 1e-12)
                    rem = u;
            }
        } else {
            double gain = 0;
            for (uint32_t u = 0; u < g.n; u++)
                if (in[u] && cnt[g.c[u]] > q[g.c[u]]) {
                    int k = deg[u];
                    double nd = double(e - k) / (sz - 1);
                    if (nd > cur + gain + 1e-12)
                        gain = nd - cur, rem = u;
                }
        }
        if (rem < 0)
            break;
        e -= deg[rem];
        in[rem] = 0;
        cnt[g.c[rem]]--;
        sz--;
        for (auto v : g.a[rem]) {
            deg[v]--;
            if (in[v] && cnt[g.c[v]] > q[g.c[v]])
                removable.push({deg[v], v});
        }
    }
    return in;
}

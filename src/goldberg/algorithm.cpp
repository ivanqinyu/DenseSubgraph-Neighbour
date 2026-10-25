#include "algorithm.hpp"
#include <algorithm>
#include <deque>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

using W = __int128_t;
static void require(bool ok, const char *msg) {
    if (!ok)
        throw std::runtime_error(msg);
}
static I narrow(W x) {
    require(x >= 0 && x <= std::numeric_limits<I>::max() / 2, "64-bit capacity bound exceeded");
    return (I)x;
}
struct Arc {
    I cap;
    int to, next;
};
struct Flow {
    int n;
    I q, p;
    const std::vector<std::pair<int, int>> &edges;
    std::vector<Arc> a;
    std::vector<int> head, cur, h;
    std::vector<I> degree, ex, sourceflow, sinkflow;
    std::vector<unsigned char> queued;
    std::deque<int> work;
    uint64_t scans = 0, limit;
    Flow(int nv, const std::vector<std::pair<int, int>> &es, I pp, I qq)
        : n(nv), q(qq), p(pp), edges(es), head(n, -1), cur(n), h(n), degree(n), ex(n),
          sourceflow(n), sinkflow(n), queued(n) {
        require(es.size() < size_t(std::numeric_limits<int>::max() / 2),
                "Too many arcs for 32-bit indices");
        narrow(W(2) * q * es.size());
        narrow(W(2) * p);
        narrow(W(2) * q);
        a.reserve(es.size() * 2);
        for (auto [u, v] : es) {
            int k = (int)a.size();
            a.push_back({q, v, head[u]});
            head[u] = k;
            a.push_back({q, u, head[v]});
            head[v] = k + 1;
            ++degree[u];
            ++degree[v];
        }
        for (int v = 0; v < n; ++v)
            ex[v] = sourceflow[v] = narrow(W(q) * degree[v]);
        limit = std::max<uint64_t>(1, 4 * a.size() + uint64_t(n));
    }
    void activate(int v) {
        if (ex[v] > 0 && !queued[v]) {
            queued[v] = 1;
            work.push_back(v);
        }
    }
    void global() {
        std::fill(h.begin(), h.end(), n + 1);
        std::vector<int> bfs;
        bfs.reserve(n);
        for (int v = 0; v < n; ++v)
            if (sinkflow[v] < 2 * p) {
                h[v] = 1;
                bfs.push_back(v);
            }
        for (size_t k = 0; k < bfs.size(); ++k) {
            int v = bfs[k];
            for (int e = head[v]; e != -1; e = a[e].next) {
                int u = a[e].to;
                if (a[e ^ 1].cap > 0 && h[u] == n + 1) {
                    h[u] = h[v] + 1;
                    bfs.push_back(u);
                }
            }
        }
        cur = head;
        scans = 0;
    }
    void discharge(int v) {
        while (ex[v] > 0) {
            if (h[v] == 1 && sinkflow[v] < 2 * p) {
                I d = std::min(ex[v], 2 * p - sinkflow[v]);
                ex[v] -= d;
                sinkflow[v] += d;
                if (!ex[v])
                    break;
            }
            if (h[v] == n + 1 && sourceflow[v] > 0) {
                I d = std::min(ex[v], sourceflow[v]);
                ex[v] -= d;
                sourceflow[v] -= d;
                if (!ex[v])
                    break;
            }
            int &e = cur[v];
            while (e != -1) {
                ++scans;
                int u = a[e].to;
                if (a[e].cap > 0 && h[v] == h[u] + 1) {
                    I d = std::min(ex[v], a[e].cap);
                    a[e].cap -= d;
                    a[e ^ 1].cap += d;
                    ex[v] -= d;
                    ex[u] += d;
                    activate(u);
                    if (!ex[v])
                        break;
                }
                e = a[e].next;
            }
            if (!ex[v])
                break;
            int best = 2 * n + 2;
            if (sinkflow[v] < 2 * p)
                best = 0;
            if (sourceflow[v] > 0)
                best = std::min(best, n);
            for (int j = head[v]; j != -1; j = a[j].next) {
                ++scans;
                if (a[j].cap > 0)
                    best = std::min(best, h[a[j].to]);
            }
            require(best < 2 * n + 2, "Active vertex without residual exit");
            h[v] = best + 1;
            cur[v] = head[v];
        }
    }
    I solve() {
        global();
        for (int v = 0; v < n; ++v)
            activate(v);
        while (!work.empty()) {
            int v = work.front();
            work.pop_front();
            queued[v] = 0;
            discharge(v);
            if (scans > limit)
                global();
        }
        // Independently check capacity constraints and conservation, without trusting
        // the push/relabel excess array or any solver success flag.
        std::vector<W> balance(n);
        W value = 0, supplied = 0;
        for (int v = 0; v < n; ++v) {
            require(sourceflow[v] >= 0 && W(sourceflow[v]) <= W(q) * degree[v],
                    "Source capacity violation");
            require(sinkflow[v] >= 0 && sinkflow[v] <= 2 * p, "Sink capacity violation");
            balance[v] = W(sourceflow[v]) - sinkflow[v];
            value += sinkflow[v];
            supplied += sourceflow[v];
        }
        for (size_t k = 0; k < edges.size(); ++k) {
            auto [u, v] = edges[k];
            I c = a[2 * k].cap, r = a[2 * k + 1].cap;
            require(c >= 0 && r >= 0 && W(c) + r == W(2) * q, "Edge residual violation");
            W f = W(q) - c;
            balance[u] -= f;
            balance[v] += f;
        }
        for (W b : balance) {
            require(b == 0, "Flow conservation violation");
        }
        require(value == supplied, "Flow value mismatch");
        return narrow(value);
    }
    std::vector<int> cut() {
        std::vector<unsigned char> seen(n);
        std::vector<int> bfs;
        for (int v = 0; v < n; ++v)
            if (W(sourceflow[v]) < W(q) * degree[v]) {
                seen[v] = 1;
                bfs.push_back(v);
            }
        for (size_t k = 0; k < bfs.size(); ++k) {
            int v = bfs[k];
            require(sinkflow[v] == 2 * p, "Augmenting path remains");
            for (int e = head[v]; e != -1; e = a[e].next)
                if (a[e].cap > 0 && !seen[a[e].to]) {
                    seen[a[e].to] = 1;
                    bfs.push_back(a[e].to);
                }
        }
        std::sort(bfs.begin(), bfs.end());
        return bfs;
    }
};

Result solve(int n, const std::vector<std::pair<int, int>> &edges) {
    I m = edges.size();
    I p = m, q = n;
    std::vector<int> core(n);
    std::iota(core.begin(), core.end(), 0);
    std::vector<Iter> its;
    if (!m) {
        p = 0;
        q = 1;
        core = {0};
    } else
        while (true) {
            I g = std::gcd(p, q);
            p /= g;
            q /= g;
            I total = narrow(W(2) * q * m);
            Flow f(n, edges, p, q);
            I value = f.solve();
            require(value <= total, "Flow exceeds source capacities");
            its.push_back({p, q, value, total});
            std::cerr << "iteration " << its.size() << ": " << p << "/" << q << ", flow " << value
                      << "/" << total << "\n";
            if (value == total)
                break;
            core = f.cut();
            require(!core.empty(), "Empty improving cut");
            std::vector<unsigned char> chosen(n);
            for (int v : core)
                chosen[v] = 1;
            I inside = 0;
            for (auto [u, v] : edges)
                inside += chosen[u] && chosen[v];
            require(W(inside) * q > W(p) * core.size(), "Cut does not improve density");
            require(W(total) + W(2) * p * core.size() - W(2) * q * inside == value,
                    "Cut/flow equality failed");
            p = inside;
            q = (I)core.size();
        }
    std::vector<unsigned char> chosen(n);
    for (int v : core)
        chosen[v] = 1;
    I inside = 0;
    for (auto [u, v] : edges)
        inside += chosen[u] && chosen[v];
    require(W(inside) * q == W(p) * core.size(), "Core fails exact lower bound");
    return {p, q, std::move(core), std::move(its)};
}

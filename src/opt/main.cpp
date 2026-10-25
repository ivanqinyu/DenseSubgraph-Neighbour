#include "algorithm.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

static int count(const Mask &x) { return int(x.count()); }
int main(int argc, char **argv) {
    try {
        if (argc < 3 || argc > 4)
            throw std::runtime_error("usage: exact INPUT SECONDS [ALPHA_PERCENT=10]");
        Solver s;
        s.seconds = std::stod(argv[2]);
        s.alpha = argc == 4 ? std::stoi(argv[3]) : 10;
        if (!std::isfinite(s.seconds) || s.seconds <= 0 || s.alpha < 0 || s.alpha > 100)
            throw std::runtime_error(
                "seconds must be positive and finite; alpha must be an integer in [0,100]");
        std::ifstream in(argv[1]);
        int m;
        if (!(in >> s.n >> m >> s.groups) || s.n < 1 || s.n > MAX_VERTICES || m < 0 ||
            s.groups < 1 || s.groups > MAX_VERTICES)
            throw std::runtime_error(
                "invalid header (requires 1<=n<=" + std::to_string(MAX_VERTICES) +
                ", 1<=g<=" + std::to_string(MAX_VERTICES) + ")");
        s.adj.assign(s.n, 0);
        s.groupMasks.assign(s.groups, 0);
        s.demand.resize(s.groups);
        for (int v = 0; v < s.n; ++v) {
            int c;
            if (!(in >> c) || c < 0 || c >= s.groups)
                throw std::runtime_error("invalid group ID");
            s.groupMasks[c] |= Mask(1) << v;
        }
        for (int e = 0; e < m; ++e) {
            int a, b;
            if (!(in >> a >> b) || a < 0 || b < 0 || a >= s.n || b >= s.n || a == b ||
                (s.adj[a] & (Mask(1) << b)).any())
                throw std::runtime_error("invalid, duplicate, or self-loop edge");
            s.adj[a] |= Mask(1) << b;
            s.adj[b] |= Mask(1) << a;
        }
        std::string extra;
        if (in >> extra)
            throw std::runtime_error("unexpected trailing input");
        for (int c = 0; c < s.groups; ++c)
            s.demand[c] = (s.alpha * count(s.groupMasks[c]) + 99) / 100;
        s.solve();
        double elapsed = std::chrono::duration<double>(Clock::now() - s.start).count();
        std::cout << std::setprecision(17) << "{\"alpha_percent\":" << s.alpha
                  << ",\"completed\":" << (s.completed ? "true" : "false")
                  << ",\"feasible\":" << (s.bestSize ? "true" : "false") << ",\"status\":\""
                  << (s.completed ? (s.bestSize ? "optimal" : "infeasible")
                                  : (s.bestSize ? "incumbent" : "unknown"))
                  << "\",\"edges\":" << s.bestEdges << ",\"size\":" << s.bestSize
                  << ",\"density\":";
        if (s.bestSize)
            std::cout << double(s.bestEdges) / s.bestSize;
        else
            std::cout << "null";
        std::cout << ",\"core\":[";
        bool first = true;
        for (int v = 0; v < s.n; ++v)
            if ((s.best & (Mask(1) << v)).any()) {
                if (!first)
                    std::cout << ',';
                first = false;
                std::cout << v;
            }
        std::cout << "],\"nodes\":" << s.nodes << ",\"density_prunes\":" << s.densityPrunes
                  << ",\"coverage_prunes\":" << s.coveragePrunes
                  << ",\"elapsed_seconds\":" << elapsed << "}\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}

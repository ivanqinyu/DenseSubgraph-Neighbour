#include "algorithm.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

template <class T> void array(std::ostream &o, const std::vector<T> &v) {
    o << '[';
    for (size_t j = 0; j < v.size(); j++) {
        if (j)
            o << ',';
        o << uint64_t(v[j]);
    }
    o << ']';
}
int main(int argc, char **argv) {
    try {
        if (argc != 5) {
            std::cerr << "usage: algorithm INPUT_DIR N M OUTPUT_DIR\n";
            return 2;
        }
        auto start = std::chrono::steady_clock::now();
        auto g = load(argv[1], std::stoull(argv[2]), std::stoull(argv[3]));
        auto ranking = rank_vertices(g);
        double prep =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::filesystem::create_directories(argv[4]);
        std::ofstream out(std::string(argv[4]) + "/records.jsonl");
        if (!out)
            throw std::runtime_error("cannot write records");
        out << std::setprecision(17);
        auto t = std::chrono::steady_clock::now();
        auto res = solve(g, ranking);
        std::vector<uint32_t> cnt(g.pop.size()), ext(g.pop.size());
        std::vector<uint8_t> covered(g.n);
        for (uint32_t u = 0; u < g.n; u++)
            if (res.in[u]) {
                cnt[g.col[u]]++;
                for (uint64_t j = g.off[u]; j < g.off[u + 1]; j++)
                    if (!res.in[g.adj[j]])
                        covered[g.adj[j]] = 1;
            }
        for (uint32_t u = 0; u < g.n; u++)
            if (covered[u])
                ext[g.col[u]]++;
        bool eq = true, rounded = true;
        double maxerr = 0, minrate = 1;
        for (size_t c = 0; c < cnt.size(); c++) {
            uint64_t x = uint64_t(res.size) * g.pop[c];
            eq = eq && (uint64_t(cnt[c]) * g.n == x);
            rounded = rounded && (cnt[c] >= x / g.n && cnt[c] <= (x + g.n - 1) / g.n);
            maxerr =
                std::max(maxerr, std::abs(double(cnt[c]) / res.size - double(g.pop[c]) / g.n));
            if (g.pop[c])
                minrate = std::min(minrate, double(ext[c]) / g.pop[c]);
        }
        if (!res.size || !rounded)
            throw std::runtime_error("internal invariant failed");
        const std::string corefile = "core.bin";
        std::ofstream core(std::string(argv[4]) + "/" + corefile, std::ios::binary);
        core.write((const char *)res.in.data(), res.in.size());
        if (!core)
            throw std::runtime_error("cannot write core");
        out << "{\"method\":\"InternalDemographic\",\"n\":" << g.n << ",\"m\":" << g.m
            << ",\"groups\":"
            << std::count_if(g.pop.begin(), g.pop.end(), [](auto p) { return p > 0; })
            << ",\"internal_feasible\":true,\"feasible\":true,\"exact_parity\":"
            << (eq ? "true" : "false") << ",\"max_proportion_error\":" << maxerr
            << ",\"core_size\":" << res.size << ",\"internal_edges\":" << res.edges
            << ",\"density\":" << double(res.edges) / res.size
            << ",\"candidate_count\":" << res.candidates << ",\"core_file\":\"" << corefile
            << "\",\"group_counts\":";
        array(out, g.pop);
        out << ",\"internal_counts\":";
        array(out, cnt);
        out << ",\"external_counts\":";
        array(out, ext);
        out << ",\"min_external_rate\":" << minrate << ",\"preparation_seconds\":" << prep
            << ",\"seconds\":"
            << prep +
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count()
            << "}\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}

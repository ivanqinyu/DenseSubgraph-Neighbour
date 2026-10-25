#include "algorithm.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

std::vector<int> targets(const std::string &s) {
    std::vector<int> r;
    size_t p = 0;
    while (p < s.size()) {
        size_t q = s.find(',', p);
        r.push_back(std::stoi(s.substr(p, q == std::string::npos ? s.size() - p : q - p)));
        if (q == std::string::npos)
            break;
        p = q + 1;
    }
    return r;
}
template <class T> void json_array(std::ostream &out, const std::vector<T> &a) {
    out << '[';
    for (size_t i = 0; i < a.size(); ++i) {
        if (i)
            out << ',';
        out << a[i];
    }
    out << ']';
}

int main(int argc, char **argv) {
    try {
        if (argc < 6) {
            std::cerr << "usage: algorithm INPUT_DIR N M OUTPUT_DIR MODE [SEED] [TARGETS]\n";
            return 2;
        }
        const std::string mode = argv[5], outdir = argv[4];
        const bool isdiv = mode == "ddsp" || mode == "internal_div";
        if (!isdiv && mode != "dalks" && mode != "internal_quota")
            throw std::runtime_error("mode must be ddsp or dalks");
        const auto ts = targets(argc > 7 ? argv[7] : "10,20,30,40");
        for (int t : ts)
            if (t < 1 || t > 100)
                throw std::runtime_error("targets must be integer percentages in [1,100]");
        const auto g = load(argv[1], std::stoull(argv[2]), std::stoull(argv[3]));
        std::filesystem::create_directories(outdir);
        std::ofstream out(outdir + "/records.jsonl");
        if (!out)
            throw std::runtime_error("cannot write records");
        out << std::setprecision(17);
        std::vector<uint64_t> pop(g.C);
        for (auto c : g.c)
            ++pop[c];
        for (int t : ts) {
            auto start = std::chrono::steady_clock::now();
            auto in = isdiv ? ddsp(g, t / 100.0) : dalks(g, t);
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const uint64_t size = std::count(in.begin(), in.end(), 1), e = edges(g, in);
            std::vector<uint64_t> cnt(g.C), ext(g.C);
            double max_share = 0;
            for (uint32_t u = 0; u < g.n; ++u) {
                if (in[u])
                    ++cnt[g.c[u]];
                else
                    for (auto v : g.a[u])
                        if (in[v]) {
                            ++ext[g.c[u]];
                            break;
                        }
            }
            for (auto count : cnt)
                max_share = std::max(max_share, size ? double(count) / size : 0.0);
            bool native_ok = size > 0 && e > 0;
            if (isdiv)
                native_ok = native_ok && max_share <= t / 100.0 + 1e-12;
            else
                for (size_t c = 0; c < pop.size(); ++c)
                    if (cnt[c] < std::max(uint64_t(1), (pop[c] * t + 99) / 100))
                        native_ok = false;
            const std::string core_file = "core_" + std::to_string(t) + ".bin";
            std::ofstream mask(outdir + "/" + core_file, std::ios::binary);
            mask.write(reinterpret_cast<const char *>(in.data()), in.size());
            if (!mask)
                throw std::runtime_error("cannot write core mask");
            out << "{\"method\":\"" << mode << "\",\"target\":" << t << ",\"parameter_name\":\""
                << (isdiv ? "beta" : "tau") << "\",\"parameter_value\":" << t / 100.0
                << ",\"alpha\":" << t / 100.0 << ",\"parameter_semantics\":\""
                << (isdiv ? "maximum_internal_group_share"
                          : "minimum_internal_group_population_fraction")
                << "\",\"n\":" << g.n << ",\"m\":" << g.m << ",\"groups\":" << g.C
                << ",\"internal_feasible\":" << (native_ok ? "true" : "false")
                << ",\"feasible\":" << (native_ok ? "true" : "false")
                << ",\"feasibility_semantics\":\"native_internal_constraint\""
                << ",\"status\":\"" << (native_ok ? "feasible" : "not_found")
                << "\",\"core_size\":" << size << ",\"internal_edges\":" << e
                << ",\"density\":" << (size ? double(e) / size : 0.0)
                << ",\"max_color_fraction\":" << max_share << ",\"core_file\":\"" << core_file
                << "\",\"group_counts\":";
            json_array(out, pop);
            out << ",\"internal_counts\":";
            json_array(out, cnt);
            out << ",\"external_counts\":";
            json_array(out, ext);
            out << ",\"seconds\":" << seconds << "}\n";
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR " << e.what() << '\n';
        return 1;
    }
}

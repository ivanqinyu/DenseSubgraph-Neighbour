#include "algorithm.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../common/arguments.hpp"
#include "../common/graph_io.hpp"
#include "../common/output.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
using namespace dense_subgraph;
namespace fs = std::filesystem;
int main(int argc, char **argv) {
    try {
        if (argc < 6 || argc > 11) {
            std::cerr << "algorithm INPUT_DIR N M OUTPUT_DIR METHOD [STARTS=1] [SEED=0] [MU=10] "
                         "[MAX_SWEEPS=0 for quota / 30 for discrete] [TARGETS=10,20,30,40]\n";
            return 2;
        }
        const std::map<std::string, std::string> names = {
            {"unconstrained_minimum_degree_peeling", "Unconstrained Minimum-Degree Peeling"},
            {"constrained_minimum_degree_peeling", "Constrained Minimum-Degree Peeling"},
        };
        const std::string method = argv[5];
        if (!names.count(method))
            throw std::invalid_argument("Unknown algorithm: " + method);
        const bool p0 = method == "unconstrained_minimum_degree_peeling";
        const bool peeling = p0 || method == "constrained_minimum_degree_peeling";
        const int starts = argc > 6 ? std::stoi(argv[6]) : 5;
        const uint64_t seed = argc > 7 ? std::stoull(argv[7]) : 0;
        const double mu = argc > 8 ? std::stod(argv[8]) : 10;
        const int sweeps = argc > 9 ? std::stoi(argv[9]) : 30;
        if (starts < 1 || sweeps < 0 || !std::isfinite(mu) || mu < 0)
            throw std::invalid_argument("Invalid algorithm parameters");
        fs::path input = argv[1], output_dir = argv[4];
        fs::create_directories(output_dir);
#ifdef _OPENMP
        omp_set_num_threads(32);
#endif
        auto now = []() { return std::chrono::steady_clock::now(); };
        auto elapsed = [&](auto start) {
            return std::chrono::duration<double>(now() - start).count();
        };
        auto loading = now();
        auto g = load_graph((input / "edges.npy").string(), (input / "labels.npy").string(),
                            std::stoull(argv[2]), std::stoull(argv[3]));
        const double load_seconds = elapsed(loading);
        std::ofstream out(output_dir / "records.jsonl");
        if (!out)
            throw std::runtime_error("Cannot write records.jsonl");
        out << std::setprecision(17);
        const auto targets = parse_targets(argc > 10 ? argv[10] : (p0 ? "0" : "10,20,30,40"), p0);
        for (int percent : targets) {
            std::vector<uint64_t> demand;
            for (auto count : g.counts)
                demand.push_back((count * percent + 99) / 100);
            auto start = now();
            ContinuousResult result;
            auto run = run_peeling(g, {percent / 100.}, 2);
            const auto &best = run.best[0];
            result.feasible = best.step >= 0;
            result.status = result.feasible ? "feasible" : "no_feasible_solution_found";
            if (result.feasible) {
                result.core.assign(run.order.begin() + best.step, run.order.end());
                result.coverage = best.coverage;
                result.internal = best.internal;
                result.edges = best.edges;
                result.density = best.density;
            }
            const double seconds = elapsed(start);
            const std::string filename = method + "-" + std::to_string(percent) + ".core.bin";
            std::ofstream core(output_dir / filename, std::ios::binary);
            if (result.feasible)
                core.write(reinterpret_cast<const char *>(result.core.data()),
                           result.core.size() * sizeof(uint32_t));
            core.close();
            if (!core)
                throw std::runtime_error("Cannot write core file");
            out << "{\"method\":\"" << method << "\",\"algorithm_name\":\"" << names.at(method)
                << "\",\"alpha\":" << percent / 100. << ",\"seed\":" << (peeling ? 0 : seed)
                << ",\"n\":" << g.n << ",\"m\":" << g.m << ",\"groups\":" << g.counts.size()
                << ",\"feasible\":" << (result.feasible ? "true" : "false") << ",\"status\":\""
                << result.status << "\",\"core_file\":\"" << filename
                << "\",\"core_size\":" << (result.feasible ? result.core.size() : 0)
                << ",\"internal_edges\":" << (result.feasible ? result.edges : 0)
                << ",\"density\":";
            if (result.feasible)
                out << result.density;
            else
                out << "null";
            out << ",\"seconds\":" << seconds << ",\"load_seconds\":" << load_seconds
                << ",\"coverage\":";
            write_group_counts(out, g, result.feasible ? result.coverage : std::vector<uint64_t>{});
            out << ",\"internal\":";
            write_group_counts(out, g, result.feasible ? result.internal : std::vector<uint64_t>{});
            out << ",\"demands\":";
            write_group_counts(out, g, demand);
            if (!peeling)
                out << ",\"restarts_requested\":" << starts;
            out << "}\n";
            out.flush();
            if (!out)
                throw std::runtime_error("Record write failed");
            std::cerr << "RESULT " << method << " alpha=" << percent / 100.
                      << " status=" << result.status << " density=" << result.density
                      << " seconds=" << seconds << '\n';
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR " << e.what() << '\n';
        return 1;
    }
}

#include "../common/arguments.hpp"
#include "../common/graph_io.hpp"
#include "../common/output.hpp"
#include "algorithm.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace dense_subgraph;
namespace fs = std::filesystem;

int main(int argc, char **argv) {
    try {
        if (argc < 6 || argc > 8) {
            std::cerr << "algorithm INPUT_DIR N M OUTPUT_DIR QAD [SEED=0] [TARGETS=10,20,30,40]\n";
            return 2;
        }
        const std::string method = argv[5];
        if (method != "QAD")
            throw std::invalid_argument("Unknown algorithm: " + method);
        const uint64_t seed = argc > 6 ? std::stoull(argv[6]) : 0;
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
        const auto targets = parse_targets(argc > 7 ? argv[7] : "10,20,30,40");
        for (int percent : targets) {
            std::vector<uint64_t> demand;
            for (auto count : g.counts)
                demand.push_back((count * percent + 99) / 100);
            auto start = now();
            const auto result = run_quota_active(g, demand, seed);
            const double seconds = elapsed(start);
            const std::string filename = method + "-" + std::to_string(percent) + ".core.bin";
            std::ofstream core(output_dir / filename, std::ios::binary);
            if (result.feasible)
                core.write(reinterpret_cast<const char *>(result.core.data()),
                           result.core.size() * sizeof(uint32_t));
            core.close();
            if (!core)
                throw std::runtime_error("Cannot write core file");
            out << "{\"method\":\"" << method << "\",\"algorithm_name\":\""
                << "Quota-Aware Deletion (QAD)"
                << "\",\"alpha\":" << percent / 100. << ",\"seed\":" << seed << ",\"n\":" << g.n
                << ",\"m\":" << g.m << ",\"groups\":" << g.counts.size()
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
            out << ",\"quota_deletions\":" << result.quota_deletions
                << ",\"raw_construction_density\":" << result.raw_density;
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

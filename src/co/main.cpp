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

static void simple_record(std::ostream &out, const fs::path &folder, const Graph &g,
                          const std::vector<uint64_t> &d, const ContinuousResult &r,
                          const SimpleRun &run, const std::string &mode, int percent, uint64_t seed,
                          double mu, double learning_rate, int steps) {
    const std::string file = "CO-" + mode + "-" + std::to_string(percent) + ".core.bin";
    std::ofstream core(folder / file, std::ios::binary);
    if (!r.core.empty())
        core.write(reinterpret_cast<const char *>(r.core.data()), r.core.size() * sizeof(uint32_t));
    core.close();
    if (!core)
        throw std::runtime_error("Core write failed");
    out << "{\"method\":\"CO\","
        << "\"algorithm_name\":\"Continuous Optimization (CO)\","
        << "\"mode\":\"" << mode << "\",\"alpha\":" << percent / 100. << ",\"seed\":" << seed
        << ",\"mu\":" << mu << ",\"learning_rate\":" << learning_rate << ",\"steps\":" << steps
        << ",\"n\":" << g.n << ",\"m\":" << g.m
        << ",\"groups\":" << g.counts.size() << ",\"feasible\":" << (r.feasible ? "true" : "false")
        << ",\"status\":\"" << r.status << "\",\"core_file\":\"" << file
        << "\",\"core_size\":" << r.core.size() << ",\"internal_edges\":" << r.edges
        << ",\"density\":";
    if (r.feasible)
        out << r.density;
    else
        out << "null";
    out << ",\"candidate_density\":" << r.density << ",\"coverage\":";
    write_group_counts(out, g, r.coverage);
    out << ",\"internal\":";
    write_group_counts(out, g, r.internal);
    out << ",\"demands\":";
    write_group_counts(out, g, d);
    out << ",\"seconds\":" << run.optimization_seconds + run.rounding_seconds
        << ",\"optimization_seconds\":" << run.optimization_seconds
        << ",\"rounding_seconds\":" << run.rounding_seconds << ",\"iterations\":" << run.iterations
        << "}\n";
    out.flush();
    if (!out)
        throw std::runtime_error("Record write failed");
}

int main(int argc, char **argv) {
    try {
        if (argc < 6 || argc > 11)
            throw std::invalid_argument("INPUT N M OUT SEED [MU=256] [TARGETS=10,20,30,40] [MODE=simple] "
                                        "[LEARNING_RATE=0.02] [STEPS=1000]");
        fs::path input = argv[1], folder = argv[4];
        fs::create_directories(folder);
        uint64_t seed = std::stoull(argv[5]);
        double mu = argc > 6 ? std::stod(argv[6]) : kDefaultMu;
        std::string targets = argc > 7 ? argv[7] : "10,20,30,40";
        std::string mode = argc > 8 ? argv[8] : "simple";
        double learning_rate = argc > 9 ? std::stod(argv[9]) : kDefaultLearningRate;
        int steps = argc > 10 ? std::stoi(argv[10]) : kDefaultSteps;
        if (mode != "simple" && mode != "random")
            throw std::invalid_argument("Unknown mode");
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        auto g = load_graph((input / "edges.npy").string(), (input / "labels.npy").string(),
                            std::stoull(argv[2]), std::stoull(argv[3]));
        std::ofstream out(folder / "records.jsonl");
        out << std::setprecision(17);
        for (int percent : parse_targets(targets)) {
            std::vector<uint64_t> d;
            for (auto n : g.counts)
                d.push_back((n * percent + 99) / 100);
            SimpleRun run;
            run = run_simple(g, d, seed, mu, mode == "simple", learning_rate, steps);
            simple_record(out, folder, g, d, run.raw, run, mode, percent, seed, mu, learning_rate, steps);
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR " << e.what() << '\n';
        return 1;
    }
}

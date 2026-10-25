#include "../common/peeling_state.hpp"
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
using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point s) {
    return std::chrono::duration<double>(Clock::now() - s).count();
}
int main(int argc, char **argv) {
    try {
        if (argc < 6 || argc > 10) {
            std::cerr << "inputdir n m outputdir family [rounds=5] [fista_iterations=50] "
                         "[reserved=0] [TARGETS=10,20,30,40]\n";
            return 2;
        }
        fs::path in = argv[1], out = argv[4];
        std::string family = argv[5];
        int rounds = argc > 6 ? std::stoi(argv[6]) : 5,
            iterations = argc > 7 ? std::stoi(argv[7]) : 50,
            reserved = argc > 8 ? std::stoi(argv[8]) : 0;
        if (rounds < 1 || iterations < 1 || reserved < 0)
            throw std::invalid_argument("Positive budgets required");
        const auto targets = parse_targets(argc > 9 ? argv[9] : "10,20,30,40");
        std::vector<double> alphas;
        for (int p : targets)
            alphas.push_back(p / 100.);
        auto started = Clock::now();
        auto g = load_graph((in / "edges.npy").string(), (in / "labels.npy").string(),
                            std::stoull(argv[2]), std::stoull(argv[3]));
        double load_seconds = seconds(started);
        fs::create_directories(out);
        std::ofstream output(out / "records.jsonl");
        output << std::setprecision(17);
        std::vector<std::vector<uint64_t>> ds;
        for (int p : targets) {
            std::vector<uint64_t> d;
            for (auto n : g.counts)
                d.push_back((n * p + 99) / 100);
            ds.push_back(d);
        }
        std::vector<uint64_t> reachable(g.counts.size(), 0);
        for (uint32_t v = 0; v < g.n; ++v)
            if (g.offsets[v] < g.offsets[v + 1])
                ++reachable[g.groups[v]];
        auto emit = [&](const std::string &method, const std::string &name, int index,
                        ContinuousResult r, double runtime, const std::string &constraint,
                        bool native_ok, int completed, const std::string &status = "") {
            auto verified = measure(g, ds[index], r.core);
            r = std::move(verified);
            int pct = targets[index];
            std::string file = method + "-" + std::to_string(pct) + ".bin";
            std::ofstream f(out / file, std::ios::binary);
            if (!r.core.empty())
                f.write((const char *)r.core.data(), r.core.size() * sizeof(uint32_t));
            if (!f)
                throw std::runtime_error("Core write failed");
            output << "{\"method\":\"" << method << "\",\"algorithm_name\":\"" << name
                   << "\",\"alpha\":" << pct / 100. << ",\"n\":" << g.n << ",\"m\":" << g.m
                   << ",\"seed\":0,\"original_constraint\":\"" << constraint
                   << "\",\"original_constraint_satisfied\":" << (native_ok ? "true" : "false")
                   << ",\"external_feasible\":" << (r.feasible ? "true" : "false")
                   << ",\"feasible\":" << (r.feasible ? "true" : "false") << ",\"status\":\""
                   << (status.empty() ? (r.feasible ? "feasible" : "external_demands_not_met")
                                      : status)
                   << "\",\"core_file\":\"" << file << "\",\"core_size\":" << r.core.size()
                   << ",\"internal_edges\":" << r.edges << ",\"candidate_density\":" << r.density
                   << ",\"density\":";
            if (r.feasible)
                output << r.density;
            else
                output << "null";
            output << ",\"numerical_variant\":\"paper_consistent_fista\",\"allocation_step_rule\":"
                      "\"1/(2*maximum_degree)\",\"momentum_rule\":\"(t-1)/"
                      "(t+2)\",\"allocation_initialization\":\"symmetric_0.5\"";
            output << ",\"seconds\":" << runtime << ",\"load_seconds\":" << load_seconds
                   << ",\"work_completed\":" << completed << ",\"internal_fraction_cap_numerator\":"
                   << *std::max_element(g.counts.begin(), g.counts.end())
                   << ",\"internal_fraction_cap_denominator\":" << g.n << ",\"coverage\":";
            write_group_counts(output, g, r.coverage);
            output << ",\"internal\":";
            write_group_counts(output, g, r.internal);
            output << ",\"demands\":";
            write_group_counts(output, g, ds[index]);
            output << "}\n";
            output.flush();
            std::cerr << "RESULT " << method << " alpha=" << pct
                      << " external_feasible=" << r.feasible << " density=" << r.density
                      << " seconds=" << runtime << std::endl;
        };
        auto imposs = [&](int i) {
            for (size_t c = 0; c < g.counts.size(); ++c)
                if (reachable[c] < ds[i][c])
                    return true;
            return false;
        };
        auto t = Clock::now();
        std::vector<ContinuousResult> best;
        int completed = 0;
        std::string name;
        if (family == "fista") {
            auto r = run_baseline_fista(g, iterations, alphas);
            best = std::move(r.best);
            completed = r.iterations_completed;
            name = "FISTA Edge Allocation and Fractional Peeling";
        } else
            throw std::invalid_argument("Unknown baseline family");
        double elapsed = seconds(t);
        if (best.size() != alphas.size())
            throw std::runtime_error("Wrong suffix scan size");
        for (int i = 0; i < int(targets.size()); ++i) {
            emit(family + "_external_prefix",
                 name + " with External-Feasible Prefix Selection (Adapted)", i, best[i], elapsed,
                 "external_coverage_adaptation", false, completed,
                 imposs(i) ? "infeasible_reachability_bound" : "");
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR " << e.what() << std::endl;
        return 1;
    }
}

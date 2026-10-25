#include "../common/graph_io.hpp"
#include "../common/output.hpp"
#include "algorithm.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
static uint64_t integer(const std::string &s) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Expected nonnegative integer: " + s);
    return std::stoull(s);
}
static std::vector<uint64_t> integers(const std::string &s) {
    if (s.empty() || s.back() == ',')
        throw std::invalid_argument("Empty demand/target");
    std::istringstream stream(s);
    std::string token;
    std::vector<uint64_t> out;
    while (std::getline(stream, token, ','))
        out.push_back(integer(token));
    return out;
}
static double number(const std::string &s) {
    size_t used = 0;
    double x = std::stod(s, &used);
    if (used != s.size() || !std::isfinite(x))
        throw std::invalid_argument("Invalid number");
    return x;
}
static void finite_json(std::ostream &out, double x) {
    if (std::isfinite(x))
        out << x;
    else
        out << "null";
}
int main(int argc, char **argv) {
    try {
        if (argc < 6) {
            std::cerr << "algorithm INPUT_DIR N M OUTPUT_DIR densest_then_complete [SEED=0] "
                         "[--targets 10,20,30,40 | --demands 1,16] "
                         "[--time-limit 30] [--stop-factor 6 (0 disables)]\n";
            return 2;
        }
        const std::string method = argv[5];
        if (method != "densest_then_complete" && method != "dtc")
            throw std::invalid_argument("Unknown method: " + method);
        dtc::Options options;
        options.time_limit = 30;
        int pos = 6;
        if (pos < argc && std::string(argv[pos]).rfind("--", 0) != 0)
            options.seed = integer(argv[pos++]);
        std::vector<uint64_t> targets{10, 20, 30, 40}, fixed;
        bool has_fixed = false, has_targets = false;
        for (; pos < argc; pos += 2) {
            if (pos + 1 >= argc)
                throw std::invalid_argument("Option requires a value");
            const std::string key = argv[pos], value = argv[pos + 1];
            if (key == "--demands") {
                fixed = integers(value);
                has_fixed = true;
            } else if (key == "--targets") {
                targets = integers(value);
                has_targets = true;
            } else if (key == "--time-limit")
                options.time_limit = number(value);
            else if (key == "--stop-factor")
                options.stop_factor = number(value);
            else
                throw std::invalid_argument("Unknown option: " + key);
        }
        if (has_fixed && has_targets)
            throw std::invalid_argument("Choose demands OR targets");
        if (options.time_limit <= 0)
            throw std::invalid_argument("Time limit must be positive");
        if (options.stop_factor != 0 && options.stop_factor < 1)
            throw std::invalid_argument("Stop factor must be 0 or >= 1");
        std::vector<uint64_t> seen;
        for (auto t : targets) {
            if (t > 100 || std::find(seen.begin(), seen.end(), t) != seen.end())
                throw std::invalid_argument("Targets must be distinct percentages in [0,100]");
            seen.push_back(t);
        }
        const auto load_start = std::chrono::steady_clock::now();
        const fs::path input = argv[1], output = argv[4];
        auto g = dense_subgraph::load_graph((input / "edges.npy").string(),
                                            (input / "labels.npy").string(), integer(argv[2]),
                                            integer(argv[3]));
        if (!g.n)
            throw std::invalid_argument("Graph must be nonempty");
        if (has_fixed && fixed.size() != g.counts.size())
            throw std::invalid_argument(
                "One demand per present color, in ascending original color-ID order, is required");
        dtc::Graph native;
        native.colors.assign(g.groups.begin(), g.groups.end());
        native.adjacency.resize(g.n);
        for (uint32_t v = 0; v < g.n; ++v)
            native.adjacency[v].assign(g.neighbors.begin() + g.offsets[v],
                                       g.neighbors.begin() + g.offsets[v + 1]);
        std::vector<uint32_t>().swap(g.neighbors);
        std::vector<uint64_t>().swap(g.offsets);
        const double load_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();
        fs::create_directories(output);
        if (fs::exists(output / "records.jsonl"))
            throw std::runtime_error("Output already contains records.jsonl; use a new directory");
        std::ofstream out(output / "records.jsonl");
        if (!out)
            throw std::runtime_error("Cannot write records.jsonl");
        out << std::setprecision(17);
        for (size_t run = 0; run < (has_fixed ? 1 : targets.size()); ++run) {
            std::vector<uint64_t> demand = fixed;
            if (!has_fixed)
                for (auto count : g.counts)
                    demand.push_back((count * targets[run] + 99) / 100);
            const auto start = std::chrono::steady_clock::now();
            const auto result = dtc::solve(native, demand, options);
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const std::string suffix = has_fixed ? "fixed" : std::to_string(targets[run]);
            const std::string filename = "densest_then_complete-" + suffix + ".core.bin";
            std::ofstream core(output / filename, std::ios::binary);
            if (result.feasible)
                core.write(reinterpret_cast<const char *>(result.core.data()),
                           result.core.size() * sizeof(uint32_t));
            core.close();
            if (!core)
                throw std::runtime_error("Cannot write core file");
            std::vector<uint64_t> internal(g.counts.size());
            for (auto v : result.core)
                ++internal[g.groups[v]];
            out << "{\"method\":\"densest_then_complete\",\"algorithm_name\":\"Densest-then-"
                   "Complete\",\"alpha\":";
            if (has_fixed)
                out << "null";
            else
                out << targets[run] / 100.;
            out << ",\"demand_mode\":\"" << (has_fixed ? "fixed" : "fraction") << "\",\"seed\":" << options.seed
                << ",\"n\":" << g.n << ",\"m\":" << g.m << ",\"groups\":" << g.counts.size()
                << ",\"feasible\":" << (result.feasible ? "true" : "false") << ",\"status\":\""
                << result.status << "\",\"termination_reason\":\"" << result.termination_reason
                << "\",\"core_file\":\"" << filename << "\",\"core_size\":" << result.core.size()
                << ",\"internal_edges\":" << result.internal_edges << ",\"density\":";
            if (result.feasible)
                out << result.density;
            else
                out << "null";
            out << ",\"seconds\":" << seconds << ",\"search_seconds\":" << result.search_seconds
                << ",\"load_seconds\":" << load_seconds << ",\"time_limit\":" << options.time_limit
                << ",\"stop_factor\":";
            if (options.stop_factor)
                out << options.stop_factor;
            else
                out << "null";
            out << ",\"coverage\":";
            dense_subgraph::write_group_counts(out, g, result.coverage);
            out << ",\"internal\":";
            dense_subgraph::write_group_counts(out, g, internal);
            out << ",\"demands\":";
            dense_subgraph::write_group_counts(out, g, demand);
            out << ",\"density_upper_bound\":";
            finite_json(out, result.density_upper_bound);
            out << ",\"approximation_factor\":";
            finite_json(out, result.approximation_factor);
            out << ",\"target_met\":" << (result.target_met ? "true" : "false");
            out << ",\"first_feasible_seconds\":";
            finite_json(out, result.first_feasible_seconds);
            out << ",\"winning_depth\":";
            if (result.winning_depth >= 0)
                out << result.winning_depth;
            else
                out << "null";
            const auto &s = result.statistics;
            out << ",\"statistics\":{\"oracle_calls\":" << s.oracle_calls
                << ",\"oracle_seconds\":" << s.oracle_seconds
                << ",\"completion_calls\":" << s.completion_calls
                << ",\"completion_seconds\":" << s.completion_seconds
                << ",\"representative_search_nodes\":" << s.representative_search_nodes
                << ",\"recursion_nodes\":" << s.recursion_nodes
                << ",\"maximum_depth\":" << s.maximum_depth
                << ",\"duplicate_nodes\":" << s.duplicate_nodes
                << ",\"small_core_calls\":" << s.small_core_calls
                << ",\"small_core_skipped_by_bound\":"
                << (s.small_core_skipped_by_bound ? "true" : "false") << ",\"winning_stage\":\""
                << s.winning_stage << "\"}}\n";
            out.flush();
            if (!out)
                throw std::runtime_error("Record write failed");
            std::cerr << "RESULT dtc status=" << result.status << " density=" << result.density
                      << " seconds=" << seconds << '\n';
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR " << e.what() << '\n';
        return 1;
    }
}

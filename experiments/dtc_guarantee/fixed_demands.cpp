// Runs QAD, CO, MinimumPeeling, Greedy++ or FISTA with absolute external demands.
// Compile with exactly one of -DMETHOD_QAD, -DMETHOD_CO, -DMETHOD_PEELING,
// -DMETHOD_GREEDY, -DMETHOD_FISTA and link the matching src/<method>/algorithm.cpp.
//
// QAD and CO take a demand vector directly. The three peeling baselines take
// coverage fractions and derive each demand as ceil(alpha * |V_c|) from
// g.counts; their deletion order reads neither. With alpha = 0.5 (integer
// percent 50), a count of 2 d_c yields demand (100 d_c + 99) / 100 = d_c
// exactly, so these counts are swapped in for the call and restored after it.
// Their internal color tallies also start from g.counts and are discarded: the
// returned core is audited directly from adjacency.
#if defined(METHOD_QAD) + defined(METHOD_CO) + defined(METHOD_PEELING) + defined(METHOD_GREEDY) + \
        defined(METHOD_FISTA) !=                                                                   \
    1
#error Define exactly one METHOD_* macro
#endif

#include "common/graph_io.hpp"
#if defined(METHOD_QAD)
#include "qad/algorithm.hpp"
#elif defined(METHOD_CO)
#include "co/algorithm.hpp"
#elif defined(METHOD_PEELING)
#include "peeling/algorithm.hpp"
#elif defined(METHOD_GREEDY)
#include "iterative_peeling/algorithm.hpp"
#else
#include "fista/algorithm.hpp"
#endif

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace dense_subgraph;
namespace fs = std::filesystem;

namespace {
std::vector<uint64_t> parse_demands(const std::string &text, size_t groups) {
    std::vector<uint64_t> result;
    std::istringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty() || token.find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument("Demands must be nonnegative integers");
        result.push_back(std::stoull(token));
    }
    if (text.empty() || text.back() == ',' || result.size() != groups)
        throw std::invalid_argument("Provide one demand per color, in sorted color-ID order");
    return result;
}

void write_counts(std::ostream &out, const std::vector<uint64_t> &values) {
    out << '{';
    for (size_t c = 0; c < values.size(); ++c)
        out << (c ? "," : "") << '"' << c << "\":" << values[c];
    out << '}';
}

struct Selected {
    std::vector<uint32_t> core;
    std::vector<uint64_t> coverage;
    uint64_t edges = 0;
    bool feasible = false;
};

#if defined(METHOD_PEELING) || defined(METHOD_GREEDY) || defined(METHOD_FISTA)
// Swaps in counts of 2 d_c for one baseline call; see the file header.
struct DemandCounts {
    Graph &g;
    std::vector<uint64_t> saved;
    DemandCounts(Graph &graph, const std::vector<uint64_t> &d) : g(graph), saved(graph.counts) {
        for (size_t c = 0; c < d.size(); ++c)
            g.counts[c] = 2 * d[c];
    }
    ~DemandCounts() { g.counts = saved; }
};
#endif

Selected run(Graph &g, const std::vector<uint64_t> &d, uint64_t seed) {
    Selected s;
#if defined(METHOD_QAD)
    const auto r = run_quota_active(g, d, seed);
    s = {r.core, r.coverage, r.edges, r.feasible};
#elif defined(METHOD_CO)
    const auto r = run_simple(g, d, seed, kDefaultMu).raw;
    s = {r.core, r.coverage, r.edges, r.feasible};
#else
    (void)seed;
    DemandCounts swap(g, d);
#if defined(METHOD_PEELING)
    const auto r = run_peeling(g, {.5}, 2);
    const auto &best = r.best[0];
    s.feasible = best.step >= 0;
    if (s.feasible) {
        s.core.assign(r.order.begin() + best.step, r.order.end());
        s.coverage = best.coverage;
        s.edges = best.edges;
    }
#elif defined(METHOD_GREEDY)
    const auto r = run_iterative_peeling(g, 5, {.5});
    const auto &best = r.best[0];
    s = {best.core, best.coverage, best.edges, best.feasible};
#else
    const auto r = run_baseline_fista(g, 50, {.5});
    const auto &best = r.best[0];
    s = {best.core, best.coverage, best.edges, best.feasible};
#endif
#endif
    return s;
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 7)
            throw std::invalid_argument("INPUT_DIR N M OUTPUT_DIR DEMANDS SEED");
        const fs::path input = argv[1], output = argv[4];
        auto g = load_graph((input / "edges.npy").string(), (input / "labels.npy").string(),
                            std::stoull(argv[2]), std::stoull(argv[3]));
        const auto d = parse_demands(argv[5], g.counts.size());
        const uint64_t seed = std::stoull(argv[6]);
        if (fs::exists(output / "records.jsonl"))
            throw std::runtime_error("Output already exists");
        fs::create_directories(output);
        const auto start = std::chrono::steady_clock::now();
        const Selected s = run(g, d, seed);
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        // Audit the returned core directly from adjacency.
        std::vector<unsigned char> in(g.n, 0), covered(g.n, 0);
        std::vector<uint64_t> coverage(d.size(), 0);
        for (auto v : s.core) {
            if (v >= g.n || in[v])
                throw std::runtime_error("Invalid or duplicate core vertex");
            in[v] = 1;
        }
        uint64_t directed_edges = 0;
        for (auto v : s.core)
            for (uint64_t j = g.offsets[v]; j < g.offsets[v + 1]; ++j) {
                const auto u = g.neighbors[j];
                if (in[u])
                    ++directed_edges;
                else
                    covered[u] = 1;
            }
        for (uint32_t v = 0; v < g.n; ++v)
            if (covered[v])
                ++coverage[g.groups[v]];
        bool feasible = !s.core.empty();
        for (size_t c = 0; c < d.size(); ++c)
            feasible = feasible && coverage[c] >= d[c];
        const uint64_t edges = directed_edges / 2;
        if (feasible != s.feasible || (s.feasible && (edges != s.edges || coverage != s.coverage)))
            throw std::runtime_error("Independent core audit differs from algorithm result");

        std::ofstream core(output / "core.bin", std::ios::binary);
        core.write(reinterpret_cast<const char *>(s.core.data()),
                   std::streamsize(s.core.size() * sizeof(uint32_t)));
        if (!core)
            throw std::runtime_error("Core write failed");
        std::ofstream out(output / "records.jsonl");
        out << std::setprecision(17) << "{\"feasible\":" << (feasible ? "true" : "false")
            << ",\"core_file\":\"core.bin\",\"core_size\":" << s.core.size()
            << ",\"internal_edges\":" << edges << ",\"density\":";
        if (feasible)
            out << double(edges) / double(s.core.size());
        else
            out << "null";
        out << ",\"coverage\":";
        write_counts(out, coverage);
        out << ",\"demands\":";
        write_counts(out, d);
        out << ",\"seconds\":" << seconds << "}\n";
        if (!out)
            throw std::runtime_error("Record write failed");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ERROR " << error.what() << '\n';
        return 1;
    }
}

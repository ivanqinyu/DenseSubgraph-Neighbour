#include "algorithm.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

static void require(bool ok, const char *msg) {
    if (!ok)
        throw std::runtime_error(msg);
}
int main(int argc, char **argv) {
    try {
        auto started = std::chrono::steady_clock::now();
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);
        int n;
        I m;
        std::ifstream binary;
        if (argc > 1) {
            binary.open(argv[1], std::ios::binary);
            require(bool(binary), "Cannot open binary input");
            uint64_t dims[2];
            binary.read(reinterpret_cast<char *>(dims), sizeof(dims));
            require(bool(binary), "Truncated binary header");
            require(dims[0] > 0 && dims[0] < uint64_t(std::numeric_limits<int>::max() / 3) &&
                        dims[1] < uint64_t(std::numeric_limits<int>::max() / 2),
                    "Invalid binary dimensions");
            n = int(dims[0]);
            m = I(dims[1]);
        } else
            require(bool(std::cin >> n >> m), "Expected n m");
        require(n > 0 && n < std::numeric_limits<int>::max() / 3 && m >= 0, "Invalid n or m");
        std::vector<std::pair<int, int>> edges;
        edges.reserve(size_t(m));
        for (I k = 0; k < m; ++k) {
            int u, v;
            if (binary.is_open()) {
                uint32_t pair[2];
                binary.read(reinterpret_cast<char *>(pair), sizeof(pair));
                require(bool(binary), "Truncated binary edges");
                require(pair[0] < uint32_t(n) && pair[1] < uint32_t(n), "Invalid binary edge");
                u = int(pair[0]);
                v = int(pair[1]);
            } else
                require(bool(std::cin >> u >> v), "Truncated edge input");
            require(u >= 0 && u < n && v >= 0 && v < n && u != v, "Invalid edge");
            if (u > v)
                std::swap(u, v);
            edges.emplace_back(u, v);
        }
        // A sorted copy is deliberately avoided at scale. Sort the retained edge
        // list itself to independently enforce the simple-graph input contract.
        std::sort(edges.begin(), edges.end());
        require(std::adjacent_find(edges.begin(), edges.end()) == edges.end(), "Duplicate edge");
        auto result = solve(n, edges);
        const auto p = result.p, q = result.q;
        const auto &core = result.core;
        const auto &its = result.its;
        double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout.precision(17);
        std::cout << "{\"numerator\":" << p << ",\"denominator\":" << q
                  << ",\"density\":" << double(p) / q << ",\"core\":[";
        for (size_t k = 0; k < core.size(); ++k) {
            if (k)
                std::cout << ',';
            std::cout << core[k];
        }
        std::cout << "],\"integer_certified\":true,\"iterations\":[";
        for (size_t k = 0; k < its.size(); ++k) {
            if (k)
                std::cout << ',';
            auto t = its[k];
            std::cout << "{\"numerator\":" << t.p << ",\"denominator\":" << t.q
                      << ",\"flow_value\":" << t.value << ",\"required_flow\":" << t.total
                      << ",\"flow_verified\":true}";
        }
        std::cout << "],\"elapsed_seconds\":" << elapsed
                  << ",\"backend\":\"cpp_int64_push_relabel\"}\n";
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}

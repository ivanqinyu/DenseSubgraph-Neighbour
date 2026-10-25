#include "graph_io.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace dense_subgraph {
struct NpyMap {
    int fd = -1;
    size_t length = 0, offset = 0;
    void *mapping = MAP_FAILED;
    NpyMap(const std::string &path, const std::string &dtype, const std::vector<uint64_t> &shape) {
        try {
            fd = ::open(path.c_str(), O_RDONLY);
            if (fd < 0)
                throw std::runtime_error("Cannot open " + path);
            struct stat s;
            if (fstat(fd, &s) || s.st_size < 10)
                throw std::runtime_error("Invalid NPY " + path);
            length = size_t(s.st_size);
            mapping = mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED)
                throw std::runtime_error("Cannot map " + path);
            const auto *b = static_cast<const unsigned char *>(mapping);
            if (std::memcmp(b, "\x93NUMPY", 6))
                throw std::runtime_error("NPY magic " + path);
            size_t prefix = b[6] == 1 ? 10 : (b[6] == 2 || b[6] == 3 ? 12 : 0);
            if (!prefix || length < prefix)
                throw std::runtime_error("NPY version " + path);
            uint64_t header = uint64_t(b[8]) | (uint64_t(b[9]) << 8);
            if (prefix == 12)
                header |= (uint64_t(b[10]) << 16) | (uint64_t(b[11]) << 24);
            if (header > length - prefix)
                throw std::runtime_error("NPY truncated header " + path);
            offset = prefix + size_t(header);
            std::string h(reinterpret_cast<const char *>(b + prefix), size_t(header));
            std::smatch match;
            if (!std::regex_search(h, match,
                                   std::regex("['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]")))
                throw std::runtime_error("NPY missing dtype");
            std::string actual = match[1];
            if (actual != dtype && actual != "=" + dtype.substr(1))
                throw std::runtime_error("NPY unexpected dtype " + actual + " in " + path);
            if (!std::regex_search(h, std::regex("['\"]fortran_order['\"]\\s*:\\s*False")))
                throw std::runtime_error("NPY must be C order");
            if (!std::regex_search(h, match, std::regex("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)")))
                throw std::runtime_error("NPY missing shape");
            std::string dims = match[1];
            std::regex number("[0-9]+");
            std::vector<uint64_t> found;
            for (auto it = std::sregex_iterator(dims.begin(), dims.end(), number);
                 it != std::sregex_iterator(); ++it)
                found.push_back(std::stoull(it->str()));
            if (found != shape)
                throw std::runtime_error("NPY shape mismatch " + path);
            uint64_t bytes = std::stoull(dtype.substr(2));
            for (auto d : shape) {
                if (d && bytes > UINT64_MAX / d)
                    throw std::runtime_error("NPY shape overflow");
                bytes *= d;
            }
            if (bytes != length - offset)
                throw std::runtime_error("NPY payload mismatch " + path);
            madvise(mapping, length, MADV_SEQUENTIAL);
        } catch (...) {
            if (mapping != MAP_FAILED)
                munmap(mapping, length);
            if (fd >= 0)
                close(fd);
            throw;
        }
    }
    NpyMap(const NpyMap &) = delete;
    NpyMap &operator=(const NpyMap &) = delete;
    ~NpyMap() {
        if (mapping != MAP_FAILED)
            munmap(mapping, length);
        if (fd >= 0)
            close(fd);
    }
    template <class T> const T *data() const {
        return reinterpret_cast<const T *>(static_cast<const unsigned char *>(mapping) + offset);
    }
};
Graph load_graph(const std::string &edge_path, const std::string &label_path, uint64_t n,
                 uint64_t m) {
    if (n >= UINT32_MAX || m > SIZE_MAX / 2)
        throw std::runtime_error("Graph too large for native IDs");
    Graph g;
    g.n = uint32_t(n);
    g.m = m;
    g.offsets.assign(n + 1, 0);
    g.groups.resize(n);
    NpyMap labels(label_path, "<u2", {n});
    const uint16_t *colors = labels.data<uint16_t>();
    std::array<uint64_t, 65536> counts{};
    std::array<uint16_t, 65536> dense{};
    for (uint64_t v = 0; v < n; ++v)
        ++counts[colors[v]];
    for (size_t c = 0; c < counts.size(); ++c)
        if (counts[c]) {
            dense[c] = uint16_t(g.counts.size());
            g.color_ids.push_back(uint16_t(c));
            g.counts.push_back(counts[c]);
        }
    for (uint64_t v = 0; v < n; ++v)
        g.groups[v] = dense[colors[v]];
    NpyMap edges(edge_path, "<u4", {m, 2});
    const uint32_t *e = edges.data<uint32_t>();
    uint64_t previous = 0;
    for (uint64_t i = 0; i < m; ++i) {
        uint32_t u = e[2 * i], v = e[2 * i + 1];
        uint64_t key = (uint64_t(u) << 32) | v;
        if (u >= v || v >= n || (i && key <= previous))
            throw std::runtime_error("Edges must be normalized sorted unique and in range");
        previous = key;
        ++g.offsets[u + 1];
        ++g.offsets[v + 1];
    }
    for (uint64_t v = 1; v <= n; ++v)
        g.offsets[v] += g.offsets[v - 1];
    g.neighbors.resize(2 * m);
    {
        std::vector<uint64_t> cursor(g.offsets.begin(), g.offsets.end() - 1);
        for (uint64_t i = 0; i < m; ++i) {
            uint32_t u = e[2 * i], v = e[2 * i + 1];
            g.neighbors[cursor[u]++] = v;
            g.neighbors[cursor[v]++] = u;
        }
    }
    std::cerr << "Loaded CSR n=" << n << " m=" << m << " groups=" << g.counts.size() << std::endl;
    return g;
}
} // namespace dense_subgraph

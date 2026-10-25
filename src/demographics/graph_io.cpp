#include "graph_io.hpp"
#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct Npy {
    int fd = -1;
    size_t len = 0, off = 0;
    void *ptr = MAP_FAILED;
    std::string dtype;
    std::vector<uint64_t> shape;
    explicit Npy(const std::string &path) {
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0)
            throw std::runtime_error("cannot open " + path);
        struct stat st{};
        if (fstat(fd, &st))
            throw std::runtime_error("stat failed");
        len = st.st_size;
        if (len < 10)
            throw std::runtime_error("short NPY");
        ptr = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED)
            throw std::runtime_error("mmap failed");
        auto b = static_cast<const uint8_t *>(ptr);
        if (memcmp(b, "\x93NUMPY", 6))
            throw std::runtime_error("invalid NPY");
        size_t pre = b[6] == 1 ? 10 : (b[6] == 2 || b[6] == 3 ? 12 : 0);
        if (!pre || len < pre)
            throw std::runtime_error("NPY version");
        uint64_t h = b[8] | (uint64_t(b[9]) << 8);
        if (pre == 12)
            h |= uint64_t(b[10]) << 16 | uint64_t(b[11]) << 24;
        if (h > len - pre)
            throw std::runtime_error("short NPY header");
        off = pre + h;
        std::string hs((const char *)b + pre, h);
        std::smatch m;
        if (!std::regex_search(hs, m, std::regex("['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)")))
            throw std::runtime_error("NPY dtype missing");
        dtype = m[1];
        if (std::regex_search(hs, std::regex("['\"]fortran_order['\"]\\s*:\\s*True")))
            throw std::runtime_error("Fortran NPY unsupported");
        if (!std::regex_search(hs, m, std::regex("['\"]shape['\"]\\s*:\\s*\\(([^)]*)")))
            throw std::runtime_error("NPY shape missing");
        std::string dims = m[1];
        std::regex num("[0-9]+");
        for (auto it = std::sregex_iterator(dims.begin(), dims.end(), num);
             it != std::sregex_iterator(); ++it)
            shape.push_back(std::stoull(it->str()));
    }
    template <class T> const T *data() const {
        return reinterpret_cast<const T *>(static_cast<const uint8_t *>(ptr) + off);
    }
    void validate(const std::vector<uint64_t> &want, size_t width) const {
        if (shape != want)
            throw std::runtime_error("NPY shape mismatch");
        uint64_t count = 1;
        for (auto d : want) {
            if (d && count > UINT64_MAX / d)
                throw std::runtime_error("NPY overflow");
            count *= d;
        }
        if (count > (len - off) / width)
            throw std::runtime_error("short NPY payload");
    }
    ~Npy() {
        if (ptr != MAP_FAILED)
            munmap(ptr, len);
        if (fd >= 0)
            close(fd);
    }
    Npy(const Npy &) = delete;
};
Graph load(const std::string &dir, uint64_t n, uint64_t m) {
    if (n == 0 || n > UINT32_MAX)
        throw std::runtime_error("invalid N");
    Npy l(dir + "/labels.npy"), e(dir + "/edges.npy");
    bool byte = l.dtype == "|u1" || l.dtype == "<u1";
    if (!byte && l.dtype != "<u2")
        throw std::runtime_error("labels must be uint8 or little-endian uint16");
    if (e.dtype != "<u4")
        throw std::runtime_error("edges must be little-endian uint32");
    l.validate({n}, byte ? 1 : 2);
    e.validate({m, 2}, 4);
    Graph g;
    g.n = n;
    g.col.resize(n);
    uint32_t mx = 0;
    for (uint32_t i = 0; i < n; i++) {
        g.col[i] = byte ? l.data<uint8_t>()[i] : l.data<uint16_t>()[i];
        mx = std::max(mx, g.col[i]);
    }
    g.pop.resize(mx + 1);
    for (auto c : g.col)
        g.pop[c]++;
    g.off.assign(n + 1, 0);
    auto ep = e.data<uint32_t>();
    for (uint64_t j = 0; j < m; j++) {
        auto u = ep[2 * j], v = ep[2 * j + 1];
        if (u >= n || v >= n)
            throw std::runtime_error("edge endpoint out of range");
        if (u == v)
            continue;
        g.off[u + 1]++;
        g.off[v + 1]++;
        g.m++;
    }
    std::partial_sum(g.off.begin(), g.off.end(), g.off.begin());
    g.adj.resize(g.off.back());
    auto pos = g.off;
    for (uint64_t j = 0; j < m; j++) {
        auto u = ep[2 * j], v = ep[2 * j + 1];
        if (u == v)
            continue;
        g.adj[pos[u]++] = v;
        g.adj[pos[v]++] = u;
    }
    return g;
}

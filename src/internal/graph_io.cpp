#include "graph_io.hpp"
#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <regex>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct Npy {
    int fd = -1;
    size_t len = 0, off = 0;
    void *map = MAP_FAILED;
    Npy(const std::string &p, const std::string &dtype, const std::vector<uint64_t> &shape) {
        fd = open(p.c_str(), O_RDONLY);
        if (fd < 0)
            throw std::runtime_error("cannot open " + p);
        struct stat s{};
        if (fstat(fd, &s))
            throw std::runtime_error("stat failed");
        len = s.st_size;
        map = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (map == MAP_FAILED)
            throw std::runtime_error("mmap failed");
        const auto *b = (const unsigned char *)map;
        if (len < 10 || memcmp(b, "\x93NUMPY", 6))
            throw std::runtime_error("invalid npy");
        size_t pre = b[6] == 1 ? 10 : (b[6] == 2 || b[6] == 3 ? 12 : 0);
        uint64_t h = b[8] | (uint64_t(b[9]) << 8);
        if (pre == 12)
            h |= uint64_t(b[10]) << 16 | uint64_t(b[11]) << 24;
        off = pre + h;
        std::string hs((const char *)b + pre, h);
        std::smatch m;
        static const std::regex descr("['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)");
        static const std::regex sh("['\"]shape['\"]\\s*:\\s*\\(([^)]*)");
        static const std::regex num("[0-9]+");
        if (!std::regex_search(hs, m, descr) || m[1] != dtype)
            throw std::runtime_error("unexpected npy dtype");
        if (!std::regex_search(hs, m, sh))
            throw std::runtime_error("missing shape");
        std::string dims = m[1].str();
        std::vector<uint64_t> got;
        for (auto it = std::sregex_iterator(dims.begin(), dims.end(), num);
             it != std::sregex_iterator(); ++it)
            got.push_back(std::stoull(it->str()));
        if (got != shape)
            throw std::runtime_error("npy shape mismatch");
    }
    template <class T> const T *data() const {
        return (const T *)((const unsigned char *)map + off);
    }
    ~Npy() {
        if (map != MAP_FAILED)
            munmap(map, len);
        if (fd >= 0)
            close(fd);
    }
    Npy(const Npy &) = delete;
};
Graph load(const std::string &dir, uint64_t n, uint64_t m) {
    Npy L(dir + "/labels.npy", "<u2", {n}), E(dir + "/edges.npy", "<u4", {m, 2});
    Graph g{(uint32_t)n, m, std::vector<uint16_t>(n), std::vector<std::vector<uint32_t>>(n), 0};
    const auto *l = L.data<uint16_t>();
    uint16_t mx = 0;
    for (uint64_t i = 0; i < n; i++) {
        g.c[i] = l[i];
        mx = std::max(mx, l[i]);
    }
    g.C = mx + 1;
    const auto *e = E.data<uint32_t>();
    for (uint64_t i = 0; i < m; i++) {
        uint32_t u = e[2 * i], v = e[2 * i + 1];
        if (u >= n || v >= n || u == v)
            continue;
        g.a[u].push_back(v);
        g.a[v].push_back(u);
    }
    return g;
}

#pragma once
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dense_subgraph {
inline std::vector<int> parse_targets(const std::string &text, bool unconstrained = false) {
    std::vector<int> result;
    std::istringstream input(text);
    std::string token;
    while (std::getline(input, token, ',')) {
        size_t used = 0;
        int value = std::stoi(token, &used);
        if (used != token.size() ||
            (unconstrained ? value != 0
                           : (value != 10 && value != 20 && value != 30 && value != 40)))
            throw std::invalid_argument(
                "Targets must be 10,20,30,40 (0 only for unconstrained peeling)");
        if (std::find(result.begin(), result.end(), value) != result.end())
            throw std::invalid_argument("Duplicate target");
        result.push_back(value);
    }
    if (result.empty() || text.back() == ',')
        throw std::invalid_argument("Empty target");
    return result;
}
} // namespace dense_subgraph

// Exhaustive meet-in-the-middle template search  F1 . Ga . M . Gb . F2  (and the variant with a three-qubit database middle)
// on the valid inputs of a classical function. Mirrors toffoli_optimizer/core/template.py::template_search.
#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "library.hpp"

namespace qcpt {

struct DbMid {                      // placeholder for a three-qubit database circuit; expanded to gates by the caller
    int q[3];
    std::array<int, 8> perm;
};

struct Step {                       // one element of the result: a library gate (index) or a database middle
    int gate = -1;
    DbMid db{};
};

struct TemplateParams {
    int n = 0, max_cost = 0, f1 = 1, fm = 0, f2 = 1, nproc = 1;
    std::vector<int> inputs, target;
    std::vector<std::string> ga, gb, middles;       // gadget names for the first / second slot, single-gate middles
    uint64_t seed = 0;
    bool verbose = false;
    std::atomic<bool>* cancel = nullptr;
};

struct TemplateResult {
    bool found = false;
    int cost = 0;
    std::vector<Step> steps;
};

// db: cost of every three-qubit permutation in the database, indexed by sum p[i] << 3i (255 = not in the database); empty = no database middle
TemplateResult template_search(const TemplateParams& P, const Library& lib, const std::vector<uint8_t>& db);

std::vector<uint8_t> load_db(const std::string& path);

}  // namespace qcpt

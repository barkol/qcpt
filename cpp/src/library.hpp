// Gate library of the classical-with-phases (Z8 monomial) search space: every gate is a permutation of the 2^n basis states plus a phase
// in Z8 (powers of exp(i pi/4)) attached to each state. Mirrors toffoli_optimizer/core/resynth.py::library line by line.
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace qcpt {

struct Gate {
    std::string name;
    int nargs = 0;
    int args[5] = {0, 0, 0, 0, 0};
    int cost = 0;                       // number of CX
    std::vector<uint16_t> p, ip;        // state before -> state after (and the inverse gate)
    std::vector<uint8_t> f, inf;        // phase added by the gate on the state before it (and of the inverse gate)
};

using Library = std::vector<Gate>;

// extra_signs adds the four remaining sign patterns of the parity gadgets (used by the template search only)
Library make_library(int n, bool extra_signs);

}  // namespace qcpt

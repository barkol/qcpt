// Exact check of a gate sequence against a target on the valid inputs: outputs equal, phases equal up to one global phase.
#pragma once
#include <vector>

#include "library.hpp"

namespace qcpt {

// gates: indices into lib, applied in order. target_phase empty = all zero (pure permutation target).
bool certify(const Library& lib, const std::vector<int>& gates, const std::vector<int>& inputs, const std::vector<int>& target, const std::vector<int>& target_phase);

}  // namespace qcpt

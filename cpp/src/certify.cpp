#include "certify.hpp"

namespace qcpt {

bool certify(const Library& lib, const std::vector<int>& gates, const std::vector<int>& inputs, const std::vector<int>& target, const std::vector<int>& target_phase) {
    const size_t m = inputs.size(); if (target.size() != m) return false;
    int ph0 = 0, tp0 = target_phase.empty() ? 0 : target_phase[0];
    for (size_t i = 0; i < m; ++i) {
        int y = inputs[i], s = 0;
        for (int gi : gates) { const Gate& g = lib[gi]; s += g.f[y]; y = g.p[y]; }
        if (y != target[i]) return false;
        if (i == 0) ph0 = s;
        const int tp = target_phase.empty() ? 0 : target_phase[i];
        if ((((s - ph0) - (tp - tp0)) & 7) != 0) return false;
    }
    return true;
}

}  // namespace qcpt

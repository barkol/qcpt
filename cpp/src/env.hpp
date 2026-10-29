// Environment variables of the QCPT binary are named QCPT_*. The old names QCT_* (from before the rename to QCPT) are still read as a deprecated fallback.
#pragma once
#include <cstdlib>
#include <string>

#if defined(QCT_CUDA) && !defined(QCPT_CUDA)   // deprecated compile-time alias
#define QCPT_CUDA
#endif

namespace qcpt {

inline const char* getenv_q(const char* name) {            // name = "QCPT_X"; falls back to the deprecated "QCT_X"
    if (const char* v = std::getenv(name)) return v;
    return std::getenv((std::string("QCT_") + (name + 5)).c_str());
}

}  // namespace qcpt

// Mellin throat recursion (code round 463; research round 1189, S2b part B): the l = 2 slicing flow's entry point -- implemented in mk_slice.cpp,
// its own translation unit (the kernel's must not see the flow's code: see mk_engine.hpp's Engine::s2b_flow).
#pragma once
#include "mk_kernel.hpp"

#include <string>
#include <vector>

namespace mk {
namespace sl {
struct FlowIn {
    const std::vector<double>* TH = nullptr;   // the engine's theta grid
    double J = 0.0;
    std::string out;                           // OUT (s2b_flow.json, fields_bare_*)
    cd a1 = 0.0;                               // PH2's pinned amplitude (the last pass)
    int jph = -1;                              // PH2's position among the PH columns
    bool quiet = false;                        // (the fit's fvec: no files, no log)
};
// F (psi2 Phb qf br bt Qb, gauge-flowed bare) -> F0 + a G + (a^2 / 2) DG.G [+ (a^3 / 6) D(DG.G).G]
void flow_apply(Kernel& k, Ser F[6], const FlowIn& in);
}  // namespace sl
}  // namespace mk

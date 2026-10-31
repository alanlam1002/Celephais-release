// Mellin throat recursion, M2d (code round 463; research rounds 1018-1024): the NONLINEAR STEPS at a centre.
//
// spinning_trumpet/scripts/n2a_nl.py, after the data rule (M2c): the rows of the block are not exactly linear in the step's unknowns, so
//   integer centre    defect correction against the full rows (--defect, default 8): rows at the current X (Galerkin, active columns),
//                     stop at 1e-13 x max|S| (or, with --defect-stop-robust, when a correction fails to halve them); the correction
//                     -M^-1 rows at the nodes with its components along the residue directions removed (the amplitudes held), then
//                     --cond-fit's data fix; the drift of the u_g residues; the rows after; Newton: re-linearised about the current u_g
//                     (sources rows(X_k) - M_k X_k at the nodes) while the rows exceed 1e-12 x max|S| and a step halves them
//   off-integer       defect correction (max(--defect, 4) + 1 evaluations), stop at 1e-14 absolute, the family's amplitudes held; the
//                     conditions (not imposed) at the final X
// The rows reach this code as callbacks built in the kernel's translation unit (mk_kernel.cpp), as M2c's conditions do.
#pragma once
#include "mk_data.hpp"

namespace mk {
namespace st {

using cd = std::complex<double>;
using Vec = std::vector<cd>;

constexpr double DC_STOP = 1e-13;      // dc_hist[-1] <= 1e-13 * max(float(np.abs(S).max()), 1e-300)
constexpr double OFF_STOP = 1e-14;     // dres_hist[-1] < 1e-14
constexpr double HALVE = 0.5;          // the robust stop; Newton's stagnation: rfin_ > 0.5 * newton_hist[-2]
constexpr double NEWTON_TOL = 1e-12;   // rfin_ > 1e-12 * max(float(np.abs(S).max()), 1e-300)

struct Rows {
    std::function<Vec(const Vec& X)> gal;      // galb(rows_a(with_grade(strip(C, g), X, g), TOP), g): K x NB
    std::function<Vec(const Vec& X)> stored;   // stored_src(rows_a(...), P, clab, Oc['dE'], g): K x nc
};

struct DefectOut {
    Vec X;
    std::vector<double> hist;
    double drift = 0.0, rows_after = 0.0;
    Vec conditions;                            // off-integer: K x nc at the final X
};

// integer centre.  Minv: N x (NB x NB); Wq: the data rule's normalised directions (rk); act: NB; dfix: --cond-fit's data fix or empty
void defect_int(const Vec& X0, const std::vector<Vec>& Minv, const std::vector<Vec>& Wq, const std::vector<char>& act, double Smax, int K,
                int NB, const Vec& e1, double R, int ndc, bool robust, const std::function<Vec(const Vec&)>& dfix, const Rows& rows, DefectOut& o);
// off-integer centre (Wq: normalised and phased, M2c's offl_start); ndef = max(--defect, 4)
void defect_off(const Vec& X0, const std::vector<Vec>& Minv, const std::vector<Vec>& Wq, int K, int NB, const Vec& e1, double R, int ndef,
                bool robust, const Rows& rows, DefectOut& o);
// Newton's sources at the nodes: nodes_of(rows(X_k)) - M_k(z) X_k (Ms: 3 x NB x NB, the re-linearised operator)
Vec newton_sources(const Vec& Xk, const Vec& Ms, int K, int NB, const Vec& e1, double R, const Rows& rows);
// Newton's decision after an iteration: continue (re-linearise) or stop
bool newton_continue(const std::vector<double>& newton_hist, double Smax, int newt, int newton_max, bool isint, int ndc);

}  // namespace st
}  // namespace mk

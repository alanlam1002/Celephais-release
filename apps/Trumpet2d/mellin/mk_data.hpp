// Mellin throat recursion, M2c (code round 463; research rounds 1018-1023): the DATA RULE and the CONDITIONS at a centre.
//
// spinning_trumpet/scripts/n2a_nl.py, after the contour step (M2b): at an integer centre with residue rank rk > 0 the residue amplitudes c
// are fixed -- the targets (Cbb's values at the centre, --tgt items, --notarget), the residue directions normalised to unit residue, the
// conditions (H_c@P0, E_sh^th@P1 and the removed Galerkin rows, moments 0 and 1) evaluated in full through the kernel; their tolerance tied to
// the measured truncation (the gate rows' largest live residual); the linearity test; Levenberg-Marquardt on [1e4 x the projected
// conditions; the data's equations], a Newton polish inside the constrained subspace, the classification (constrained / genuine directions,
// the genuine amplitudes); with --cond-fit nothing imposed (the data rule alone, linear) and the data fix (the data rule re-solved on the
// final coefficients, applied after every defect correction, M2d).  At an off-integer centre the family's datum: the directions normalised
// and phased, the start X0 + c_f W_0, and (without --cond-fit) the amplitude fixed by the l = 0 conditions when they respond.
//
// The conditions and the gate rows need the kernel: they reach this code as callbacks (Eval), built in mk_kernel.cpp's translation unit, so
// this file (compiled -ffp-contract=off, numpy's arithmetic as in M2b) never instantiates the kernel's inline code with other flags.
#pragma once
#include "mk_contour.hpp"

#include <functional>
#include <string>

namespace mk {
namespace dt {

using cd = std::complex<double>;
using Vec = std::vector<cd>;

// named constants (n2a_nl lines quoted)
constexpr double COND_W = 1e4;        // resid = np.r_[1e4 * (Uc.conj().T @ condf(c_)), dres(c_)]
constexpr double LIN_TOL = 1e-8;      // if lin_err < 1e-8 ...: the linear model
constexpr double JAC_H = 1e-3;        // hc_ = 1e-3 * (1 + np.abs(c_).max())
constexpr double LIN_STEP = 1e-2;     // dts_ = 1e-2 * (1 + np.abs(c).max()) * exp(1j arange(rk)) / sqrt(rk)
constexpr double MU0 = 1e-3, MU_MIN = 1e-15;
constexpr int LM_ITERS = 60, LM_INNER = 30, POLISH_ITERS = 20;
constexpr double STEP_STOP = 1e-13;   // float(np.abs(dc).max()) < 1e-13 * (1 + np.abs(c).max())
constexpr double SIZE_MIN = 1e-3;     // ssz = max(..., 1e-3)
constexpr double CONDTOL_MIN = 1e-14; // condtol = max(10 * tau / ssz, 1e-14)

struct Eval {
    std::function<Vec(const Vec& X)> cond;                   // cond_val(base, X, g): [S0 (nc), G0 (xrows), S1, G1]
    std::function<double(const Vec* X, double gg)> gate;    // gate_abs(X ? with_grade(base, X, g) : base, gg)
};

struct Hist {
    int kind = 0;                     // 0 LM step, 1 polish, 2 direct
    double res = 0.0, mu = 0.0, polish = 0.0;
    int rank = 0;
};
struct Fitted {
    cd amp;
    int cols[3] = {0, 0, 0};
    double l0_share = 0.0, l2_content = 0.0;
};

struct DataIn {
    int nn = 0, NB = 0, rk = 0;
    double R = 0.0;
    Vec e1, X0;                       // X0: N x NB
    std::vector<Vec> Wq;              // rk x (N x NB): the contour's directions (unnormalised)
    Vec tgtv, prevv, tg0;             // 2 NB
    std::vector<double> wt;           // 2 NB
    bool notgt = false, cond_fit = false, cond_full = false;
    double condtol_fixed = 0.0;       // --condtol (0: tied to the truncation)
    bool warm = false;
    Vec cwarm;
    int j_br0 = -1;                   // the BR0 column (alt_tau's b at the other roots)
    std::vector<int> lc;              // nn
    double g = 0.0;
};
struct DataOut {
    std::vector<Vec> Wq;              // normalised
    Vec At;                           // 2 NB x rk
    Vec c, F_start, F_final, Xg;
    std::vector<double> cond_sv;
    int cond_rank = 0;
    double lin_err = 0.0, tau = 0.0, size = 1.0, condtol = 0.0;
    std::vector<Hist> hist;
    std::vector<Fitted> fitted;
    // alt_tau (research round 1026: a LOGGED DIAGNOSTIC, outside every decision -- it monitors the b1 branch): with ONE constrained direction
    // v, H_c@P0 (moment 0) along c + tau v at 7 points, its quartic fit, the fit's other real roots, b (BR0's residue) and every condition row
    // (relative to its start value) there
    bool has_alt = false;
    int poly_deg = 0;
    double alt_b1 = 0.0;              // |BR0's residue per unit tau| (round 1027: v phased so that it is real positive)
    std::vector<double> poly, alt_tau, b_alt, tau_fit;
    std::vector<std::vector<double>> rows_at_alt;
};

// the targets at integer centre g (nb = 1): tgtv from the target series (Cbb: column -> the moment 0 at t = g + kofs(j)), the --tgt items
// matched on the column labels 'FAM L@tT', the weights and tg0.  Returns the number of hits; throws when hits > rk (a root outside the contour)
struct TgtIn {
    int nn = 0, rk = 0;
    double g = 0.0;
    std::vector<int> kofs, lc;
    std::vector<std::string> collab;  // 'PS0', 'BR2', ... (cl + lc)
    std::vector<int> cbb_col;         // the target series' entries (integer centres only)
    std::vector<double> cbb_t;
    Vec cbb_m0;
    std::vector<std::pair<std::string, double>> tgt;  // --tgt items, in order
    bool notgt = false;
};
int targets(const TgtIn& in, Vec& tgtv, Vec& prevv, std::vector<double>& wt, Vec& tg0, std::vector<std::pair<int, double>>& hit);

// tg_(X): moments 0 and 1 of the node values (2 NB)
Vec tg(const Vec& X, int NB, const Vec& e1, double R);

void data_rule(const DataIn& in, const Eval& ev, DataOut& o);
// --cond-fit's data fix (the data rule re-solved on X's coefficients)
Vec data_fix(const Vec& X, const DataIn& in, const DataOut& d);

// off-integer centre: the directions normalised and phased (largest u_g residue real positive), the start X0 + datum W_0, and without
// --cond-fit the amplitude fixed by the conditions when they respond (amp_fixed set, has_fixed true)
void offl_start(const Vec& X0, const std::vector<Vec>& Wq_raw, int nn, int NB, const Vec& e1, double R, bool labelled, double datum,
                bool cond_fit, const Eval& ev, std::vector<Vec>& Wq, Vec& Xg, bool& has_fixed, cd& amp_fixed);

}  // namespace dt
}  // namespace mk

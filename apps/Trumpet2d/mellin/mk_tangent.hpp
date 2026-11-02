// Mellin throat recursion, M3 (code round 463; research round 1027): the TANGENTS -- the recursion linearised about the final series.
//
// spinning_trumpet/scripts/n2a_nl.py fit_out (--tangent): the tangent of the series with respect to a parameter p, carried through the
// forward's centres (its last pass's order), with each centre's own contour inverse and data rule (the step state the forward kept):
//   tan_step     at centre g: the block's d u_g solves D rows_g[dC with the block] = 0 -- the start -M^-1 D rows of the stripped tangent
//                about the stripped series, then preconditioned Richardson (the defect correction's linearisation, the amplitudes held:
//                the update's components along the residue directions removed), the residue amplitudes' derivatives from the data rule
//                (integer centres: the weighted least squares on the u_g residues against the targets' derivative -- 1 on the seed's --tgt
//                column; off-integer: the family's datum, added once when the seed is that family); the stops measured against the tangent's
//                own scale (the absolute stall measure, research rounds 1012-1013)
// The rows' directional derivative D rows reaches this code as a callback (drows), built where the kernel is (mk_engine.hpp): the oracle's
// four-point stencil (batched) or the kernel's exact linear response.  Compiled -ffp-contract=off, numpy's arithmetic as in M2b-M2d.
#pragma once
#include "mk_steps.hpp"

#include <string>

namespace mk {
namespace tn {

using cd = std::complex<double>;
using Vec = std::vector<cd>;

// n2a_nl's step_state[ckey(g)] (the forward's last pass)
struct StepState {
    bool isint = false, kept_only = false;
    std::vector<Vec> Minv;                     // N x (NB x NB)
    std::vector<char> act;
    int rk = 0;
    std::vector<Vec> Wq;                       // rk x (N x NB): normalised (integer) / normalised and phased (off-integer)
    Vec X;                                     // N x NB: the step's final node values
    bool has_label = false;                    // off-integer: the family's label
    std::string label;
    bool has_src = false;                      // integer centres with residues: the data rule's weights and sources
    std::vector<double> wt;                    // 2 NB
    std::vector<std::string> src_tgt;          // 2 NB: the --tgt name of a hit column ('' elsewhere; nb = 1 has no 'prev' sources)
};

struct Seed {
    std::string kind;                          // data / tgt / lap / gauge
    std::string name;                          // 'PS0@t1' / 'BR0@t1' / 'lapse l4' / 'l2'
};

struct StepOut {
    Vec dX;
    std::vector<double> hh, hha;               // the Richardson updates: relative to the tangent's scale / absolute
};

// research rounds 1028-1031: the DIRECT solve at a family's own centre (n2a_nl fam_matrix / tan_step's direct branch, the oracle's default):
// the Richardson update is affine in the block's moments, upd(mu) = f0 + L (mu - mu_a); L (Kn x Kn, Kn = K NB; column k NB + j: moment k of
// column j) built once per family centre from the rows' four-point stencil about the step's evaluation point (es = h, -h, 2h, -2h on that
// moment), then -M^-1 and the hold; C (rk x Kn) the hold's constraint (moment 0 orthogonal to the residue directions' tg_dc, conjugated).
// Each step solves f0 + L delta = 0 with C delta = 0 by least squares in SCALED moments (moment m x R^m: the similarity D^-1 L D).
struct FamMap {
    Vec L, C;
    int rk = 0;
};
// rows4(cols): the rows projected at g (4 len(cols), K, NB) at the evaluation point with moment k of column j moved by es[b] (copy 4 qi + b)
void fam_matrix(const StepState& st, int K, int NB, int nn, const Vec& e1, double R, double h, int chunk,
                const std::function<Vec(const std::vector<std::pair<int, int>>&)>& rows4, FamMap& o);

constexpr int TAN_ITERS = 12;                  // opt('--tan-iters', 12)
constexpr double TAN_CTOL = 1e-11;             // opt('--tan-ctol', 1e-11)

// drows(dX): the rows' directional derivative at aligned g (K x NB, the active columns kept): dX given -- the tangent with the block set to
// dX about the step's evaluation point; dX null -- the base: the stripped tangent about the stripped series
void tan_step(const StepState& st, const Seed& seed, double sc, double ctol, int iters, int K, int NB, int nn, const Vec& e1, double R,
              const std::function<Vec(const Vec*)>& drows, StepOut& o, const FamMap* fm = nullptr);

}  // namespace tn
}  // namespace mk

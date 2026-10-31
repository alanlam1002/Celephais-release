// Mellin throat recursion, M2b (code round 463; research rounds 1018-1022): the CONTOUR INVERSE and the RESIDUES at a centre.
//
// spinning_trumpet/scripts/n2a_nl.py, the recursion after the operator build: the operator at the N contour nodes sigma = g + R e1_k,
// M(z) = Ms0 + z Ms1 + z^2 Ms2 (z = R e1_k); the active columns (g + kofs(j) >= 1: the cylinder is data); the null test of the scaled
// operator at node 0 (singular values < 1e-13 x the largest: kn) and, when kn > 0, the bordering by fixed unit vectors (pivoted QR on the
// null vectors picks the column, on the left null vectors the slack row); the scaled square inverse at every node; X0 = -M^-1 S at the
// nodes; the residue matrix Res = sum_k R/N e1_k M^-1_k, its SVD, its rank against the absolute reference rref = R max_k |M^-1_k|_2 at 1e-8
// (fixed at the centre's first determination: rk_fix); the residue directions Wq = M^-1 conj(v_q) and their moments 0 and 1 (the log's
// residue_dirs).  nodes_of / moments_of: the moments <-> node values of the series' columns at a centre.
//
// Arithmetic as numpy does it, so that given the oracle's inputs the outputs are BITWISE the oracle's (mk_contour.cpp is compiled with
// -ffp-contract=off): complex product (ar br - ai bi, ar bi + ai br); complex division by Smith's form with the reciprocal (numpy's
// loops: a real divisor r becomes a product with 1/r); integer powers by numpy's nc_pow (repeated squaring); matrix @ vector by MKL's zgemv
// as numpy's matmul calls it (column-major, transposed); reductions in numpy's order (Python's sum over nodes, sum(axis 0) row by row).
#pragma once
#include "mk_la.hpp"

#include <utility>
#include <vector>

namespace mk {
namespace ct {

using cd = std::complex<double>;

// the named thresholds (n2a_nl lines quoted)
constexpr double NULL_REL = 1e-13;   // kn = int((Se_ < 1e-13 * Se_[0]).sum())
constexpr double RANK_REL = 1e-8;    // rk = int((Sr > 1e-8 * rref).sum())
constexpr double LSPLIT = 1e-6;      // if lsplit > 1e-6: the slack row from the predictor's rows
constexpr double ACT_TOL = 1e-9;     // act: float(g) + b + kofs(j) >= 1 - 1e-9

cd mul(cd a, cd b);
cd div(cd a, cd b);
cd ipow(cd a, int n);

struct Contour {
    int NB = 0, N = 0, na = 0, kn = 0, rk_thr = 0, rk = 0;
    std::vector<char> act;
    std::vector<int> ia;                         // the active columns
    std::vector<std::pair<int, int>> bord;       // (row, column) of each bordering, full indices
    std::vector<std::vector<cd>> Mnod;           // N x (NB x NB): the operator at the nodes
    std::vector<std::vector<cd>> Minv;           // N x (NB x NB): the scaled inverse, zero off the active block
    std::vector<cd> X0;                          // N x NB
    std::vector<cd> Res;                         // NB x NB
    std::vector<double> Sr;                      // NB
    std::vector<cd> Vr;                          // NB x NB (rows: right singular vectors, conjugated as LAPACK's Vh)
    double rref = 0.0;
    std::vector<std::vector<cd>> Wq;             // rk x (N x NB)
    std::vector<cd> rdirs;                       // rk x 2NB: moments 0 and 1 of each Wq
};

// active columns at aligned g (nb = 1)
std::vector<char> active(double g, const std::vector<int>& kofs);
// moments (K x NB) -> node values (N x NB); node values -> moments
std::vector<cd> nodes_of(const std::vector<cd>& Mom, int K, int NB, const std::vector<cd>& e1, double R);
std::vector<cd> moments_of(const std::vector<cd>& X, int K, int NB, const std::vector<cd>& e1, double R);
// the contour step.  Ms: 3 x NB x NB (row-major); rw, cw: the row / column scales (nb = 1); Sn: N x NB; rkfix < 0: the rank by the
// threshold (and the caller records it as the centre's), else kept at rkfix
void contour(const std::vector<cd>& Ms, int NB, const std::vector<double>& rw, const std::vector<double>& cw, const std::vector<cd>& e1,
             double R, const std::vector<char>& act, const std::vector<cd>& Sn, int rkfix, Contour& o);

}  // namespace ct
}  // namespace mk

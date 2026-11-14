// Mellin throat recursion (code round 463; research rounds 1074 / 1089 / 1092, the forward-mode replay, STAGE 1(c), part 1): the CONTOUR
// STEP on duals -- ct::contour's outputs and their tangents along ND directions, given the operator's (dMs) and the sources' (dSn) tangents.
// The DECISIONS are the value's: the active columns, the bordering (its pivots), the residue rank.
//   the nodes' operator     dM_k = dMs0 + z_k dMs1 + z_k^2 dMs2
//   the inverses            dMinv_k = -Minv_k dM_k Minv_k   (the active block; the bordered inverse's top-left block obeys the same rule,
//                           its border rows / columns being constant; the row / column scalings cancel)
//   the start               dX0_k = -dMinv_k Sn_k - Minv_k dSn_k
//   the residue matrix      dRes = sum_k (R / N) e1_k dMinv_k
//   the residue directions  Res = U S V^H, the leading rk right singular vectors V_r: dV_r = (I - V_r V_r^H) dRes^H U_r S_r^-1 (the
//                           subspace's tangent, gauge V_r^H dV_r = 0; exact to O(s_(rk+1) / s_rk), the residue rank's own separation --
//                           the step uses the directions only through their span), then dW_q = dMinv_k v_q + Minv_k dv_q, v_q = V_r[:, q]
// Unit-tested against central differences of ct::contour (mk_contour_dual_test.cpp).
#pragma once
#include "mk_ad.hpp"
#include "mk_contour.hpp"

namespace mk {
namespace cdl {

using cd = std::complex<double>;
using Vec = std::vector<cd>;

struct ContourT {
    std::vector<std::vector<Vec>> dMinv;        // [k][q]: N x ND, NB x NB each
    std::vector<Vec> dX0;                       // [q]: N x NB
    std::vector<Vec> dRes;                      // [q]: NB x NB
    std::vector<std::vector<Vec>> dWq;          // [q][r]: ND x rk, N x NB each
    std::vector<Vec> V;                         // the value's directions v_r (NB each, Wq_r = Minv_k v_r)
};

inline Vec matmul(const Vec& A, const Vec& B, int n) { return ad::mul(A, B, n, n, n); }
inline Vec matvec(const Vec& A, const cd* x, int n) {
    Vec y(n, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) y[i] += A[size_t(i) * n + j] * x[j];
    return y;
}

// o: ct::contour's value (computed by the caller with the same arguments); dMs[q]: 3 x NB x NB; dSn[q]: N x NB (empty = zero)
inline void contour_t(const Vec& Ms, const std::vector<Vec>& dMs, int NB, const Vec& e1, double R, const Vec& Sn, const std::vector<Vec>& dSn,
                      const ct::Contour& o, int ND, ContourT& t) {
    (void)Ms;
    (void)Sn;
    int N = int(e1.size());
    size_t nm = size_t(NB) * NB;
    t = ContourT();
    t.dMinv.assign(N, std::vector<Vec>(ND));
    t.dX0.assign(ND, Vec(size_t(N) * NB, 0.0));
    t.dRes.assign(ND, Vec(nm, 0.0));
    for (int k = 0; k < N; k++) {
        cd zz = R * e1[k], z2 = zz * zz;
        for (int q = 0; q < ND; q++) {
            Vec dM(nm, 0.0);
            if (q < int(dMs.size()) && !dMs[q].empty())
                for (size_t i = 0; i < nm; i++) dM[i] = dMs[q][i] + zz * dMs[q][nm + i] + z2 * dMs[q][2 * nm + i];
            Vec dMi = matmul(matmul(o.Minv[k], dM, NB), o.Minv[k], NB);
            for (auto& v : dMi) v = -v;
            t.dMinv[k][q] = dMi;
            // dX0_k = -dMinv_k Sn_k - Minv_k dSn_k
            Vec a = matvec(dMi, &Sn[size_t(k) * NB], NB);
            Vec b = (q < int(dSn.size()) && !dSn[q].empty()) ? matvec(o.Minv[k], &dSn[q][size_t(k) * NB], NB) : Vec(NB, 0.0);
            for (int j = 0; j < NB; j++) t.dX0[q][size_t(k) * NB + j] = -a[j] - b[j];
            cd s = (R / N) * e1[k];
            for (size_t i = 0; i < nm; i++) t.dRes[q][i] += s * dMi[i];
        }
    }
    // the residue directions: Res = U S V^H (la::svd: o.Vr holds V^H's rows), the leading rk
    int rk = o.rk;
    t.dWq.assign(ND, std::vector<Vec>(rk));
    t.V.assign(rk, Vec(NB));
    if (!rk) return;
    std::vector<double> S;
    Vec U, Vh;
    la::svd(o.Res, NB, NB, S, &U, &Vh, true);
    for (int r = 0; r < rk; r++)
        for (int j = 0; j < NB; j++) t.V[r][j] = std::conj(Vh[size_t(r) * NB + j]);
    for (int q = 0; q < ND; q++) {
        // dV_r = (I - V V^H) dRes^H u_r / s_r
        std::vector<Vec> dv(rk, Vec(NB, 0.0));
        for (int r = 0; r < rk; r++) {
            Vec w(NB, 0.0);                                    // dRes^H u_r
            for (int i = 0; i < NB; i++)
                for (int j = 0; j < NB; j++) w[j] += std::conj(t.dRes[q][size_t(i) * NB + j]) * U[size_t(i) * NB + r];
            for (auto& x : w) x /= S[r];
            Vec pw(NB, 0.0);                                   // V V^H w
            for (int s_ = 0; s_ < rk; s_++) {
                cd c = 0.0;
                for (int j = 0; j < NB; j++) c += std::conj(t.V[s_][j]) * w[j];
                for (int j = 0; j < NB; j++) pw[j] += c * t.V[s_][j];
            }
            for (int j = 0; j < NB; j++) dv[r][j] = w[j] - pw[j];
        }
        for (int r = 0; r < rk; r++) {
            Vec W(size_t(N) * NB);
            for (int k = 0; k < N; k++) {
                Vec a = matvec(t.dMinv[k][q], t.V[r].data(), NB), b = matvec(o.Minv[k], dv[r].data(), NB);
                for (int j = 0; j < NB; j++) W[size_t(k) * NB + j] = a[j] + b[j];
            }
            t.dWq[q][r] = W;
        }
    }
}

}  // namespace cdl
}  // namespace mk

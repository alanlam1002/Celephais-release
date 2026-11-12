// Mellin throat recursion (code round 463; research round 1074): FORWARD-MODE DERIVATIVE RULES for the linear-algebra primitives of mk_la.
// Each rule takes the primitive's inputs and their tangents along ND directions (dA[k], dB[k], k < ND; an empty list = zero tangent) and
// returns the primitive's value (as mk_la computes it -- the same LAPACK calls) and its tangents.  The discrete decisions a primitive takes
// (the least-squares rank, the pivot order, which eigenvalue is which) are FROZEN at the base: a rule differentiates the computation as
// executed, never the decision.  Every rule is unit-tested against central finite differences in isolation (mk_ad_test.cpp).
//   primitive              rule
//   inv                    d(A^-1) = -A^-1 dA A^-1
//   solve                  dX = A^-1 (dB - dA X)
//   lstsq (rank r frozen)  X = A_r^+ B; dX = -A^+ dA X + A^+ A^+H dA^H (B - A X) + (I - A^+ A) dA^H A^+H X   (Golub-Pereyra on A_r;
//                          exact for full column rank, first order in sigma_{r+1} / sigma_r otherwise)
//   svd values             ds_i = Re(u_i^H dA v_i)                  (simple singular values)
//   null vector            A v = s v-direction smallest: dv = -A_{n-1}^+ (dA v - (u^H dA v) u) gauge-fixed: Im(v^H dv) = 0 and
//                          v^H dv = 0 (unit norm, phase held -- the caller's convention is applied to v and dv alike)
//   eigvals                dlam_i = w_i^H dA v_i / (w_i^H v_i)      (simple eigenvalues; left / right vectors from zgeev)
//   eiggen                 dlam_i = w_i^H (dA - lam_i dB) v_i / (w_i^H B v_i)   (zggev's left / right vectors)
//   qrpiv                  no derivative: the permutation is a decision (frozen)
#pragma once
#include "mk_la.hpp"

namespace mk {
namespace ad {

using cd = std::complex<double>;
using Mat = std::vector<cd>;          // row-major
using Tans = std::vector<Mat>;        // one tangent per direction

// C (m x p) = A (m x n) B (n x p)
inline Mat mul(const Mat& A, const Mat& B, int m, int n, int p) {
    Mat C(size_t(m) * p, 0.0);
    for (int i = 0; i < m; i++)
        for (int k = 0; k < n; k++) {
            cd a = A[size_t(i) * n + k];
            if (a == cd(0.0)) continue;
            for (int j = 0; j < p; j++) C[size_t(i) * p + j] += a * B[size_t(k) * p + j];
        }
    return C;
}
// A^H (n x m) of A (m x n)
inline Mat herm(const Mat& A, int m, int n) {
    Mat H(size_t(n) * m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) H[size_t(j) * m + i] = std::conj(A[size_t(i) * n + j]);
    return H;
}
inline void axpy(Mat& Y, cd a, const Mat& X) { for (size_t i = 0; i < Y.size(); i++) Y[i] += a * X[i]; }
inline const Mat* at(const Tans& T, size_t k) { return k < T.size() && !T[k].empty() ? &T[k] : nullptr; }

// ---------------------------------------------------------------------------------------------------------------- inv, solve
inline Mat inv_t(const Mat& A, int n, const Tans& dA, Tans& dAi) {
    Mat Ai = la::inv(A, n);
    dAi.assign(dA.size(), Mat());
    for (size_t k = 0; k < dA.size(); k++) {
        if (!at(dA, k)) { dAi[k].assign(size_t(n) * n, 0.0); continue; }
        Mat T = mul(mul(Ai, dA[k], n, n, n), Ai, n, n, n);
        for (auto& v : T) v = -v;
        dAi[k] = T;
    }
    return Ai;
}

// A X = B: dX = A^-1 (dB - dA X); ND = max(|dA|, |dB|)
inline Mat solve_t(const Mat& A, int n, const Mat& B, int nrhs, const Tans& dA, const Tans& dB, Tans& dX) {
    Mat X = la::solve(A, n, B, nrhs);
    size_t nd = std::max(dA.size(), dB.size());
    dX.assign(nd, Mat());
    for (size_t k = 0; k < nd; k++) {
        Mat R(size_t(n) * nrhs, 0.0);
        if (const Mat* b = at(dB, k)) R = *b;
        if (const Mat* a = at(dA, k)) axpy(R, -1.0, mul(*a, X, n, n, nrhs));
        dX[k] = la::solve(A, n, R, nrhs);
    }
    return X;
}

// ---------------------------------------------------------------------------------------------------------------- least squares
// the rank-r pseudo-inverse from the SVD (r < 0: numpy's rcond = eps max(m, n) -- gelsd's own rank)
inline Mat pinv_r(const Mat& A, int m, int n, int& r, double rcond = -1.0) {
    std::vector<double> s;
    Mat U, Vh;
    la::svd(A, m, n, s, &U, &Vh, false);
    int k = std::min(m, n);
    if (r < 0) {
        double tol = (rcond < 0 ? std::numeric_limits<double>::epsilon() * std::max(m, n) : rcond) * (k ? s[0] : 0.0);
        r = 0;
        while (r < k && s[r] > tol) r++;
    }
    Mat P(size_t(n) * m, 0.0);           // V_r S_r^-1 U_r^H
    for (int q = 0; q < r; q++)
        for (int i = 0; i < n; i++) {
            cd v = std::conj(Vh[size_t(q) * n + i]) / s[q];
            for (int j = 0; j < m; j++) P[size_t(i) * m + j] += v * std::conj(U[size_t(j) * k + q]);
        }
    return P;
}

// X = lstsq(A, B) (the value as mk_la's gelsd); the rank FROZEN: rank_in >= 0 imposes it, else gelsd's (returned in *rank_out)
inline Mat lstsq_t(const Mat& A, int m, int n, const Mat& B, int nrhs, const Tans& dA, const Tans& dB, Tans& dX, int rank_in = -1,
                   int* rank_out = nullptr) {
    int rk = 0;
    Mat X = la::lstsq(A, m, n, B, nrhs, nullptr, &rk);
    int r = rank_in >= 0 ? rank_in : rk;
    if (rank_out) *rank_out = r;
    Mat Ap = pinv_r(A, m, n, r);                       // n x m
    if (rank_in >= 0) X = mul(Ap, B, n, m, nrhs);      // (an imposed rank: the value at that rank)
    Mat ApH = herm(Ap, n, m);                          // m x n
    Mat Res = B;                                       // B - A X
    axpy(Res, -1.0, mul(A, X, m, n, nrhs));
    Mat PA = mul(Ap, A, n, m, n);                      // A^+ A (n x n)
    size_t nd = std::max(dA.size(), dB.size());
    dX.assign(nd, Mat());
    for (size_t k = 0; k < nd; k++) {
        Mat D(size_t(n) * nrhs, 0.0);
        if (const Mat* b = at(dB, k)) D = mul(Ap, *b, n, m, nrhs);
        if (const Mat* a = at(dA, k)) {
            Mat aH = herm(*a, m, n);                   // n x m
            axpy(D, -1.0, mul(Ap, mul(*a, X, m, n, nrhs), n, m, nrhs));
            axpy(D, 1.0, mul(Ap, mul(ApH, mul(aH, Res, n, m, nrhs), m, n, nrhs), n, m, nrhs));
            Mat T = mul(aH, mul(ApH, X, m, n, nrhs), n, m, nrhs);   // dA^H A^+H X  (n x nrhs)
            axpy(D, 1.0, T);
            axpy(D, -1.0, mul(PA, T, n, n, nrhs));
        }
        dX[k] = D;
    }
    return X;
}

// ---------------------------------------------------------------------------------------------------------------- SVD
// singular values and their tangents (simple values): ds_i = Re(u_i^H dA v_i)
inline std::vector<double> svals_t(const Mat& A, int m, int n, const Tans& dA, std::vector<std::vector<double>>& ds) {
    std::vector<double> s;
    Mat U, Vh;
    la::svd(A, m, n, s, &U, &Vh, false);
    int k = std::min(m, n);
    ds.assign(dA.size(), std::vector<double>(k, 0.0));
    for (size_t d = 0; d < dA.size(); d++) {
        if (!at(dA, d)) continue;
        for (int q = 0; q < k; q++) {
            cd acc = 0.0;
            for (int i = 0; i < m; i++) {
                cd row = 0.0;
                for (int j = 0; j < n; j++) row += dA[d][size_t(i) * n + j] * std::conj(Vh[size_t(q) * n + j]);
                acc += std::conj(U[size_t(i) * k + q]) * row;
            }
            ds[d][q] = acc.real();
        }
    }
    return s;
}

// the right singular vector of the smallest singular value (A m x n, m >= n -- a null vector of a nearly singular square or tall A) and its
// tangent: (A^H A) v = s^2 v; dv = -(A^H A - s^2)^+_{n-1} (dA^H A + A^H dA) v restricted to v's complement (gauge v^H dv = 0); the phase
// convention: v's largest-modulus entry real positive at the base, held to first order (Im(dv_p / v_p) = 0 at that entry p)
inline Mat nullvec_t(const Mat& A, int m, int n, const Tans& dA, Tans& dv, double* smin = nullptr) {
    std::vector<double> s;
    Mat U, Vh;
    la::svd(A, m, n, s, &U, &Vh, true);
    Mat v(n);
    for (int j = 0; j < n; j++) v[j] = std::conj(Vh[size_t(n - 1) * n + j]);
    int p = 0;
    for (int j = 1; j < n; j++) if (std::abs(v[j]) > std::abs(v[p])) p = j;
    cd ph = std::abs(v[p]) / v[p];
    for (auto& x : v) x *= ph;
    double sn = n - 1 < int(s.size()) ? s[n - 1] : 0.0;
    if (smin) *smin = sn;
    dv.assign(dA.size(), Mat());
    for (size_t d = 0; d < dA.size(); d++) {
        Mat t(n, 0.0);
        if (const Mat* a = at(dA, d)) {
            // g = (dA^H A + A^H dA) v ; dv = - sum_{q < n-1} vq (vq^H g) / (s_q^2 - s_n^2)
            Mat Av = mul(A, v, m, n, 1), dAv = mul(*a, v, m, n, 1);
            Mat g = mul(herm(*a, m, n), Av, n, m, 1);
            axpy(g, 1.0, mul(herm(A, m, n), dAv, n, m, 1));
            for (int q = 0; q < n - 1; q++) {
                cd c = 0.0;
                for (int j = 0; j < n; j++) c += Vh[size_t(q) * n + j] * g[j];      // vq^H g  (Vh row q = vq^H)
                double den = s[q] * s[q] - sn * sn;
                for (int j = 0; j < n; j++) t[j] -= std::conj(Vh[size_t(q) * n + j]) * c / den;
            }
            // the phase: Im(t_p / v_p) = 0 -- add i alpha v with alpha chosen so (removes the gauge freedom the formula leaves at 0)
            double alpha = -(t[p] / v[p]).imag();
            for (int j = 0; j < n; j++) t[j] += cd(0.0, alpha) * v[j];
        }
        dv[d] = t;
    }
    return v;
}

// ---------------------------------------------------------------------------------------------------------------- eigenvalues
// eigenvalues (zgeev, as la::eigvals) and their tangents: left / right vectors from one zgeev 'V','V'
inline std::vector<cd> eigvals_t(const Mat& A, int n, const Tans& dA, Tans& dl) {
    Mat a = A, vl(size_t(n) * n), vr(size_t(n) * n);
    std::vector<cd> w(n);
    la::check(LAPACKE_zgeev(LAPACK_ROW_MAJOR, 'V', 'V', n, a.data(), n, w.data(), vl.data(), n, vr.data(), n), "zgeev (V)");
    dl.assign(dA.size(), Mat(n, 0.0));
    for (size_t d = 0; d < dA.size(); d++) {
        if (!at(dA, d)) continue;
        for (int i = 0; i < n; i++) {
            cd num = 0.0, den = 0.0;
            for (int r = 0; r < n; r++) {
                cd row = 0.0;
                for (int c = 0; c < n; c++) row += dA[d][size_t(r) * n + c] * vr[size_t(c) * n + i];
                num += std::conj(vl[size_t(r) * n + i]) * row;
                den += std::conj(vl[size_t(r) * n + i]) * vr[size_t(r) * n + i];
            }
            dl[d][i] = num / den;
        }
    }
    return w;
}

// generalized eigenvalues lam = alpha / beta of (A, B) (zggev, as la::eiggen) and their tangents: dlam = w^H (dA - lam dB) v / (w^H B v)
inline std::vector<cd> eiggen_t(const Mat& A, const Mat& B, int n, const Tans& dA, const Tans& dB, Tans& dl) {
    Mat a = A, b = B, vl(size_t(n) * n), vr(size_t(n) * n);
    std::vector<cd> al(n), be(n);
    la::check(LAPACKE_zggev(LAPACK_ROW_MAJOR, 'V', 'V', n, a.data(), n, b.data(), n, al.data(), be.data(), vl.data(), n, vr.data(), n), "zggev (V)");
    std::vector<cd> lam(n);
    for (int i = 0; i < n; i++) lam[i] = be[i] == cd(0.0) ? cd(std::numeric_limits<double>::infinity(), 0.0) : al[i] / be[i];
    size_t nd = std::max(dA.size(), dB.size());
    dl.assign(nd, Mat(n, 0.0));
    for (size_t d = 0; d < nd; d++) {
        const Mat* pa = at(dA, d);
        const Mat* pb = at(dB, d);
        for (int i = 0; i < n; i++) {
            if (!std::isfinite(lam[i].real())) { dl[d][i] = 0.0; continue; }
            cd num = 0.0, den = 0.0;
            for (int r = 0; r < n; r++) {
                cd row = 0.0, rowB = 0.0;
                for (int c = 0; c < n; c++) {
                    cd vc = vr[size_t(c) * n + i];
                    cd m_ = (pa ? (*pa)[size_t(r) * n + c] : cd(0.0)) - lam[i] * (pb ? (*pb)[size_t(r) * n + c] : cd(0.0));
                    row += m_ * vc;
                    rowB += B[size_t(r) * n + c] * vc;
                }
                num += std::conj(vl[size_t(r) * n + i]) * row;
                den += std::conj(vl[size_t(r) * n + i]) * rowB;
            }
            dl[d][i] = num / den;
        }
    }
    return lam;
}

}  // namespace ad
}  // namespace mk

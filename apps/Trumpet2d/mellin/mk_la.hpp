// Mellin throat recursion (code round 463 M2; research rounds 1018-1020): THE ONE LINEAR-ALGEBRA LIBRARY -- Intel MKL LAPACK through
// LAPACKE, routine for routine as numpy / scipy call it (NOTES_M2_DESIGN.md section 4), so every decision taken from singular values sees the
// same kernels as the oracle when linked to numpy's MKL (anaconda 2021.05: MKL 2021.2; build option MK_MKL=numpy), or Kadath's (oneAPI 2025.2;
// MK_MKL=kadath) for the two-library check and M4.  Matrices are row-major std::vector<cd> (LAPACKE_ROW_MAJOR: LAPACKE transposes into the
// column-major layout numpy also hands LAPACK); workspaces from LAPACK's own optimal query (numpy's), so blocked algorithms block alike.
//   numpy / scipy                         here
//   np.linalg.inv                         inv        zgesv with the identity
//   np.linalg.solve                       solve      zgesv
//   np.linalg.svd (full / thin / no uv)   svd        zgesdd, jobz 'A' / 'S' / 'N'
//   np.linalg.lstsq (rcond=None)          lstsq      zgelsd, rcond = eps x max(m, n)
//   scipy.linalg.qr(pivoting=True)[2]     qrpiv      zgeqp3 (the column permutation)
//   scipy.linalg.eig(A, B) eigenvalues    eiggen     zggev, right vectors on (scipy's default), alpha / beta (beta = 0 -> inf)
#pragma once
#include <complex>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#define MKL_Complex16 std::complex<double>
#include <mkl_lapacke.h>

namespace mk {
namespace la {

using cd = std::complex<double>;

inline void check(lapack_int info, const char* what) {
    if (info != 0) throw std::runtime_error(std::string("mk::la: ") + what + " failed, info " + std::to_string(info));
}

// A^-1 (n x n)
inline std::vector<cd> inv(std::vector<cd> A, int n) {
    std::vector<cd> B(size_t(n) * n, 0.0);
    for (int i = 0; i < n; i++) B[size_t(i) * n + i] = 1.0;
    std::vector<lapack_int> piv(n);
    check(LAPACKE_zgesv(LAPACK_ROW_MAJOR, n, n, A.data(), n, piv.data(), B.data(), n), "zgesv (inv)");
    return B;
}

// A X = B, A n x n, B n x nrhs
inline std::vector<cd> solve(std::vector<cd> A, int n, std::vector<cd> B, int nrhs) {
    std::vector<lapack_int> piv(n);
    check(LAPACKE_zgesv(LAPACK_ROW_MAJOR, n, nrhs, A.data(), n, piv.data(), B.data(), nrhs), "zgesv");
    return B;
}

// the SVD: s (min(m, n)); with uv: U (m x m or m x k), Vh (n x n or k x n) by jobz 'A' / 'S'
inline void svd(std::vector<cd> A, int m, int n, std::vector<double>& s, std::vector<cd>* U = nullptr, std::vector<cd>* Vh = nullptr,
                bool full = true) {
    int k = std::min(m, n);
    s.assign(k, 0.0);
    if (!U) {
        std::vector<cd> du(std::max(m, 1)), dv(std::max(n, 1));      // not referenced by jobz 'N'; MKL's LAPACKE still checks ldvt >= n
        check(LAPACKE_zgesdd(LAPACK_ROW_MAJOR, 'N', m, n, A.data(), n, s.data(), du.data(), std::max(m, 1), dv.data(), std::max(n, 1)), "zgesdd (N)");
        return;
    }
    int ucols = full ? m : k, vrows = full ? n : k;
    U->assign(size_t(m) * ucols, 0.0);
    Vh->assign(size_t(vrows) * n, 0.0);
    check(LAPACKE_zgesdd(LAPACK_ROW_MAJOR, full ? 'A' : 'S', m, n, A.data(), n, s.data(), U->data(), ucols, Vh->data(), n), "zgesdd");
}

// least squares (numpy lstsq, rcond=None): X (n x nrhs), the singular values, the rank
inline std::vector<cd> lstsq(std::vector<cd> A, int m, int n, const std::vector<cd>& B, int nrhs, std::vector<double>* sv = nullptr,
                             int* rank = nullptr) {
    int ld = std::max(m, n);
    std::vector<cd> Bx(size_t(ld) * nrhs, 0.0);
    for (int i = 0; i < m; i++)
        for (int r = 0; r < nrhs; r++) Bx[size_t(i) * nrhs + r] = B[size_t(i) * nrhs + r];
    std::vector<double> s(std::min(m, n));
    lapack_int rk = 0;
    double rcond = std::numeric_limits<double>::epsilon() * std::max(m, n);
    check(LAPACKE_zgelsd(LAPACK_ROW_MAJOR, m, n, nrhs, A.data(), n, Bx.data(), nrhs, s.data(), rcond, &rk), "zgelsd");
    if (sv) *sv = s;
    if (rank) *rank = int(rk);
    Bx.resize(size_t(n) * nrhs);
    return Bx;
}

// QR with column pivoting: the permutation (0-based, scipy's P)
inline std::vector<int> qrpiv(std::vector<cd> A, int m, int n) {
    std::vector<lapack_int> jpvt(n, 0);
    std::vector<cd> tau(std::min(m, n));
    check(LAPACKE_zgeqp3(LAPACK_ROW_MAJOR, m, n, A.data(), n, jpvt.data(), tau.data()), "zgeqp3");
    std::vector<int> p(n);
    for (int i = 0; i < n; i++) p[i] = int(jpvt[i]) - 1;
    return p;
}

// generalized eigenvalues of (A, B), n x n: alpha / beta (scipy.linalg.eig(A, B, right=False))
inline std::vector<cd> eiggen(std::vector<cd> A, std::vector<cd> B, int n) {
    std::vector<cd> alpha(n), beta(n), vl(1), vr(size_t(n) * n);     // right vectors requested, as scipy's default (right=True)
    check(LAPACKE_zggev(LAPACK_ROW_MAJOR, 'N', 'V', n, A.data(), n, B.data(), n, alpha.data(), beta.data(), vl.data(), 1, vr.data(), n), "zggev");
    std::vector<cd> ev(n);
    for (int i = 0; i < n; i++)
        ev[i] = beta[i] == cd(0.0) ? cd(std::numeric_limits<double>::infinity(), 0.0) : alpha[i] / beta[i];
    return ev;
}

}  // namespace la
}  // namespace mk

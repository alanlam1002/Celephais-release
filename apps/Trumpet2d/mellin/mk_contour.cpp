// Mellin throat recursion, M2b (code round 463): the contour inverse and the residues (mk_contour.hpp), and their C ABI.
// Compiled with -ffp-contract=off (CMakeLists): numpy's loops do not fuse, and the outputs are compared bitwise.
#include "mk_contour.hpp"

#include <mkl_cblas.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace mk {
namespace ct {

// numpy's complex product (umath loops, scalarmath): (ar br - ai bi, ar bi + ai br)
cd mul(cd a, cd b) { return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()); }

// numpy's complex division (umath loops.c.src, Smith's form with the reciprocal)
cd div(cd a, cd b) {
    double ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
    double abr = std::fabs(br), abi = std::fabs(bi);
    if (abr >= abi) {
        if (abr == 0 && abi == 0) return cd(ar / abr, ai / abi);
        double rat = bi / br, scl = 1.0 / (br + bi * rat);
        return cd((ar + ai * rat) * scl, (ai - ar * rat) * scl);
    }
    double rat = br / bi, scl = 1.0 / (bi + br * rat);
    return cd((ar * rat + ai) * scl, (ai * rat - ar) * scl);
}

// numpy's nc_pow for an integer exponent 1 <= n < 100 (funcs.inc.src): n = 1, 2, 3 unrolled, else repeated squaring
cd ipow(cd a, int n) {
    if (n == 1) return a;
    if (n == 2) return mul(a, a);
    if (n == 3) return mul(a, mul(a, a));
    if (n < 1 || n >= 100) throw std::runtime_error("mk::ct::ipow: exponent outside 1..99");
    cd aa(1.0, 0.0), p = a;
    int mask = 1;
    while (true) {
        if (n & mask) aa = mul(aa, p);
        mask <<= 1;
        if (n < mask || mask <= 0) break;
        p = mul(p, p);
    }
    return aa;
}

// a real scalar times a complex one, as numpy does it (the real cast to complex)
static inline cd rmul(double r, cd z) { return mul(cd(r, 0.0), z); }

// numpy matmul (n x n C-contiguous) @ (n): cblas_zgemv(ColMajor, Trans, n, n, 1, A, n, x, 1, 0, y, 1)
static std::vector<cd> gemv(const std::vector<cd>& A, const cd* x, int n) {
    std::vector<cd> y(n);
    const cd one(1.0, 0.0), zero(0.0, 0.0);
    cblas_zgemv(CblasColMajor, CblasTrans, n, n, &one, A.data(), n, x, 1, &zero, y.data(), 1);
    return y;
}

std::vector<char> active(double g, const std::vector<int>& kofs) {
    std::vector<char> act(kofs.size());
    for (size_t j = 0; j < kofs.size(); j++) act[j] = (g + 0.0) + kofs[j] >= 1 - ACT_TOL;
    return act;
}

// n2a_nl.nodes_of: sum(Mom[m][None, :] / z_[:, None] ** (m + 1) for m in range(K)), z_ = R e1
std::vector<cd> nodes_of(const std::vector<cd>& Mom, int K, int NB, const std::vector<cd>& e1, double R) {
    int N = int(e1.size());
    std::vector<cd> out(size_t(N) * NB, cd(0.0, 0.0));
    for (int k = 0; k < N; k++) {
        cd z = rmul(R, e1[k]);
        for (int m = 0; m < K; m++) {
            cd zp = ipow(z, m + 1);
            for (int c = 0; c < NB; c++) {
                cd t = div(Mom[size_t(m) * NB + c], zp);
                cd& o = out[size_t(k) * NB + c];
                o = m == 0 ? t : o + t;
            }
        }
    }
    return out;
}

// n2a_nl.moments_of: [(z_[:, None] ** (m + 1) * X).sum(0) / N for m in range(K)]
std::vector<cd> moments_of(const std::vector<cd>& X, int K, int NB, const std::vector<cd>& e1, double R) {
    int N = int(e1.size());
    std::vector<cd> out(size_t(K) * NB);
    for (int m = 0; m < K; m++) {
        std::vector<cd> acc(NB);
        for (int k = 0; k < N; k++) {
            cd zp = ipow(rmul(R, e1[k]), m + 1);
            for (int c = 0; c < NB; c++) {
                cd t = mul(zp, X[size_t(k) * NB + c]);
                acc[c] = k == 0 ? t : acc[c] + t;
            }
        }
        for (int c = 0; c < NB; c++) out[size_t(m) * NB + c] = div(acc[c], cd(double(N), 0.0));
    }
    return out;
}

void contour(const std::vector<cd>& Ms, int NB, const std::vector<double>& rw, const std::vector<double>& cw, const std::vector<cd>& e1,
             double R, const std::vector<char>& act, const std::vector<cd>& Sn, int rkfix, Contour& o) {
    int N = int(e1.size());
    size_t nm = size_t(NB) * NB;
    o = Contour();
    o.NB = NB; o.N = N; o.act = act;
    for (int j = 0; j < NB; j++)
        if (act[j]) o.ia.push_back(j);
    int na = o.na = int(o.ia.size());
    // the operator at the nodes: Ms[0] + zz * Ms[1] + zz * zz * Ms[2]
    o.Mnod.resize(N);
    for (int k = 0; k < N; k++) {
        cd zz = rmul(R, e1[k]), z2 = mul(zz, zz);
        std::vector<cd>& M = o.Mnod[k];
        M.resize(nm);
        for (size_t i = 0; i < nm; i++) M[i] = (Ms[i] + mul(zz, Ms[nm + i])) + mul(z2, Ms[2 * nm + i]);
    }
    // the scaled active block: (Mk / rwb[:, None] / cwb[None, :])[np.ix_(ia, ia)]
    auto scaled = [&](const std::vector<cd>& M) {
        std::vector<cd> Me(size_t(na) * na);
        for (int a = 0; a < na; a++)
            for (int b = 0; b < na; b++) {
                int i = o.ia[a], j = o.ia[b];
                Me[size_t(a) * na + b] = div(div(M[size_t(i) * NB + j], cd(rw[i], 0.0)), cd(cw[j], 0.0));
            }
        return Me;
    };
    // the null test at node 0 and the bordering (nb = 1: the slack row from the left null vectors, the column from the right ones)
    std::vector<std::pair<int, int>> bord;
    {
        std::vector<double> Se;
        std::vector<cd> Ue, Vhe;
        if (na) la::svd(scaled(o.Mnod[0]), na, na, Se, &Ue, &Vhe, true);
        int kn = 0;
        for (double s : Se) kn += s < NULL_REL * Se[0];
        if (na == 0) kn = 0;
        o.kn = kn;
        if (kn) {
            std::vector<cd> Vt(size_t(kn) * na), Wt(size_t(kn) * na);      // Vn.T.conj() = Vhe[-kn:]; Wn.T.conj() = Ue[:, -kn:].T.conj()
            for (int q = 0; q < kn; q++)
                for (int a = 0; a < na; a++) {
                    Vt[size_t(q) * na + a] = Vhe[size_t(na - kn + q) * na + a];
                    Wt[size_t(q) * na + a] = std::conj(Ue[size_t(a) * na + (na - kn + q)]);
                }
            auto cpiv = la::qrpiv(Vt, kn, na), rpiv = la::qrpiv(Wt, kn, na);
            for (int q = 0; q < kn; q++) bord.emplace_back(rpiv[q], cpiv[q]);
        }
    }
    // the scaled inverse at every node, bordered if kn
    o.Minv.resize(N);
    int nbd = int(bord.size());
    for (int k = 0; k < N; k++) {
        std::vector<cd> Me = scaled(o.Mnod[k]), Mi_r(size_t(na) * na);
        if (nbd) {
            int nt = na + nbd;
            std::vector<cd> Bt(size_t(nt) * nt, cd(0.0, 0.0));
            for (int a = 0; a < na; a++)
                for (int b = 0; b < na; b++) Bt[size_t(a) * nt + b] = Me[size_t(a) * na + b];
            for (int q = 0; q < nbd; q++) {
                Bt[size_t(bord[q].first) * nt + na + q] = 1.0;
                Bt[size_t(na + q) * nt + bord[q].second] = 1.0;
            }
            auto Bi = la::inv(Bt, nt);
            for (int a = 0; a < na; a++)
                for (int b = 0; b < na; b++) Mi_r[size_t(a) * na + b] = Bi[size_t(a) * nt + b];
        } else {
            Mi_r = la::inv(Me, na);
        }
        std::vector<cd> Mi(nm, cd(0.0, 0.0));
        for (int a = 0; a < na; a++)
            for (int b = 0; b < na; b++) Mi[size_t(o.ia[a]) * NB + o.ia[b]] = Mi_r[size_t(a) * na + b];
        for (int i = 0; i < NB; i++)                                     // Mi_ / cwb[:, None] / rwb[None, :]
            for (int j = 0; j < NB; j++) {
                cd& v = Mi[size_t(i) * NB + j];
                v = div(div(v, cd(cw[i], 0.0)), cd(rw[j], 0.0));
            }
        o.Minv[k] = std::move(Mi);
    }
    for (auto& b : bord) o.bord.emplace_back(o.ia[b.first], o.ia[b.second]);
    // X0 = -Minv[k] @ Sn[k]
    o.X0.resize(size_t(N) * NB);
    for (int k = 0; k < N; k++) {
        std::vector<cd> nM(nm);
        for (size_t i = 0; i < nm; i++) nM[i] = -o.Minv[k][i];
        auto y = gemv(nM, &Sn[size_t(k) * NB], NB);
        std::copy(y.begin(), y.end(), o.X0.begin() + long(k) * NB);
    }
    // Res = sum(R / N * e1[k] * Minv[k] for k in range(N))
    o.Res.assign(nm, cd(0.0, 0.0));
    for (int k = 0; k < N; k++) {
        cd s = rmul(R / N, e1[k]);
        for (size_t i = 0; i < nm; i++) o.Res[i] = o.Res[i] + mul(s, o.Minv[k][i]);
    }
    std::vector<cd> Ur;
    la::svd(o.Res, NB, NB, o.Sr, &Ur, &o.Vr, true);
    // rref = R * max(np.linalg.norm(Mi, 2) for Mi in Minv); rk against 1e-8 rref, fixed at the centre's first determination
    double mx = 0.0;
    for (int k = 0; k < N; k++) {
        std::vector<double> s;
        la::svd(o.Minv[k], NB, NB, s);
        if (k == 0 || s[0] > mx) mx = s[0];
    }
    o.rref = R * mx;
    o.rk_thr = 0;
    for (double s : o.Sr) o.rk_thr += s > RANK_REL * o.rref;
    o.rk = rkfix >= 0 ? rkfix : o.rk_thr;
    // Wq = [np.array([Minv[k] @ Vr[q].conj() for k in range(N)]) for q in range(rk)], and their moments 0, 1 (the log's residue_dirs)
    o.Wq.resize(o.rk);
    o.rdirs.assign(size_t(o.rk) * 2 * NB, cd(0.0, 0.0));
    double a0 = R / N, a1 = R * R / N;
    for (int q = 0; q < o.rk; q++) {
        std::vector<cd> v(NB);
        for (int j = 0; j < NB; j++) v[j] = std::conj(o.Vr[size_t(q) * NB + j]);
        std::vector<cd>& W = o.Wq[q];
        W.resize(size_t(N) * NB);
        for (int k = 0; k < N; k++) {
            auto y = gemv(o.Minv[k], v.data(), NB);
            std::copy(y.begin(), y.end(), W.begin() + long(k) * NB);
        }
        cd* r0 = &o.rdirs[size_t(q) * 2 * NB];
        cd* r1 = r0 + NB;
        for (int k = 0; k < N; k++) {
            cd s0 = rmul(a0, e1[k]), s1 = rmul(a1, mul(e1[k], e1[k]));
            for (int j = 0; j < NB; j++) {
                cd t0 = mul(s0, W[size_t(k) * NB + j]), t1 = mul(s1, W[size_t(k) * NB + j]);
                r0[j] = k == 0 ? t0 : r0[j] + t0;
                r1[j] = k == 0 ? t1 : r1[j] + t1;
            }
        }
    }
}

}  // namespace ct
}  // namespace mk

// ---- C ABI (complex arrays as interleaved doubles, row-major) ----
namespace {
using cd = std::complex<double>;
thread_local std::string g_err;
std::vector<cd> cv(const double* p, size_t n) {
    std::vector<cd> v(n);
    for (size_t i = 0; i < n; i++) v[i] = cd(p[2 * i], p[2 * i + 1]);
    return v;
}
void put(const std::vector<cd>& v, double* p) {
    for (size_t i = 0; i < v.size(); i++) { p[2 * i] = v[i].real(); p[2 * i + 1] = v[i].imag(); }
}
}  // namespace

extern "C" {

const char* mk_ct_error() { return g_err.c_str(); }

int mk_nodes_of(const double* Mom, int K, int NB, int N, const double* e1, double R, double* out) {
    try { put(mk::ct::nodes_of(cv(Mom, size_t(K) * NB), K, NB, cv(e1, N), R), out); return 0; }
    catch (const std::exception& ex) { g_err = ex.what(); return 1; }
}

int mk_moments_of(const double* X, int K, int NB, int N, const double* e1, double R, double* out) {
    try { put(mk::ct::moments_of(cv(X, size_t(N) * NB), K, NB, cv(e1, N), R), out); return 0; }
    catch (const std::exception& ex) { g_err = ex.what(); return 1; }
}

// the contour step at aligned g.  Outputs: act (NB), kn, nbord and bord (2 x nbord, cap NB), X0 (N x NB), Res (NB x NB), Sr (NB), rref,
// rk_thr, rk, rdirs (rk x 2NB, cap NB x 2NB), Wq (rk x N x NB, cap NB x N x NB; may be null), Minv (N x NB x NB; may be null)
int mk_contour(int NB, int N, double R, double g, const int* kofs, const double* Ms, const double* rw, const double* cw, const double* e1,
               const double* Sn, int rkfix, int* act, int* kn, int* nbord, int* bord, double* X0, double* Res, double* Sr, double* rref,
               int* rk_thr, int* rk, double* rdirs, double* Wq, double* Minv) {
    try {
        mk::ct::Contour o;
        auto a = mk::ct::active(g, std::vector<int>(kofs, kofs + NB));
        mk::ct::contour(cv(Ms, size_t(3) * NB * NB), NB, std::vector<double>(rw, rw + NB), std::vector<double>(cw, cw + NB), cv(e1, N), R, a,
                        cv(Sn, size_t(N) * NB), rkfix, o);
        for (int j = 0; j < NB; j++) act[j] = o.act[j];
        *kn = o.kn;
        *nbord = int(o.bord.size());
        for (size_t q = 0; q < o.bord.size(); q++) { bord[2 * q] = o.bord[q].first; bord[2 * q + 1] = o.bord[q].second; }
        put(o.X0, X0);
        put(o.Res, Res);
        for (int j = 0; j < NB; j++) Sr[j] = o.Sr[j];
        *rref = o.rref;
        *rk_thr = o.rk_thr;
        *rk = o.rk;
        put(o.rdirs, rdirs);
        if (Wq)
            for (int q = 0; q < o.rk; q++) put(o.Wq[q], Wq + size_t(q) * 2 * N * NB);
        if (Minv)
            for (int k = 0; k < N; k++) put(o.Minv[k], Minv + size_t(k) * 2 * NB * NB);
        return 0;
    } catch (const std::exception& ex) { g_err = ex.what(); return 1; }
}

}  // extern "C"

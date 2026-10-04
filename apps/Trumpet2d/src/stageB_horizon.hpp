// Round 347 (research round 678, STAGE_B_SPEC sections 3 and 9): sub-step (iv), the finite-J bulk conditions.
//
// HORIZON ROWS, rebuilt on the current state (HZShared), outside the Jacobian:
//   * x_H(theta_m): the root of 1 - beta^A beta_A / alpha^2 = 1 - PS^4 e^{2q} (BR^2 + r^2 BT^2) / PH^2 in domain 1, by
//     Newton in r at each Gauss-Legendre node theta_m (NTH = 4 nt + 24, python_bulk's projection rule), on Kadath's state;
//   * A0, B0, C0 at the surface: the linearised radial operator A F'' + B F' + C F of each equation's test row (equation,
//     k) on each field-mode column (F, j), theta-projected as python_bulk.abc_at -- the partials of the equations in the
//     36 field jets come from the generated stageB_gen.hpp (sbg::eq_*), the jets from Kadath's own fields at the points;
//   * per sector (twist: EQTW rows; scalar: the rest), on the padded tests / columns (EQTW, QB by 2, the rest by 1:
//     round 341's --horizon-pad), rows normalised by max |A, B, C|: the left kernel l of A0 (singular values < 1e-9, as
//     python_bulk.horizon_rows mode 'all'), w = l / rs;
//   * row q = sc_q sum_k w_qk sum_m W_m T_k(theta_m) E_k(r_H(theta_m), theta_m): the equations THEMSELVES on the surface
//     (l . E = 0 is the condition the tau projection discards); its derivative at fixed (x_H, l): sum P . d(jets).  At
//     J = 0 its linearisation is python_bulk's literal row l . (B0 F'(r_H) + C0 F(r_H)) (l kills A0); its value at the
//     J = 0 seed is the seed's own defect, which the defect correction (--defect) subtracts.  sc_q: python_bulk's scale
//     1 / max |v| of the row's radial-Chebyshev column vector at the (mean) surface radius.
// KOMAR's beta^phi TERM (Ope_kbphi), appended as one part to the Komar row:
//   the row is integ(KMG) = 2 pi R2^2 int KMG sin th dth; the term -psi^6 e^{2q} K^r_phi beta^phi inside KMG, with
//   K^r_phi = r^2 sin^2 th d_r beta^phi / (2 alpha e^{2q}) and d_r beta^phi = alpha Ombar_th / (psi^6 r^4 sin^3 th),
//   Ombar_th = d_th Q + 6 J sin^3 th (Q = sin^4 QB), contributes  -pi int_0^pi Ombar_th(R2) beta^phi(R2) dth,
//   beta^phi(R2, th) = - int_0^{1/R2} (alpha / psi^6) (Ombar_th / sin^3 th) u^2 du  (u = 1/r, domain 2), by Gauss-Legendre
//   in u and theta on Kadath's fields; exactly 0 (value and derivative) at J = 0 with QB = 0.
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "stageB_gen.hpp"

extern "C" void dgesvd_(const char* jobu, const char* jobvt, const int* m, const int* n, double* a, const int* lda, double* s,
                        double* u, const int* ldu, double* vt, const int* ldvt, double* work, const int* lwork, int* info);

namespace sbh {

// Gauss-Legendre nodes / weights on [a, b]
inline void gauss(int n, double a, double b, std::vector<double>& x, std::vector<double>& w)
{
    x.resize(n);
    w.resize(n);
    for (int i = 0; i < n; i++) {
        double z = std::cos(M_PI * (i + 0.75) / (n + 0.5)), dp = 0;
        for (int it = 0; it < 100; it++) {
            double p0 = 1, p1 = z;
            for (int k = 2; k <= n; k++) { const double p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
            dp = n * (z * p1 - p0) / (z * z - 1);
            const double dz = p1 / dp;
            z -= dz;
            if (std::fabs(dz) < 1e-16) break;
        }
        x[n - 1 - i] = 0.5 * (b - a) * z + 0.5 * (b + a);
        w[n - 1 - i] = (b - a) / ((1 - z * z) * dp * dp);
    }
}

// the generated equations, in python_bulk's EQN order (EQTW, ESHR, ESHT, EPHI, EQFN, ESIG)
typedef void (*EqFn)(const sbg::cplx*, sbg::cplx, double, double, sbg::cplx&, sbg::cplx*, sbg::cplx*);
static const char* EQNAME[6] = {"EQTW", "ESHR", "ESHT", "EPHI", "EQFN", "ESIG"};
inline EqFn eqfn(int e)
{
    static const EqFn t[6] = {sbg::eq_EQ, sbg::eq_Eshr, sbg::eq_Eshth, sbg::eq_EPhi, sbg::eq_Eq, sbg::eq_Esigma};
    return t[e];
}
// test functions per equation (python_bulk.tests with --eqfn-sin2)
inline int ntests(int e, int NT) { return e == 0 ? NT - 1 : (e == 2 ? NT - 2 : NT); }
inline double test(int e, int k, double th)
{
    switch (e) {
    case 0: return std::cos((2 * k + 1) * th);
    case 2: return std::sin(2 * (k + 1) * th);
    case 4: return std::sin(th) * std::sin(th) * std::cos(2 * k * th);
    default: return std::cos(2 * k * th);
    }
}
// fields in python_bulk's FIELDS order (PS, PH, QF, BR, BT, QB): the generated jet block (sorted names: Phb 0, Qb 6, br 12,
// bt 18, psi2 24, qf 30; within a block 00 01 02 10 11 20) and the angular basis of a column (value, d/dth, d2/dth2) of
// the SYMPY field (q = sin^2 th QF, Q = sin^4 th QB)
static const int JBLK[6] = {24, 0, 30, 12, 18, 6};
static const char* FNAME[6] = {"PS", "PH", "QF", "BR", "BT", "QB"};
inline int ncols(int F, int NT) { return F == 4 ? NT - 2 : (F == 5 ? NT - 1 : NT); }
inline void angular(int F, int j, double t, double* o)
{
    const double s = std::sin(t), c = std::cos(t);
    if (F == 4) {
        const double w = 2.0 * (j + 1);
        o[0] = std::sin(w * t); o[1] = w * std::cos(w * t); o[2] = -w * w * std::sin(w * t);
        return;
    }
    if (F == 5) {
        const double w = 2.0 * j + 1, a = std::cos(w * t), a1 = -w * std::sin(w * t), a2 = -w * w * a;
        const double p = s * s * s * s, p1 = 4 * s * s * s * c, p2 = 12 * s * s * c * c - 4 * s * s * s * s;
        o[0] = p * a; o[1] = p1 * a + p * a1; o[2] = p2 * a + 2 * p1 * a1 + p * a2;
        return;
    }
    const double w = 2.0 * j, a = std::cos(w * t), a1 = -w * std::sin(w * t), a2 = -w * w * a;
    if (F == 2) {
        const double p = s * s, p1 = 2 * s * c, p2 = 2 * (c * c - s * s);
        o[0] = p * a; o[1] = p1 * a + p * a1; o[2] = p2 * a + 2 * p1 * a1 + p * a2;
        return;
    }
    o[0] = a; o[1] = a1; o[2] = a2;
}

// the left kernel of an m x n matrix (row-major): columns of U with singular value < tol (python_bulk: the last nk
// columns of U, nk = #(sv < tol)); returns m x nk (column-major) and the singular values
inline std::vector<double> left_kernel(const std::vector<double>& A, int m, int n, double tol, int& nk, std::vector<double>& sv)
{
    std::vector<double> a(static_cast<size_t>(m) * n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) a[static_cast<size_t>(j) * m + i] = A[static_cast<size_t>(i) * n + j];
    const int mn = std::min(m, n);
    sv.assign(mn, 0.0);
    std::vector<double> U(static_cast<size_t>(m) * m), VT(1);
    int lwork = -1, info = 0, one = 1;
    double wq = 0;
    dgesvd_("A", "N", &m, &n, a.data(), &m, sv.data(), U.data(), &m, VT.data(), &one, &wq, &lwork, &info);
    lwork = static_cast<int>(wq) + 1;
    std::vector<double> work(lwork);
    dgesvd_("A", "N", &m, &n, a.data(), &m, sv.data(), U.data(), &m, VT.data(), &one, work.data(), &lwork, &info);
    if (info != 0) throw std::runtime_error("horizon: dgesvd failed");
    nk = 0;
    for (double v : sv) if (v < tol) nk++;
    return std::vector<double>(U.begin() + static_cast<size_t>(m) * (m - nk), U.end());
}

}  // namespace sbh

// ---------------------------------------------------------------------------------------------------------------------
// The Kadath side (included after Kadath's headers): the shared horizon state, the horizon-row and Komar beta^phi Opes.
namespace sbh {

inline const Kadath::Val_domain& vd(const Kadath::Tensor& T, int dom)
{
    const Kadath::Array<int> ind(T.indices(0));
    return T(ind)(dom);
}
inline double at(const Kadath::Val_domain& v, const Kadath::Point& num)
{
    if (v.check_if_zero()) return 0.0;
    return v.get_base().summation(num, v.get_coef_ref());
}
// a cheap fingerprint of a Val_domain (its coefficients, position-weighted)
inline double fp(const Kadath::Val_domain& v)
{
    if (v.check_if_zero()) return 0.0;
    const Kadath::Array<double>& c = v.get_coef_ref();
    double s = 0.0;
    const int n = c.get_nbr();
    const double* p = c.get_data();
    for (int i = 0; i < n; i++) s += p[i] * (1.0 + 1e-3 * i);
    return s;
}

struct HZShared {
    const Kadath::System_of_eqs* sys = nullptr;
    const Kadath::Space* space = nullptr;
    int NT = 0, NTH = 0, dom = 1;
    double Jv = 0.0, lo = 0.0, hi = 0.0, r0 = 0.7793271081;
    std::vector<std::unique_ptr<Kadath::Ope_eq>> base, jet;     // 6 Kadath variables; 36 generated-order jets
    std::vector<double> th, W, rH;
    std::vector<Kadath::Point> num;
    // the test rows (global, python_bulk order: equation-major) and the padded columns
    std::vector<int> re, rk, cF, cj;                             // row: equation, k; column: field, j
    std::vector<int> rowsec;                                     // per registered row: 0 twist, 1 scalar
    std::vector<std::vector<double>> w;                          // per registered row: weights over the global test rows
    std::vector<double> sc, val, G, A0, B0, C0;
    std::vector<std::vector<double>> dval;                       // per registered row, per lane
    std::vector<std::array<double, 36>> Jt;                      // the jets at the nodes (this state)
    std::vector<int> nker;                                       // kernel dimension per sector at the last rebuild
    std::vector<double> svmin;                                   // per sector: the largest kept / smallest dropped sv
    double fval = std::nan(""), tbuild = 0, tder = 0, xHspread = 0, fres = 0;
    std::vector<double> fder;
    long nbuild = 0, nder = 0, ncall = 0;
    bool built = false;
    int nlanes = 0;

    void init(const Kadath::System_of_eqs* s, const Kadath::Space* sp, int nt, double J, double l, double h)
    {
        sys = s; space = sp; NT = nt; Jv = J; lo = l; hi = h;
        NTH = 4 * NT + 24;
        gauss(NTH, 0.0, M_PI, th, W);
        rH.assign(NTH, r0);
        num.assign(NTH, Kadath::Point(2));
        auto ope = [&](const std::string& ex) {
            char nrm[Kadath::LMAX];
            Kadath::trim_spaces(nrm, ex.c_str());
            return std::unique_ptr<Kadath::Ope_eq>(sys->give_ope(dom, nrm));
        };
        for (const char* F : FNAME) base.push_back(ope(F));
        const char* bx[6] = {"PH", "HZT", "BR", "BT", "PS", "HZQ"};          // Phb Qb br bt psi2 qf
        for (const char* X : bx) {
            const std::string x(X);
            for (const std::string& e : {x, "dt(" + x + ")", "dt(dt(" + x + "))", "dr(" + x + ")", "dr(dt(" + x + "))",
                                          "dr(dr(" + x + "))"})
                jet.push_back(ope(e));
        }
        for (int e = 0; e < 6; e++) {
            const int depth = e == 0 ? 2 : 1, nk = ntests(e, NT);
            for (int k = 0; k <= nk - 1 - depth; k++) { re.push_back(e); rk.push_back(k); }
        }
        for (int F = 0; F < 6; F++) {
            const int depth = F == 5 ? 2 : 1, nc = ncols(F, NT);
            for (int j = 0; j <= nc - 1 - depth; j++) { cF.push_back(F); cj.push_back(j); }
        }
    }
    Kadath::Point point(double r, double t) const
    {
        Kadath::Point M(2);
        M.set(1) = r * std::sin(t);
        M.set(2) = r * std::cos(t);
        return space->get_domain(dom)->absol_to_num(M);
    }
    // the degenerate surface at node m on the jets' Terms (Newton in r)
    double locate(const std::vector<Kadath::Term_eq>& T, int m, double rg, double& res) const
    {
        const Kadath::Val_domain &ps = vd(T[24].get_val_t(), dom), &psr = vd(T[27].get_val_t(), dom),
                                 &ph = vd(T[0].get_val_t(), dom), &phr = vd(T[3].get_val_t(), dom),
                                 &br = vd(T[12].get_val_t(), dom), &brr = vd(T[15].get_val_t(), dom),
                                 &bt = vd(T[18].get_val_t(), dom), &btr = vd(T[21].get_val_t(), dom),
                                 &q = vd(T[30].get_val_t(), dom), &qr = vd(T[33].get_val_t(), dom);
        double r = rg;
        for (int it = 0; it < 60; it++) {
            const Kadath::Point p = point(r, th[m]);
            const double PS = at(ps, p), PH = at(ph, p), BR = at(br, p), BT = at(bt, p), Q = at(q, p);
            const double dPS = at(psr, p), dPH = at(phr, p), dBR = at(brr, p), dBT = at(btr, p), dQ = at(qr, p);
            const double b2 = BR * BR + r * r * BT * BT, db2 = 2 * BR * dBR + 2 * r * BT * BT + 2 * r * r * BT * dBT;
            const double g = std::pow(PS, 4) * std::exp(2 * Q) / (PH * PH);
            const double dg = g * (4 * dPS / PS + 2 * dQ - 2 * dPH / PH);
            const double f = 1 - g * b2, df = -(dg * b2 + g * db2);
            const double dr = f / df;
            r -= dr;
            res = f;
            if (std::fabs(dr) < 1e-15 * r) break;
        }
        return r;
    }
    void rebuild(const std::vector<Kadath::Term_eq>& T)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const char* fix = std::getenv("HZ_RH_FIX");                 // diagnostic only: the surface pinned to a given sphere
        for (int m = 0; m < NTH; m++) rH[m] = fix ? std::atof(fix) : locate(T, m, built ? rH[m] : r0, fres);
        double rmin = 1e300, rmax = -1e300, rbar = 0;
        for (int m = 0; m < NTH; m++) { rmin = std::min(rmin, rH[m]); rmax = std::max(rmax, rH[m]); rbar += rH[m] / NTH; }
        xHspread = rmax - rmin;
        Jt.assign(NTH, {});
        const int NR = static_cast<int>(re.size()), NC = static_cast<int>(cF.size());
        A0.assign(static_cast<size_t>(NR) * NC, 0.0); B0 = A0; C0 = A0;
        std::vector<double> E(static_cast<size_t>(NTH) * 6), P(static_cast<size_t>(NTH) * 6 * 36);
        std::vector<sbg::cplx> Hbuf(4096);
        for (int m = 0; m < NTH; m++) {
            num[m] = point(rH[m], th[m]);
            sbg::cplx jc[36];
            for (int a = 0; a < 36; a++) { Jt[m][a] = at(vd(T[a].get_val_t(), dom), num[m]); jc[a] = Jt[m][a]; }
            for (int e = 0; e < 6; e++) {
                sbg::cplx Ev, Pv[36];
                for (auto& x : Pv) x = 0.0;
                eqfn(e)(jc, sbg::cplx(rH[m]), th[m], Jv, Ev, Pv, Hbuf.data());
                E[m * 6 + e] = Ev.real();
                for (int a = 0; a < 36; a++) P[(static_cast<size_t>(m) * 6 + e) * 36 + a] = Pv[a].real();
            }
        }
        // A0, B0, C0 (python_bulk.abc_at at the surface): jets within a block 00 01 02 10 11 20
        for (int i = 0; i < NR; i++)
            for (int c = 0; c < NC; c++) {
                double a = 0, b = 0, cc = 0, ang[3];
                const int bk = JBLK[cF[c]];
                for (int m = 0; m < NTH; m++) {
                    angular(cF[c], cj[c], th[m], ang);
                    const double tw = W[m] * test(re[i], rk[i], th[m]);
                    const double* p = &P[(static_cast<size_t>(m) * 6 + re[i]) * 36 + bk];
                    a += tw * p[5] * ang[0];
                    b += tw * (p[3] * ang[0] + p[4] * ang[1]);
                    cc += tw * (p[0] * ang[0] + p[1] * ang[1] + p[2] * ang[2]);
                }
                A0[static_cast<size_t>(i) * NC + c] = a; B0[static_cast<size_t>(i) * NC + c] = b; C0[static_cast<size_t>(i) * NC + c] = cc;
            }
        // per sector: the left kernel of A0 (sector rows x every padded column), rows normalised by max |A, B, C|
        std::vector<std::vector<double>> wn;
        std::vector<int> secn;
        nker.assign(2, 0); svmin.assign(4, 0.0);
        for (int s = 0; s < 2; s++) {
            std::vector<int> ri;
            for (int i = 0; i < NR; i++) if ((re[i] == 0) == (s == 0)) ri.push_back(i);
            const int m = static_cast<int>(ri.size());
            std::vector<double> A(static_cast<size_t>(m) * NC), rs(m, 0.0);
            for (int a = 0; a < m; a++) {
                for (int c = 0; c < NC; c++) {
                    const size_t ix = static_cast<size_t>(ri[a]) * NC + c;
                    rs[a] = std::max({rs[a], std::fabs(A0[ix]), std::fabs(B0[ix]), std::fabs(C0[ix])});
                }
                for (int c = 0; c < NC; c++) A[static_cast<size_t>(a) * NC + c] = A0[static_cast<size_t>(ri[a]) * NC + c] / rs[a];
            }
            int nk = 0;
            std::vector<double> sv;
            const std::vector<double> L = left_kernel(A, m, NC, 1e-9, nk, sv);
            nker[s] = nk;
            const int mn = static_cast<int>(sv.size());
            svmin[2 * s] = mn - nk - 1 >= 0 ? sv[mn - nk - 1] : 0.0;
            svmin[2 * s + 1] = nk > 0 ? sv[mn - nk] : 0.0;
            for (int q = 0; q < nk; q++) {
                std::vector<double> wq(NR, 0.0);
                for (int a = 0; a < m; a++) wq[ri[a]] = L[static_cast<size_t>(q) * m + a] / rs[a];
                wn.push_back(wq);
                secn.push_back(s);
            }
        }
        if (!built) { w = wn; rowsec = secn; }
        else if (wn.size() != w.size())
            throw std::runtime_error("horizon: the left kernel changed dimension (" + std::to_string(w.size()) + " -> " +
                                     std::to_string(wn.size()) + ")");
        else { w = wn; rowsec = secn; }
        const int nrow = static_cast<int>(w.size());
        // scale (python_bulk: 1 / max |v|, v the row's domain-1 Chebyshev column vector at the surface radius)
        const double L = std::log(hi / lo), x0 = 2 * std::log(rbar / lo) / L - 1, xr = 2 / (rbar * L);
        const int NR1 = 33;
        std::vector<double> T0(NR1), T1(NR1);
        for (int i = 0; i < NR1; i++) {
            T0[i] = std::cos(i * std::acos(std::max(-1.0, std::min(1.0, x0))));
            const double s0 = std::sqrt(std::max(1e-300, 1 - x0 * x0));
            T1[i] = (i == 0 ? 0.0 : i * std::sin(i * std::acos(x0)) / s0) * xr;
        }
        sc.assign(nrow, 0.0);
        val.assign(nrow, 0.0);
        G.assign(static_cast<size_t>(nrow) * NTH * 36, 0.0);
        for (int q = 0; q < nrow; q++) {
            double vmax = 0;
            for (int c = 0; c < NC; c++) {
                double b = 0, cc = 0;
                for (int i = 0; i < NR; i++) { b += w[q][i] * B0[static_cast<size_t>(i) * NC + c]; cc += w[q][i] * C0[static_cast<size_t>(i) * NC + c]; }
                for (int i = 0; i < NR1; i++) vmax = std::max(vmax, std::fabs(b * T1[i] + cc * T0[i]));
            }
            sc[q] = 1.0 / vmax;
            for (int i = 0; i < NR; i++) {
                if (w[q][i] == 0.0) continue;
                for (int m = 0; m < NTH; m++) {
                    const double f = sc[q] * w[q][i] * W[m] * test(re[i], rk[i], th[m]);
                    val[q] += f * E[m * 6 + re[i]];
                    const double* p = &P[(static_cast<size_t>(m) * 6 + re[i]) * 36];
                    double* g = &G[(static_cast<size_t>(q) * NTH + m) * 36];
                    for (int a = 0; a < 36; a++) g[a] += f * p[a];
                }
            }
        }
        built = true;
        nbuild++;
        tbuild += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    // called by every horizon-row Ope: rebuild on a new state, derivative lanes on new lanes
    void sync()
    {
        ncall++;
        std::vector<Kadath::Term_eq> B;
        B.reserve(6);
        for (auto& o : base) B.push_back(o->action());
        double f = 0;
        for (int k = 0; k < 6; k++) f += (k + 1.0) * fp(vd(B[k].get_val_t(), dom));
        const int nl = B[0].has_der_t(0) ? std::max(1, B[0].get_derivative_lane_count()) : 0;
        std::vector<double> fd(nl, 0.0);
        for (int l = 0; l < nl; l++)
            for (int k = 0; k < 6; k++)
                if (B[k].has_der_t(l)) fd[l] += (k + 1.0) * fp(vd(B[k].get_der_t(l), dom));
        const bool newval = !built || f != fval;
        const bool newder = nl > 0 && (newval || nl != nlanes || fd != fder);
        if (!newval && !newder) return;
        std::vector<Kadath::Term_eq> T;
        T.reserve(36);
        for (auto& o : jet) T.push_back(o->action());
        if (newval) { rebuild(T); fval = f; }
        if (nl > 0 && newder) {
            const auto t0 = std::chrono::steady_clock::now();
            const int nrow = static_cast<int>(w.size());
            dval.assign(nrow, std::vector<double>(nl, 0.0));
            for (int l = 0; l < nl; l++) {
                bool any = false;
                for (int a = 0; a < 36; a++) if (T[a].has_der_t(l) && !vd(T[a].get_der_t(l), dom).check_if_zero()) { any = true; break; }
                if (!any) continue;
                for (int a = 0; a < 36; a++) {
                    if (!T[a].has_der_t(l)) continue;
                    const Kadath::Val_domain& d = vd(T[a].get_der_t(l), dom);
                    if (d.check_if_zero()) continue;
                    for (int m = 0; m < NTH; m++) {
                        const double dj = at(d, num[m]);
                        if (dj == 0.0) continue;
                        for (int q = 0; q < nrow; q++) dval[q][l] += G[(static_cast<size_t>(q) * NTH + m) * 36 + a] * dj;
                    }
                }
            }
            fder = fd;
            nlanes = nl;
            nder++;
            tder += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        } else if (nl == 0) {
            nlanes = 0;
        }
    }
};

class Ope_hz : public Kadath::Ope_eq {
    HZShared* sh;
    int q;
  public:
    Ope_hz(const Kadath::System_of_eqs* s, HZShared* h, int qq) : Kadath::Ope_eq(s, 1, 0), sh(h), q(qq) {}
    Kadath::Term_eq action() const override
    {
        sh->sync();
        Kadath::Term_eq res(dom, sh->val[q]);
        if (sh->nlanes > 0) {
            res.set_der_d(sh->dval[q][0]);
            res.set_derivative_lane_count(sh->nlanes);
            for (int l = 1; l < sh->nlanes; l++) res.set_der_d(l, sh->dval[q][l]);
        }
        return res;
    }
};

// Komar's beta^phi term: pi int_0^pi sin^3 Om(R2) [int_0^{1/R2} u^2 PH Om / PS^4 du] dth, Om = 4 cos QB + sin dt(QB) + 6 J
class Ope_kbphi : public Kadath::Ope_eq {
    std::unique_ptr<Kadath::Ope_eq> oph, ops, oqb, oqt;
    const Kadath::Space* space;
    int d2;
    double J;
    std::vector<double> tq, wt, uq, wu;
    std::vector<std::vector<Kadath::Point>> pu;                  // [theta][u]
    std::vector<Kadath::Point> p2;                               // at R2
  public:
    mutable double last = 0, tcost = 0, fval = std::nan("");
    mutable long ncall = 0, nval = 0;
    mutable std::vector<double> c_om2, c_I, c_PH, c_PS, c_om;     // cached at the last state: per theta, per (theta, u)
    mutable std::vector<double> bphi;                            // beta^phi(R2, theta_i) at the last evaluation
    Ope_kbphi(const Kadath::System_of_eqs* s, const Kadath::Space* sp, int dd, double R2, double JJ, int nth = 64, int nu = 48)
        : Kadath::Ope_eq(s, dd, 0), space(sp), d2(dd), J(JJ)
    {
        auto ope = [&](const char* ex) {
            char nrm[Kadath::LMAX];
            Kadath::trim_spaces(nrm, ex);
            return std::unique_ptr<Kadath::Ope_eq>(s->give_ope(d2, nrm));
        };
        oph = ope("PH"); ops = ope("PS"); oqb = ope("QB"); oqt = ope("dt(QB)");
        gauss(nth, 0.0, M_PI, tq, wt);
        gauss(nu, 0.0, 1.0 / R2, uq, wu);
        auto pt = [&](double r, double t) {
            Kadath::Point M(2);
            M.set(1) = r * std::sin(t);
            M.set(2) = r * std::cos(t);
            return space->get_domain(d2)->absol_to_num(M);
        };
        pu.assign(nth, {});
        for (int i = 0; i < nth; i++) {
            p2.push_back(pt(R2, tq[i]));
            for (int k = 0; k < nu; k++) pu[i].push_back(pt(1.0 / uq[k], tq[i]));
        }
    }
    Kadath::Term_eq action() const override
    {
        const auto t0 = std::chrono::steady_clock::now();
        ncall++;
        const Kadath::Term_eq TH(oph->action()), TS(ops->action()), TB(oqb->action()), TT(oqt->action());
        const Kadath::Val_domain &ph = vd(TH.get_val_t(), d2), &ps = vd(TS.get_val_t(), d2), &qb = vd(TB.get_val_t(), d2),
                                 &qt = vd(TT.get_val_t(), d2);
        const int nth = static_cast<int>(tq.size()), nu = static_cast<int>(uq.size());
        const double f = fp(ph) + 2 * fp(ps) + 3 * fp(qb) + 4 * fp(qt);
        if (!(f == fval)) {                                      // a new state: the value and the cached point values
            nval++;
            fval = f;
            c_om2.assign(nth, 0.0); c_I.assign(nth, 0.0);
            c_PH.assign(static_cast<size_t>(nth) * nu, 0.0); c_PS = c_PH; c_om = c_PH;
            bphi.assign(nth, 0.0);
            last = 0;
            for (int i = 0; i < nth; i++) {
                const double sn = std::sin(tq[i]), c = std::cos(tq[i]);
                c_om2[i] = 4 * c * at(qb, p2[i]) + sn * at(qt, p2[i]) + 6 * J;
                for (int k = 0; k < nu; k++) {
                    const Kadath::Point& p = pu[i][k];
                    const size_t ik = static_cast<size_t>(i) * nu + k;
                    c_PH[ik] = at(ph, p); c_PS[ik] = at(ps, p); c_om[ik] = 4 * c * at(qb, p) + sn * at(qt, p) + 6 * J;
                    c_I[i] += wu[k] * uq[k] * uq[k] / std::pow(c_PS[ik], 4) * c_PH[ik] * c_om[ik];
                }
                bphi[i] = -c_I[i];
                last += M_PI * wt[i] * sn * sn * sn * c_om2[i] * c_I[i];
            }
        }
        const int nl = TS.has_der_t(0) ? std::max(1, TS.get_derivative_lane_count()) : 0;
        Kadath::Term_eq res(dom, last);
        if (nl > 0) {
            std::vector<double> dv(nl, 0.0);
            for (int l = 0; l < nl; l++) {
                const Kadath::Val_domain* dH = TH.has_der_t(l) && !vd(TH.get_der_t(l), d2).check_if_zero() ? &vd(TH.get_der_t(l), d2) : nullptr;
                const Kadath::Val_domain* dS = TS.has_der_t(l) && !vd(TS.get_der_t(l), d2).check_if_zero() ? &vd(TS.get_der_t(l), d2) : nullptr;
                const Kadath::Val_domain* dB = TB.has_der_t(l) && !vd(TB.get_der_t(l), d2).check_if_zero() ? &vd(TB.get_der_t(l), d2) : nullptr;
                const Kadath::Val_domain* dT = TT.has_der_t(l) && !vd(TT.get_der_t(l), d2).check_if_zero() ? &vd(TT.get_der_t(l), d2) : nullptr;
                if (!dH && !dS && !dB && !dT) continue;
                for (int i = 0; i < nth; i++) {
                    const double sn = std::sin(tq[i]), c = std::cos(tq[i]);
                    double dI = 0;
                    for (int k = 0; k < nu; k++) {
                        const Kadath::Point& p = pu[i][k];
                        const size_t ik = static_cast<size_t>(i) * nu + k;
                        const double dPH = dH ? at(*dH, p) : 0.0, dPS = dS ? at(*dS, p) : 0.0;
                        const double dom_ = (dB ? 4 * c * at(*dB, p) : 0.0) + (dT ? sn * at(*dT, p) : 0.0);
                        const double g = wu[k] * uq[k] * uq[k] / std::pow(c_PS[ik], 4);
                        dI += g * (dPH * c_om[ik] + c_PH[ik] * dom_ - 4 * c_PH[ik] * c_om[ik] * dPS / c_PS[ik]);
                    }
                    const double dom2 = (dB ? 4 * c * at(*dB, p2[i]) : 0.0) + (dT ? sn * at(*dT, p2[i]) : 0.0);
                    dv[l] += M_PI * wt[i] * sn * sn * sn * (dom2 * c_I[i] + c_om2[i] * dI);
                }
            }
            res.set_der_d(dv[0]);
            res.set_derivative_lane_count(nl);
            for (int l = 1; l < nl; l++) res.set_der_d(l, dv[l]);
        }
        tcost += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return res;
    }
};

}  // namespace sbh

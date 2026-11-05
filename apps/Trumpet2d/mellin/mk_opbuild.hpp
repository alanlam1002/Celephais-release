// Mellin throat recursion, M3.5 (code round 463; research rounds 1034-1035): THE OPERATOR ABOUT THE CYLINDER, natively, from the gated
// kernel's linearisation -- in place of the sympy builder (spinning_trumpet/scripts/e32_exponents.py build kcyl -> n2a_lk.py --ham --pshape).
//
// The series empty (the cylinder alone, the bundle's), every column given a unit moment-2 direction at one exponent e0 (off the integers): the
// kernel's exact linear response (forward mode) read for every stored row fam@P_D at e0 + p + the row's offset, for each level p, returns
// moments (2, 1, 0) = (M_p(e0), M_p'(e0), M_p''(e0) / 2) in the aligned exponent e; with s = n e (n2a_rec.build_ops: s = n (sigma + kE)) the
// builder's M_p(s) = A + s B + s^2 C.  The rows are projected with stored-row weights computed here (Gauss-Legendre nodes and P_D by the
// three-term recurrence; the grid -> node interpolation n2a_nl.interp's).  The derived quantities follow n2a_rec.build_ops' rules
// (kE_fixed, the leading levels at 1e-7 of each row's largest entry, rw / cw from the leading operator at sigma 0.37 + 0.11i), n2a_nl's
// --dEbt0 and --rw-out-used, the condition rows (H_c@P0, E_sh^th@P1) and the gate rows (degree <= edge, live at 1e-8 of the family's largest).
//
// Conventions against the builder (research round 1035, code round 1035's finding): (i) the builder's background fixes the lapse
// normalisation K = 1 ('an r-scaling'), the kernel's is the bundle's (Klapse): the level -2 entries differ by K^2; (ii) n2a_lk divides the
// linearised E_Phi rows by Phibar row by row, dropping - (E_Phi / Phibar)_bg dPhibar / Phibar -- nonzero, since the cylinder's own H_c and
// E_sigma do not vanish at grade 0 -- the kernel's L_K is the exact linearisation.  lkdiv = rowwise restores the builder's convention.
#pragma once
#include "mk_operator.hpp"

#include <cmath>
#include <string>

namespace mk {
namespace ob {

// Gauss-Legendre nodes (ascending, as numpy's leggauss) and weights: Newton on P_n by the recurrence from the Chebyshev guesses
inline void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
    x.assign(n, 0.0);
    w.assign(n, 0.0);
    const double pi = std::acos(-1.0);
    for (int i = 0; i < n; i++) {
        double z = -std::cos(pi * (i + 0.75) / (n + 0.5));
        double dp = 1.0;
        for (int it = 0; it < 100; it++) {
            double p0 = 1.0, p1 = z;
            for (int k = 2; k <= n; k++) {
                double p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            dp = n * (z * p1 - p0) / (z * z - 1.0);
            double dz = p1 / dp;
            z -= dz;
            if (std::abs(dz) < 1e-16) break;
        }
        double p0 = 1.0, p1 = z;
        for (int k = 2; k <= n; k++) {
            double p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
            p0 = p1;
            p1 = p2;
        }
        dp = n * (z * p1 - p0) / (z * z - 1.0);
        x[i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
    }
}

// P_0 .. P_Dmax at z (the three-term recurrence)
inline void legendre_upto(int Dmax, double z, std::vector<double>& P) {
    P.assign(Dmax + 1, 0.0);
    P[0] = 1.0;
    if (Dmax >= 1) P[1] = z;
    for (int k = 2; k <= Dmax; k++) P[k] = ((2 * k - 1) * z * P[k - 1] - (k - 1) * P[k - 2]) / k;
}

struct Quad {
    int NF = 0, NG = 0;
    std::vector<double> x, w, th;
    std::vector<cd> I;                          // (NG, NF): n2a_nl.interp(arccos x): sum_m e^(i m th_g) e^(-i m th_q) / NF, Nyquist dropped
    void init(int NF_, int NG_ = 200) {
        NF = NF_;
        NG = NG_;
        gauss_legendre(NG, x, w);
        th.resize(NG);
        for (int g = 0; g < NG; g++) th[g] = std::acos(x[g]);
        const double pi = std::acos(-1.0);
        std::vector<int> m;
        for (int k = 0; k < NF; k++) {
            int mm = k < (NF + 1) / 2 ? k : k - NF;
            if (std::abs(mm) < NF / 2) m.push_back(mm);
        }
        I.assign(size_t(NG) * NF, 0.0);
        for (int g = 0; g < NG; g++)
            for (int q = 0; q < NF; q++) {
                double thq = 2 * pi * (q + 0.5) / NF;
                cd s = 0.0;
                for (int mm : m) s += std::exp(cd(0.0, mm * (th[g] - thq)));
                I[size_t(g) * NF + q] = s / double(NF);
            }
    }
    // a node function f_g -> its grid weights: W[q] = sum_g I[g, q] f_g
    std::vector<cd> weights(const std::vector<double>& f) const {
        std::vector<cd> W(NF, 0.0);
        for (int g = 0; g < NG; g++)
            for (int q = 0; q < NF; q++) W[q] += I[size_t(g) * NF + q] * f[g];
        return W;
    }
    // the stored row's weights (n2a_nl Proj.stored_w(D) @ P.I): (2D + 1) / 2 w_g P_D(x_g)
    std::vector<cd> stored(int D) const {
        std::vector<double> f(NG), P;
        for (int g = 0; g < NG; g++) {
            legendre_upto(D, x[g], P);
            f[g] = (2 * D + 1) / 2.0 * w[g] * P[D];
        }
        return weights(f);
    }
};

static const char* ROWS[6] = {"H_c", "E_sigma", "L_K", "E_sh^r", "E_sh^th", "E_Q"};   // the kernel's row families (mk_binding.ROWS)
inline int rowfam(const std::string& s) {
    for (int i = 0; i < 6; i++)
        if (s == ROWS[i]) return i;
    return -1;
}

struct Op {
    std::vector<int> PL;                        // levels
    std::vector<int> rf, rD;                    // tall rows: family, degree
    int nn = 0;
    std::vector<double> A, B, C;                // (np, nr, nn)
    double imag = 0.0;
    std::vector<double> bg_ephi;                // (E_Phi / Phibar)_bg = H_c + 2 E_sigma - L_K on the cylinder, P_0, P_2, P_4
    size_t at(int ip, int r, int j) const { return (size_t(ip) * rf.size() + r) * nn + j; }
};

// the operator about the bundle's cylinder (the kernel's state: series empty), tall rows (rf, rD), levels PL, columns 0 .. nn - 1
inline Op build(Kernel& kr, int nn, const std::vector<int>& rf, const std::vector<int>& rD, const std::vector<int>& PL, double e0, double top,
                int chunk, bool rowwise, const Quad& Q) {
    Ctx& c = kr.c;
    int K = c.K, NF = c.NF, nr = int(rf.size()), np_ = int(PL.size());
    Op o;
    o.PL = PL; o.rf = rf; o.rD = rD; o.nn = nn;
    std::vector<cd> Wt(size_t(nr) * NF);
    {
        std::map<int, std::vector<cd>> cache;
        for (int r = 0; r < nr; r++) {
            if (!cache.count(rD[r])) cache[rD[r]] = Q.stored(rD[r]);
            std::copy(cache[rD[r]].begin(), cache[rD[r]].end(), Wt.begin() + long(r) * NF);
        }
    }
    std::vector<cd> M(size_t(np_) * 3 * nr * nn, 0.0);           // [ip][m][r][j]
    std::vector<Entry> empty;
    Key k0 = ckey(c, e0);
    for (int c0 = 0; c0 < nn; c0 += chunk) {
        int nq = std::min(nn, c0 + chunk) - c0;
        std::vector<Entry> dC;
        for (int q = 0; q < nq; q++) {
            std::vector<cd> arr(size_t(nq) * K, 0.0);
            arr[size_t(q) * K + 2] = 1.0;
            dC.push_back(Entry{c0 + q, k0, nq, std::move(arr)});
        }
        kr.run(empty, &dC, std::max(top, std::ceil(e0) + 4));
        for (int ip = 0; ip < np_; ip++) {
            std::vector<int> lev(nr, PL[ip]);
            auto R = project(c, kr.dout, e0, nq, rf, lev, Wt, false);   // (nq, K, nr)
            for (int q = 0; q < nq; q++)
                for (int m = 0; m < 3; m++)
                    for (int r = 0; r < nr; r++)
                        M[((size_t(ip) * 3 + m) * nr + r) * nn + c0 + q] = R[(size_t(q) * K + (2 - m)) * nr + r];
        }
    }
    // M(e) = M0 + (e - e0) M1 + (e - e0)^2 M2, e = s / n  ->  A + s B + s^2 C
    double n = c.n;
    size_t sz = size_t(np_) * nr * nn;
    o.A.assign(sz, 0.0); o.B.assign(sz, 0.0); o.C.assign(sz, 0.0);
    for (int ip = 0; ip < np_; ip++)
        for (int r = 0; r < nr; r++)
            for (int j = 0; j < nn; j++) {
                cd m0 = M[((size_t(ip) * 3 + 0) * nr + r) * nn + j], m1 = M[((size_t(ip) * 3 + 1) * nr + r) * nn + j],
                   m2 = M[((size_t(ip) * 3 + 2) * nr + r) * nn + j];
                cd Cn = m2 / (n * n), Bn = m1 / n - 2 * e0 * m2 / n, An = m0 - e0 * m1 + e0 * e0 * m2;
                o.imag = std::max({o.imag, std::abs(An.imag()), std::abs(Bn.imag()), std::abs(Cn.imag())});
                o.A[o.at(ip, r, j)] = An.real(); o.B[o.at(ip, r, j)] = Bn.real(); o.C[o.at(ip, r, j)] = Cn.real();
            }
    // the cylinder's own rows at grade 0: (E_Phi / Phibar)_bg = H_c + 2 E_sigma - L_K (on the grid, moment 0)
    kr.run(empty, nullptr, std::max(top, std::ceil(e0) + 4));
    std::vector<cd> ephi(NF, 0.0);
    const double cf[3] = {1.0, 2.0, -1.0};
    for (int f = 0; f < 3; f++) {
        Piece buf;
        const Piece* p = piece_at(c, kr.out[f], ROWOFF1[f] + 0.0, buf);
        if (!p) continue;
        for (int q = 0; q < NF; q++) ephi[q] += cf[f] * p->a[q];
    }
    for (int D : {0, 2, 4}) {
        auto W = Q.stored(D);
        cd s = 0.0;
        for (int q = 0; q < NF; q++) s += ephi[q] * W[q];
        o.bg_ephi.push_back(s.real());
    }
    if (rowwise) {
        // the builder's convention (n2a_lk): A[p 0][L_K@P_D, PH_l] -= P_D[(E_Phi / Phibar)_bg P_l]
        int ip0 = -1;
        for (int ip = 0; ip < np_; ip++)
            if (PL[ip] == 0) ip0 = ip;
        int fPH = Kernel::fam("PH");
        for (int j = 0; j < nn; j++) {
            if (kr.colfam[j] != fPH) continue;
            const double* Bv = &kr.cols.B[fPH][size_t(kr.colidx[j]) * NF];
            for (int r = 0; r < nr; r++) {
                if (rf[r] != 2) continue;
                cd s = 0.0;
                for (int q = 0; q < NF; q++) s += ephi[q] * Bv[q] * Wt[size_t(r) * NF + q];
                o.A[o.at(ip0, r, j)] -= s.real();
            }
        }
    }
    return o;
}

// ---- the derived quantities: n2a_rec.build_ops (kE_fixed) + _finish, n2a_nl's --dEbt0 / --rw-out-used, conditions, gate rows ----
struct Sel {                                     // an operator on selected rows: M_p (np, nsel, nn) as A, B, C
    std::vector<double> A, B, C;
    int nr = 0;
};

inline Sel select(const Op& o, const std::vector<std::vector<double>>& P) {   // P: (nsel, nr) row combinations
    Sel s;
    int np_ = int(o.PL.size()), nr = int(o.rf.size()), nn = o.nn;
    s.nr = int(P.size());
    s.A.assign(size_t(np_) * s.nr * nn, 0.0); s.B = s.A; s.C = s.A;
    for (int ip = 0; ip < np_; ip++)
        for (int i = 0; i < s.nr; i++)
            for (int r = 0; r < nr; r++) {
                double p = P[i][r];
                if (p == 0.0) continue;
                for (int j = 0; j < nn; j++) {
                    size_t a = (size_t(ip) * s.nr + i) * nn + j, b = o.at(ip, r, j);
                    s.A[a] += p * o.A[b]; s.B[a] += p * o.B[b]; s.C[a] += p * o.C[b];
                }
            }
    return s;
}

struct Derived {
    std::vector<int> dE;                         // rows' leading levels (99: absent)
    std::vector<double> rw, cw, rw_out;
};

// build_ops (kE fixed): pm = the first level where |A|, |B|, |C| > 1e-7 x the row's largest; dE = min_j pm + kE_j; rw, cw from the leading
// operator at sigma 0.37 + 0.11i (levE = dE - kE); rw_out (--rw-out-used) at the levels after the BT rows' change (dEbt0: BT rows -> 0).
// dE_fix (M4e, the levels frozen per J step): the rows' levels given, not decided -- the 1e-7 test is a discrete decision, and a non-Kerr
// b(theta) moves the BR l >= 8 rows' level -2 term across it (J 0.10: b1 ~ 4e-9), a structural jump inside a finite difference
inline Derived derive(const Sel& s, const std::vector<int>& PL, const std::vector<int>& kE, double n, const std::vector<int>* dE_used,
                      const std::vector<int>* dE_fix = nullptr) {
    int np_ = int(PL.size()), nr = s.nr, nn = int(kE.size());
    Derived d;
    d.dE.assign(nr, 99);
    if (dE_fix && int(dE_fix->size()) != nr) throw std::runtime_error("mk: op-build: frozen levels for " + std::to_string(dE_fix->size()) + " rows, the build has " + std::to_string(nr));
    auto mag = [&](int ip, int i, int j) {
        size_t a = (size_t(ip) * nr + i) * nn + j;
        return std::max({std::abs(s.A[a]), std::abs(s.B[a]), std::abs(s.C[a])});
    };
    for (int i = 0; i < nr; i++) {
        double rmax = 0.0;
        for (int ip = 0; ip < np_; ip++)
            for (int j = 0; j < nn; j++) rmax = std::max(rmax, mag(ip, i, j));
        rmax = std::max(rmax, 1e-300);
        for (int j = 0; j < nn; j++) {
            int pm = 99;
            for (int ip = 0; ip < np_ && pm == 99; ip++)
                if (mag(ip, i, j) > 1e-7 * rmax) pm = PL[ip];
            if (pm < 99) d.dE[i] = std::min(d.dE[i], pm + kE[j]);
        }
    }
    if (dE_fix) d.dE = *dE_fix;
    auto lead = [&](const std::vector<int>& dE_) {
        std::vector<cd> M(size_t(nr) * nn, 0.0);
        cd sig(0.37, 0.11);
        for (int i = 0; i < nr; i++)
            for (int j = 0; j < nn; j++) {
                int lev = dE_[i] - kE[j];
                for (int ip = 0; ip < np_; ip++)
                    if (PL[ip] == lev) {
                        cd sj = n * (sig + double(kE[j]));
                        size_t a = (size_t(ip) * nr + i) * nn + j;
                        M[size_t(i) * nn + j] = s.A[a] + sj * s.B[a] + sj * sj * s.C[a];
                    }
            }
        return M;
    };
    auto M1 = lead(d.dE);
    d.rw.assign(nr, 0.0);
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nn; j++) d.rw[i] = std::max(d.rw[i], std::abs(M1[size_t(i) * nn + j]));
        if (d.rw[i] == 0.0) d.rw[i] = 1.0;
    }
    d.cw.assign(nn, 0.0);
    for (int j = 0; j < nn; j++) {
        for (int i = 0; i < nr; i++) d.cw[j] = std::max(d.cw[j], std::abs(M1[size_t(i) * nn + j]) / d.rw[i]);
        if (d.cw[j] == 0.0) d.cw[j] = 1.0;
    }
    if (dE_used) {
        auto Mu = lead(*dE_used);
        d.rw_out.assign(nr, 0.0);
        for (int i = 0; i < nr; i++) {
            double r = 0.0;
            for (int j = 0; j < nn; j++) r = std::max(r, std::abs(Mu[size_t(i) * nn + j]));
            d.rw_out[i] = r > 0 ? r : d.rw[i];
        }
    }
    return d;
}

}  // namespace ob
}  // namespace mk

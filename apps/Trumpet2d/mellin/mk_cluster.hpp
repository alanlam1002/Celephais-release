// Mellin throat recursion, the stage (ii) port (code round 463, research rounds 1056-1057; spinning_trumpet NOTES_M4_DESIGN.md section 24):
// a CLUSTER step's data paths -- n2a_nl.py's clus_map / nodes_of / moments_of / with_grade / with_grade_mom / strip at a cluster key, where
// the step's content lives at its storage poles' exact exponents (offsets dm from the key), each pole's principal part K log moments, STACKED
// (poles x K).  The registry (Ctx::clus / clroots / clabs) and the stacked readout (mk_operator.hpp piece_at_cl, project) are the other half.
//
// Not bitwise: the clusters are gated at 10 x max(the oracle's spread, the native self-spread) (research round 1057).
#pragma once
#include "mk_operator.hpp"

#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

namespace mk {
namespace cl {

using Vec = std::vector<cd>;

inline double comb(int n, int k) {
    double r = 1.0;
    for (int i = 1; i <= k; i++) r = r * (n - k + i) / i;
    return r;
}

// the members' offsets from cluster key g (n2a_nl clus_mem; empty with ok = false: not a cluster key)
inline std::vector<double> clus_mem(const Ctx& c, double g, bool& ok) {
    ok = false;
    if (c.clus.empty() || c.noclus) return {};
    double key;
    std::vector<double> mem;
    if (!c.clus_at(g, key, mem)) return {};
    ok = true;
    for (auto& m : mem) m -= key;
    return mem;
}

// n2a_nl clus_map: the contour moments about the key (n = 0 .. NM-1, scaled by R^-n) of unit poles of order m + 1 at the offsets dm:
// mu_n = C(n, m) d^(n - m) / R^(n - m) (n >= m); NM = min(3 L K, N / 2); A row-major (NM x L K)
inline std::vector<double> clus_map(const std::vector<double>& dm, int K, int N, double R, int& NM) {
    int L = int(dm.size());
    NM = std::min(3 * L * K, N / 2);
    std::vector<double> A(size_t(NM) * L * K, 0.0);
    for (int i = 0; i < L; i++)
        for (int m = 0; m < K; m++)
            for (int n = m; n < NM; n++) A[size_t(n) * L * K + i * K + m] = comb(n, m) * std::pow(dm[i], n - m) / std::pow(R, n - m);
    return A;
}

// stacked moments (L K x NB) -> the sum of the poles' principal parts at the contour's nodes (N x NB)
inline Vec nodes_of(const Vec& Mom, const std::vector<double>& dm, int K, int NB, const Vec& e1, double R) {
    int N = int(e1.size()), L = int(dm.size());
    Vec X(size_t(N) * NB, 0.0);
    for (int k = 0; k < N; k++) {
        cd z = R * e1[k];
        for (int i = 0; i < L; i++) {
            cd zd = z - dm[i], p = zd;
            for (int m = 0; m < K; m++) {
                for (int j = 0; j < NB; j++) X[size_t(k) * NB + j] += Mom[size_t(i * K + m) * NB + j] / p;
                p *= zd;
            }
        }
    }
    return X;
}

// node values (N x NB) -> the poles' stacked moments (L K x NB): the contour moments about the key, least squares on clus_map (scaled columns),
// back from the scaled map (c = c' R^m)
inline Vec moments_of(const Vec& X, const std::vector<double>& dm, int K, int NB, const Vec& e1, double R) {
    int N = int(e1.size()), L = int(dm.size()), LK = L * K, NM;
    auto A = clus_map(dm, K, N, R, NM);
    Vec mu(size_t(NM) * NB, 0.0);
    for (int n = 0; n < NM; n++) {
        double Rn = std::pow(R, n);
        for (int j = 0; j < NB; j++) {
            cd s = 0.0;
            for (int k = 0; k < N; k++) s += std::pow(R * e1[k], n + 1) * X[size_t(k) * NB + j];
            mu[size_t(n) * NB + j] = s / double(N) / Rn;
        }
    }
    std::vector<double> cn(LK, 0.0);
    for (int q = 0; q < LK; q++) {
        double s = 0.0;
        for (int n = 0; n < NM; n++) s += A[size_t(n) * LK + q] * A[size_t(n) * LK + q];
        cn[q] = std::max(std::sqrt(s), 1e-300);
    }
    Vec As(size_t(NM) * LK);
    for (int n = 0; n < NM; n++)
        for (int q = 0; q < LK; q++) As[size_t(n) * LK + q] = A[size_t(n) * LK + q] / cn[q];
    auto sol = la::lstsq(As, NM, LK, mu, NB);
    Vec out(size_t(LK) * NB);
    for (int i = 0; i < L; i++)
        for (int m = 0; m < K; m++) {
            double f = std::pow(R, m) / cn[i * K + m];
            for (int j = 0; j < NB; j++) out[size_t(i * K + m) * NB + j] = sol[size_t(i * K + m) * NB + j] * f;
        }
    return out;
}

// n2a_nl with_grade_mom (and with_grade at a cluster key, after moments_of): the key's entries removed, the stacked moments Mo (L K x NB)
// written at the poles' exact exponents (t = ckey(g + d + kofs(j)) >= 1)
inline std::vector<Entry> with_grade_mom(Ctx& c, const std::vector<Entry>& Cx, const Vec& Mo, double g, const std::vector<double>& dm,
                                         const std::vector<int>& kofs) {
    int nn = int(kofs.size()), K = c.K;
    std::vector<Entry> Cy;
    std::vector<Key> kk;
    for (int j = 0; j < nn; j++) kk.push_back(ckey(c, g + kofs[j]));
    for (auto& en : Cx)
        if (!keq(en.t, kk[en.j])) Cy.push_back(en);
    for (int j = 0; j < nn; j++)
        for (size_t i = 0; i < dm.size(); i++) {
            Key t = ckey(c, g + dm[i] + kofs[j]);
            if (t.v < 1 - 1e-9) continue;
            std::vector<cd> m(K);
            for (int k = 0; k < K; k++) m[k] = Mo[size_t(i * K + k) * nn + j];
            bool found = false;
            for (auto& en : Cy)
                if (en.j == j && keq(en.t, t)) { en.nb = 1; en.m = m; found = true; break; }
            if (!found) Cy.push_back(Entry{j, t, 1, std::move(m)});
        }
    return Cy;
}

inline std::vector<Entry> with_grade(Ctx& c, const std::vector<Entry>& Cx, const Vec& X, double g, const std::vector<double>& dm,
                                     const std::vector<int>& kofs, const Vec& e1, double R) {
    return with_grade_mom(c, Cx, moments_of(X, dm, c.K, int(kofs.size()), e1, R), g, dm, kofs);
}

// n2a_nl strip at a cluster key: the key's own columns and every member's (data grades, t < 1, never stripped)
inline std::vector<Entry> strip(Ctx& c, const std::vector<Entry>& C, double g, const std::vector<double>& mem_abs, const std::vector<int>& kofs) {
    std::vector<std::vector<Key>> ks(kofs.size());
    for (size_t j = 0; j < kofs.size(); j++) {
        ks[j].push_back(ckey(c, g + kofs[j]));
        for (double m : mem_abs) ks[j].push_back(ckey(c, m + kofs[j]));
    }
    std::vector<Entry> o;
    for (auto& en : C) {
        bool drop = false;
        for (auto& t : ks[en.j])
            if (t.v >= 1 && keq(en.t, t)) { drop = true; break; }
        if (!drop) o.push_back(en);
    }
    return o;
}


// the scaled quadratic pencil's finite eigenvalues on the active set ia (offsets from the key): scipy.linalg.eig(
// [[0, I], [-Q0, -Q1]], [[I, 0], [0, Q2]], right=False), Q_m = (Ms_m / rw / cw)[ia, ia]
inline Vec pencil_roots(const Vec& Ms, int NB, const std::vector<int>& ia, const std::vector<double>& rw, const std::vector<double>& cw) {
    int nq = int(ia.size()), n2 = 2 * nq;
    size_t nm = size_t(NB) * NB;
    Vec A(size_t(n2) * n2, 0.0), B(size_t(n2) * n2, 0.0);
    for (int i = 0; i < nq; i++) {
        A[size_t(i) * n2 + nq + i] = 1.0;
        B[size_t(i) * n2 + i] = 1.0;
        for (int j = 0; j < nq; j++) {
            size_t ij = size_t(ia[i]) * NB + ia[j];
            double sc = rw[ia[i]] * cw[ia[j]];
            A[size_t(nq + i) * n2 + j] = -Ms[ij] / sc;
            A[size_t(nq + i) * n2 + nq + j] = -Ms[nm + ij] / sc;
            B[size_t(nq + i) * n2 + nq + j] = Ms[2 * nm + ij] / sc;
        }
    }
    Vec ev = la::eiggen(A, B, n2), o;
    for (auto& e : ev)
        if (std::isfinite(e.real()) && std::isfinite(e.imag())) o.push_back(e);
    return o;
}

// the bordered active inverse at z (n2a_nl local_pp's minv_at): ((Ms0 + z Ms1 + z^2 Ms2) / rw / cw)[ia, ia], bordered by unit vectors at
// bordl (local indices), inverted, the active block back to full size and unscaled (/ cw[:, None] / rw[None, :]); NB x NB
inline Vec minv_at(const Vec& Ms, int NB, cd z, const std::vector<int>& ia, const std::vector<std::pair<int, int>>& bordl,
                   const std::vector<double>& rw, const std::vector<double>& cw) {
    int na = int(ia.size()), nb_ = int(bordl.size()), n = na + nb_;
    size_t nm = size_t(NB) * NB;
    Vec Bt(size_t(n) * n, 0.0);
    for (int i = 0; i < na; i++)
        for (int j = 0; j < na; j++) {
            size_t ij = size_t(ia[i]) * NB + ia[j];
            Bt[size_t(i) * n + j] = (Ms[ij] + z * Ms[nm + ij] + z * z * Ms[2 * nm + ij]) / (rw[ia[i]] * cw[ia[j]]);
        }
    for (int q = 0; q < nb_; q++) {
        Bt[size_t(bordl[q].first) * n + na + q] = 1.0;
        Bt[size_t(na + q) * n + bordl[q].second] = 1.0;
    }
    Vec Bi = la::inv(Bt, n), M(nm, 0.0);
    for (int i = 0; i < na; i++)
        for (int j = 0; j < na; j++) M[size_t(ia[i]) * NB + ia[j]] = Bi[size_t(i) * n + j] / (cw[ia[i]] * rw[ia[j]]);
    return M;
}

// the thin-QR Q of an m x n matrix (np.linalg.qr(A)[0], mode 'reduced': m x min(m, n))
inline Vec qr_q(Vec A, int m, int n) {
    int k = std::min(m, n);
    Vec tau(k);
    la::check(LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, m, n, A.data(), n, tau.data()), "zgeqrf");
    la::check(LAPACKE_zungqr(LAPACK_ROW_MAJOR, m, k, k, A.data(), n, tau.data()), "zungqr");
    Vec Q(size_t(m) * k);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++) Q[size_t(i) * k + j] = A[size_t(i) * n + j];
    return Q;
}

// spec section 29, n2a_nl local_pp: X's principal part at each storage pole (offsets dm from the key), LOCALLY -- M^-1's Laurent
// coefficients about the pole, G_k = (1 / 2 pi i) oint M^-1 (s - p)^(-k-1) ds on a 64-node circle enclosing no other root of M (radius half
// the distance to the nearest other root, <= Rc / 2; a root within eps_s of the pole enclosed: merged, re-expanded to K orders), applied to
// F's principal part there (F stacked like the storage: L K x NB) and, for k < 0, to F's Taylor part at the pole (the other poles' parts
// expanded there); at a root (G_-1 above 1e-6 rho |G_0|) the Keldysh direction v^ = G_-1's leading left singular vector, phased (the largest
// u_g entry real positive), and the merged root's offset dl.  amp: {pole: amplitude} -- the root's datum on v^ (re-expanded: dl^m v^).
// hold (N2A_LOCALHOLD=contour, not the default): not ported.  Returns the stacked moments (L K x NB)
struct LocalInfo {
    std::vector<double> singular;                      // the root poles' offsets
    std::vector<double> roots_inside;                  // the pencil's real roots inside the contour (|e| < Rc, |Im| < 1e-7)
    std::map<int, Vec> vhat;
    std::map<int, cd> dl;
    struct RootRec { double pole, rho, ratio, gm2, gm1; };
    std::vector<RootRec> roots;
};
inline Vec local_pp(const Vec& Ms, int NB, int nn, int K, const std::vector<int>& ia, const std::vector<std::pair<int, int>>& bordl,
                    const std::vector<double>& dm, const Vec& F, double Rc, double eps_s, const std::vector<double>& rw,
                    const std::vector<double>& cw, const std::map<int, cd>* amp, LocalInfo* info) {
    int L = int(dm.size());
    Vec evp = pencil_roots(Ms, NB, ia, rw, cw);
    const int Nl = 64, kmin = -K;
    Vec om(Nl);
    for (int j = 0; j < Nl; j++) om[j] = std::exp(cd(0.0, 2 * M_PI * (j + 0.5) / Nl));
    Vec X(size_t(L) * K * NB, 0.0);
    std::map<int, Vec> vh;
    std::map<int, cd> dl;
    std::vector<int> roots;
    auto matvec = [&](const Vec& G, const cd* x) {
        Vec y(NB, 0.0);
        for (int i = 0; i < NB; i++) {
            cd s = 0.0;
            for (int j = 0; j < NB; j++) s += G[size_t(i) * NB + j] * x[j];
            y[i] = s;
        }
        return y;
    };
    for (int i = 0; i < L; i++) {
        double d = dm[i];
        std::vector<double> near, oth;
        for (auto& e : evp) (std::abs(e - d) < eps_s ? near : oth).push_back(std::abs(e - d));
        double mo = oth.empty() ? 0.0 : *std::min_element(oth.begin(), oth.end());
        double rho = std::min(oth.empty() ? Rc / 2 : 0.5 * mo, Rc / 2);
        if (!near.empty()) {
            double mn = *std::max_element(near.begin(), near.end());
            if (rho < 2 * mn) rho = 0.5 * (mn + (oth.empty() ? Rc : mo));
        }
        std::vector<Vec> Mi(Nl);
        for (int j = 0; j < Nl; j++) Mi[j] = minv_at(Ms, NB, d + rho * om[j], ia, bordl, rw, cw);
        std::map<int, Vec> G;
        for (int k = kmin; k < K; k++) {
            Vec g(size_t(NB) * NB, 0.0);
            for (int j = 0; j < Nl; j++) {
                cd f = std::pow(rho * om[j], -k);
                for (size_t q = 0; q < g.size(); q++) g[q] += Mi[j][q] * f;
            }
            for (auto& x : g) x /= double(Nl);
            G[k] = std::move(g);
        }
        auto gmax = [](const Vec& v) { double m = 0.0; for (auto& z : v) m = std::max(m, std::abs(z)); return m; };
        bool sing = gmax(G[-1]) > 1e-6 * rho * std::max(gmax(G[0]), 1e-300);
        std::vector<Vec> Ft(K, Vec(NB, 0.0));
        for (int q = 0; q < L; q++) {
            if (q == i) continue;
            for (int j = 0; j < K; j++)
                for (int t = 0; t < K; t++) {
                    double f = ((t % 2) ? -1.0 : 1.0) * comb(j + t, t) * std::pow(d - dm[q], -(j + 1 + t));
                    for (int c_ = 0; c_ < NB; c_++) Ft[t][c_] += f * F[size_t(q * K + j) * NB + c_];
                }
        }
        for (int m = 0; m < K; m++) {
            Vec acc(NB, 0.0);
            for (int k = kmin; k < K; k++) {
                int j = m + k;
                if (j >= 0 && j < K) {
                    auto y = matvec(G[k], &F[size_t(i * K + j) * NB]);
                    for (int c_ = 0; c_ < NB; c_++) acc[c_] += y[c_];
                } else if (j < 0 && -(j + 1) < K) {
                    auto y = matvec(G[k], Ft[-(j + 1)].data());
                    for (int c_ = 0; c_ < NB; c_++) acc[c_] += y[c_];
                }
            }
            for (int c_ = 0; c_ < NB; c_++) X[size_t(i * K + m) * NB + c_] = acc[c_];
        }
        if (sing) {
            std::vector<double> sv;
            Vec U, Vh;
            la::svd(G[-1], NB, NB, sv, &U, &Vh, true);
            Vec v(NB);
            for (int r = 0; r < NB; r++) v[r] = U[size_t(r) * NB + 0];
            int im = 0;
            for (int r = 1; r < nn; r++)
                if (std::abs(v[r]) > std::abs(v[im])) im = r;
            cd ph = std::exp(cd(0.0, -std::arg(v[im])));
            for (auto& z : v) z *= ph;
            vh[i] = v;
            roots.push_back(i);
            cd best = 0.0;
            double bd = 1e300;
            bool have = false;
            for (auto& e : evp)
                if (std::abs(e - d) < eps_s && std::abs(e - d) < bd) { bd = std::abs(e - d); best = e; have = true; }
            dl[i] = have ? best - d : cd(0.0, 0.0);
            if (info) info->roots.push_back({d, rho, sv.size() > 1 ? sv[1] / sv[0] : 0.0, gmax(G[-2]), sv[0]});
        }
    }
    if (amp)
        for (auto& kv : *amp) {
            int i = kv.first;
            if (!vh.count(i)) {
                std::fprintf(stdout, "    [local_pp] a datum at the pole %+.6f, where M is regular: not applied\n", dm[i]);
                continue;
            }
            for (int m = 0; m < K; m++) {
                cd f = kv.second * std::pow(dl[i], m);
                for (int c_ = 0; c_ < NB; c_++) X[size_t(i * K + m) * NB + c_] += f * vh[i][c_];
            }
        }
    if (info) {
        info->singular.clear();
        for (int i : roots) info->singular.push_back(dm[i]);
        info->roots_inside.clear();
        for (auto& e : evp)
            if (std::abs(e) < Rc && std::abs(e.imag()) < 1e-7) info->roots_inside.push_back(e.real());
        std::sort(info->roots_inside.begin(), info->roots_inside.end());
        info->vhat = vh;
        info->dl = dl;
    }
    return X;
}

}  // namespace cl
}  // namespace mk

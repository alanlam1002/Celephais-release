// Mellin throat recursion (code round 463; research rounds 1187-1189, S2b): the l = 2 SLICING FLOW -- a port of spinning_trumpet/scripts/
// slice_s1.py's build (the generator G[F; a = 1] of one lapse family) and of scripts/s2b_flow.py's flow (F = F0 + a G + (a^2 / 2) DG.G), operation
// by operation, in the kernel's series algebra (the Python prototype is the oracle, to round-off).
//   a. the cylinder's Sturm-Liouville problem  d/dx( g (1 - x^2) Y' ) + mu g Y = 0, g = P^2 e^(-2u): mu_j, Y_j in the even-Legendre basis (30
//      functions, 400 Gauss-Legendre nodes), Y_j normalised to a unit coefficient on its own P_(2j); s_k = -n + sqrt(n^2 + mu_k);
//   b. the tilt h = r^(s_k) [Y_k + tower], div(lambda alpha^2 grad h) = 0 key by key (the divisor per SL mode n^2 g (g + 2) - mu_j, through the log
//      moments; an exact zero solved through the moments >= 1 by least squares);
//   c. zeta = h beta + zeta_1, the conformal Killing operator of zeta_1 = -h (that of beta), per Fourier mode;
//   d. G from SLICING_FAMILIES section 5's six formulas in the engine's variables (psi2, Phb, qf, br, bt, Qb).
// Every operation sequenced left to right as Python evaluates it (products intern exponents as they arrive).
#pragma once
#include "mk_families.hpp"
#include "mk_la.hpp"

#include <cmath>
#include <vector>

namespace mk {
namespace sl {

using Vec = std::vector<cd>;

struct Modes {                                       // slice_s1.sl_modes
    int nl = 30, nq = 400;
    std::vector<double> mu;                          // ascending
    std::vector<double> V;                           // (nl x nl), V[a * nl + j]: Legendre P_(2a) coefficient of Y_j
    std::vector<double> xq, wq, gq;
};

inline double legP(int l, double x) {
    double p0 = 1, p1 = x;
    if (l == 0) return 1;
    for (int k = 2; k <= l; k++) { double p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
    return p1;
}

inline Modes sl_modes(const std::vector<double>& u, const std::vector<double>& P, int nl = 30, int nq = 400) {
    Modes M;
    M.nl = nl; M.nq = nq;
    M.xq.resize(nq); M.wq.resize(nq); M.gq.resize(nq);
    for (int i = 0; i < nq; i++) {                                 // Gauss-Legendre (ascending, as numpy's leggauss)
        double x = -std::cos(M_PI * (i + 0.75) / (nq + 0.5)), dp = 0;
        for (int it = 0; it < 100; it++) {
            double p0 = 1, p1 = x;
            for (int k = 2; k <= nq; k++) { double p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
            dp = nq * (x * p1 - p0) / (x * x - 1);
            double dx = p1 / dp;
            x -= dx;
            if (std::abs(dx) < 1e-16) break;
        }
        M.xq[i] = x; M.wq[i] = 2 / ((1 - x * x) * dp * dp);
    }
    const int L = 2 * nl;
    std::vector<double> Pv(size_t(nl) * nq), dP(size_t(nl) * nq);
    for (int i = 0; i < nq; i++) {
        double x = M.xq[i], th = std::acos(x), uu = 0, pp = 0;
        for (size_t j = 0; j < u.size(); j++) uu += u[j] * std::cos(2.0 * j * th);
        for (size_t j = 0; j < P.size(); j++) pp += P[j] * std::cos(2.0 * j * th);
        M.gq[i] = pp * pp * std::exp(-2 * uu);
        std::vector<double> pl(L + 1), dl(L + 1);
        pl[0] = 1; pl[1] = x; dl[0] = 0; dl[1] = 1;
        for (int k = 2; k <= L; k++) { pl[k] = ((2 * k - 1) * x * pl[k - 1] - (k - 1) * pl[k - 2]) / k; dl[k] = dl[k - 2] + (2 * k - 1) * pl[k - 1]; }
        for (int a = 0; a < nl; a++) { Pv[size_t(a) * nq + i] = pl[2 * a]; dP[size_t(a) * nq + i] = dl[2 * a]; }
    }
    std::vector<double> A(size_t(nl) * nl), B(size_t(nl) * nl);
    for (int a = 0; a < nl; a++)
        for (int b = 0; b < nl; b++) {
            double sa = 0, sb = 0;
            for (int i = 0; i < nq; i++) {
                sa += dP[size_t(a) * nq + i] * dP[size_t(b) * nq + i] * M.gq[i] * (1 - M.xq[i] * M.xq[i]) * M.wq[i];
                sb += Pv[size_t(a) * nq + i] * Pv[size_t(b) * nq + i] * M.gq[i] * M.wq[i];
            }
            A[size_t(a) * nl + b] = sa; B[size_t(a) * nl + b] = sb;
        }
    std::vector<double> w(nl);
    la::check(LAPACKE_dsygv(LAPACK_ROW_MAJOR, 1, 'V', 'U', nl, A.data(), nl, B.data(), nl, w.data()), "dsygv (slicing modes)");
    M.mu = w;                                                       // ascending; A holds the eigenvectors (columns)
    M.V = A;
    for (int j = 0; j < nl; j++) {                                  // Y_j normalised to a unit coefficient on its own P_(2j)
        double s = M.V[size_t(j) * nl + j];
        for (int a = 0; a < nl; a++) M.V[size_t(a) * nl + j] /= s;
    }
    return M;
}

struct Grid {                                        // n2a_nl.setup's Fourier grid: mk (Nyquist dropped), Fm (m, j) = e^(-i m th_j) / NF
    std::vector<double> mk;
    Vec Fm;                                          // (nm x NF)
    int nm = 0;
};
inline Grid grid(int NF, const std::vector<double>& TH) {
    Grid g;
    for (int i = 0; i < NF; i++) {                                  // np.fft.fftfreq(NF, 1 / NF), |m| < NF / 2
        int m = i < (NF + 1) / 2 ? i : i - NF;
        if (std::abs(m) < NF / 2) g.mk.push_back(m);
    }
    g.nm = int(g.mk.size());
    g.Fm.resize(size_t(g.nm) * NF);
    for (int a = 0; a < g.nm; a++)
        for (int j = 0; j < NF; j++) g.Fm[size_t(a) * NF + j] = std::conj(std::exp(cd(0.0, g.mk[a] * TH[j])) / double(NF));
    return g;
}

struct Rep {
    double s_k = 0, sigma = 0, h_res = 0, zeta_res = 0;
    int h_iters = 0;
};

// G[F; a = 1] of the lapse family l (slice_s1.build); F: psi2 Phb qf br bt Qb (the engine's fields, gauge-flowed); Ph: the cylinder's lapse
// constant K (C0(Ph / r)); Pg: the cylinder's lapse shape on the grid; TOP: the generator's top grade
inline void build(Ctx& c, const Ser F[6], const std::vector<double>& TH, const Modes& M, double Ph, double J, double TOP, int lfam, Ser G[6], Rep& rep,
                  Ser* hout = nullptr) {
    const int K = c.K, NF = c.NF, nl = M.nl, nq = M.nq;
    const double n = c.n;
    const Grid gr = grid(NF, TH);
    const int k = lfam / 2;
    const double s_k = -n + std::sqrt(n * n + M.mu[k]);
    rep.s_k = s_k;
    rep.sigma = 1 + s_k / n;
    // Pgrid (modes x NF) on the engine's grid; Ygrid = V^T Pgrid; at the nodes Yq; Nj = <Y_j, Y_j>_g; Iq: grid -> nodes (nq x NF)
    std::vector<double> Ygrid(size_t(nl) * NF, 0.0), Yq(size_t(nl) * nq, 0.0), Nj(nl, 0.0), sinq(nq);
    for (int q = 0; q < NF; q++) {
        const double x = std::cos(TH[q]);
        std::vector<double> pa(nl);
        for (int a = 0; a < nl; a++) pa[a] = legP(2 * a, x);
        for (int j = 0; j < nl; j++) {
            double s = 0;
            for (int a = 0; a < nl; a++) s += M.V[size_t(a) * nl + j] * pa[a];
            Ygrid[size_t(j) * NF + q] = s;
        }
    }
    for (int i = 0; i < nq; i++) {
        std::vector<double> pa(nl);
        for (int a = 0; a < nl; a++) pa[a] = legP(2 * a, M.xq[i]);
        for (int j = 0; j < nl; j++) {
            double s = 0;
            for (int a = 0; a < nl; a++) s += M.V[size_t(a) * nl + j] * pa[a];
            Yq[size_t(j) * nq + i] = s;
        }
        sinq[i] = std::sqrt(1 - M.xq[i] * M.xq[i]);
    }
    for (int j = 0; j < nl; j++) {
        double s = 0;
        for (int i = 0; i < nq; i++) s += Yq[size_t(j) * nq + i] * Yq[size_t(j) * nq + i] * M.gq[i] * M.wq[i];
        Nj[j] = s;
    }
    Vec Iq(size_t(nq) * NF, 0.0);                                   // interp(arccos xq) = exp(i outer(th_t, mk)) @ Fm
    for (int i = 0; i < nq; i++) {
        const double tt = std::acos(M.xq[i]);
        for (int a = 0; a < gr.nm; a++) {
            const cd e = std::exp(cd(0.0, tt * gr.mk[a]));
            for (int q = 0; q < NF; q++) Iq[size_t(i) * NF + q] += e * gr.Fm[size_t(a) * NF + q];
        }
    }
    auto cser = [&](int e, double v) { return c.const_ser(e, std::vector<cd>(NF, cd(v, 0.0))); };
    auto tser = [&](const std::vector<double>& vals) { return fm::theta_ser(c, vals); };
    auto piece = [&](double g, const std::vector<cd>& v) {          // MS({g: v}, 0, TOP) with v (K x NF)
        Piece p = c.zeros(1);
        p.a = v;
        std::vector<std::pair<Key, Piece>> d;
        const double gi = std::nearbyint(g);
        d.emplace_back(Key{std::abs(g - gi) < 1e-9 ? gi : g, std::abs(g - gi) < 1e-9}, std::move(p));
        return c.make(std::move(d), true, 0, TOP);
    };
    std::vector<double> sin_g(NF), cot_g(NF);
    for (int q = 0; q < NF; q++) { sin_g[q] = std::sin(TH[q]); cot_g[q] = std::cos(TH[q]) / std::sin(TH[q]); }
    const Ser rr = cser(1, 1.0), rinv = cser(-1, 1.0), rinv2 = cser(-2, 1.0);
    const Ser sinp = tser(sin_g), cotp = tser(cot_g);
    const Ser& psi2 = F[0]; const Ser& Phb = F[1]; const Ser& qf = F[2]; const Ser& br = F[3]; const Ser& bt = F[4]; const Ser& Qb = F[5];
    // ---- b. the tilt h
    Ser W;
    {
        Ser a1 = c.mul(rr, sinp);
        Ser a2 = c.mul(a1, Phb);
        Ser a3 = c.mul(a2, Phb);
        Ser ip = c.inv(psi2);
        W = c.mul(a3, ip);
    }
    const cd Kc = cd(Ph * Ph, 0.0);
    auto Lop = [&](const Ser& h) {
        Ser hr = c.drad(h);
        Ser t1 = c.mul(rr, W);
        Ser t2 = c.mul(t1, hr);
        Ser t3 = c.drad(t2);
        Ser A = c.mul(rinv, t3);
        Ser ht = c.dth(h);
        Ser u1 = c.mul(W, ht);
        Ser u2 = c.dth(u1);
        Ser B = c.mul(rinv2, u2);
        return c.add(A, B);
    };
    const double g0 = s_k / n;
    std::vector<cd> hp(size_t(K) * NF, 0.0);
    for (int q = 0; q < NF; q++) hp[q] = Ygrid[size_t(k) * NF + q];
    Ser h = piece(g0, hp);
    Ser R = Lop(h);
    std::vector<double> done;
    auto isdone = [&](double v) { for (double x : done) if (x == v) return true; return false; };
    int it = 0;
    for (it = 0; it < 400; it++) {
        double kk = 0;
        bool have = false;
        for (auto& kv : R.d) {
            const double v = kv.first.v;
            if (!(v - 2 <= TOP + 1e-9) || isdone(v)) continue;
            double mx = 0;
            for (auto& z : kv.second.a) mx = std::max(mx, std::abs(z));
            if (!(mx > 1e-13 * std::max(1.0, std::abs(Kc)))) continue;
            if (!have || v < kk) { kk = v; have = true; }
        }
        if (!have) break;
        done.push_back(kk);
        const double g = kk - 2.0;
        const Piece& Sp = R.d[R.find(Key{kk, false})].second;
        // Sq = (S @ Iq.T) / (Kc sinq gq): (K x nq); cjm = sum_q Sq Yq gq wq / Nj: (K x nl)
        std::vector<cd> Sq(size_t(K) * nq, 0.0), cjm(size_t(K) * nl, 0.0);
        for (int m = 0; m < K; m++)
            for (int i = 0; i < nq; i++) {
                cd s = 0.0;
                for (int q = 0; q < NF; q++) s += -Sp.a[size_t(m) * NF + q] * Iq[size_t(i) * NF + q];
                Sq[size_t(m) * nq + i] = s / (Kc * sinq[i] * M.gq[i]);
            }
        for (int m = 0; m < K; m++)
            for (int j = 0; j < nl; j++) {
                cd s = 0.0;
                for (int i = 0; i < nq; i++) s += Sq[size_t(m) * nq + i] * Yq[size_t(j) * nq + i] * (M.gq[i] * M.wq[i]);
                cjm[size_t(m) * nl + j] = s / Nj[j];
            }
        std::vector<cd> av(size_t(K) * nl, 0.0);
        for (int j = 0; j < nl; j++) {
            double cm = 0;
            for (int m = 0; m < K; m++) cm = std::max(cm, std::abs(cjm[size_t(m) * nl + j]));
            if (cm == 0) continue;
            // M = n^2 (g I + N)((g + 2) I + N) - mu_j I, N the shift (N v)_m = v_(m+1)
            std::vector<cd> A1(size_t(K) * K, 0.0), A2(size_t(K) * K, 0.0), Mm(size_t(K) * K, 0.0);
            for (int a = 0; a < K; a++) {
                A1[size_t(a) * K + a] = g; A2[size_t(a) * K + a] = g + 2;
                if (a + 1 < K) { A1[size_t(a) * K + a + 1] = 1.0; A2[size_t(a) * K + a + 1] = 1.0; }
            }
            for (int a = 0; a < K; a++)
                for (int b = 0; b < K; b++) {
                    cd s = 0.0;
                    for (int q = 0; q < K; q++) s += A1[size_t(a) * K + q] * A2[size_t(q) * K + b];
                    Mm[size_t(a) * K + b] = n * n * s - (a == b ? M.mu[j] : 0.0);
                }
            const double dj = n * n * g * (g + 2) - M.mu[j];
            std::vector<cd> rhs(K);
            for (int m = 0; m < K; m++) rhs[m] = cjm[size_t(m) * nl + j];
            if (std::abs(dj) < 1e-9) {                              // exact resonance: no homogeneous part, through the log moments
                std::vector<cd> M1(size_t(K) * (K - 1));
                for (int a = 0; a < K; a++)
                    for (int b = 1; b < K; b++) M1[size_t(a) * (K - 1) + b - 1] = Mm[size_t(a) * K + b];
                auto x = la::lstsq_rc(M1, K, K - 1, rhs, 1, std::numeric_limits<double>::epsilon() * K);
                for (int b = 1; b < K; b++) av[size_t(b) * nl + j] = x[b - 1];
            } else {
                auto x = la::solve(Mm, K, rhs, 1);
                for (int m = 0; m < K; m++) av[size_t(m) * nl + j] = x[m];
            }
        }
        std::vector<cd> v(size_t(K) * NF, 0.0);                     // v = a @ Ygrid
        for (int m = 0; m < K; m++)
            for (int j = 0; j < nl; j++) {
                const cd a_ = av[size_t(m) * nl + j];
                if (a_ == cd(0.0)) continue;
                for (int q = 0; q < NF; q++) v[size_t(m) * NF + q] += a_ * Ygrid[size_t(j) * NF + q];
            }
        Ser dh = piece(g, v);
        h = c.add(h, dh);
        Ser Ld = Lop(dh);
        R = c.add(R, Ld);
    }
    rep.h_iters = it;
    for (auto& kv : R.d)
        if (kv.first.v - 2 <= TOP)
            for (auto& z : kv.second.a) rep.h_res = std::max(rep.h_res, std::abs(z));
    if (hout) *hout = h;
    // ---- c. zeta = h beta + zeta_1
    auto ckv = [&](const Ser& zr, const Ser& zt, Ser& A1, Ser& A2) {
        Ser a = c.drad(zr);
        Ser b = c.mul(rinv, zr);
        Ser ab = c.sub(a, b);
        Ser cth = zt.empty() ? c.emptyser() : c.dth(zt);
        A1 = c.sub(ab, cth);
        Ser d1;
        if (zt.empty()) d1 = c.emptyser();
        else { Ser r2 = c.mul(rr, rr); Ser zd = c.drad(zt); d1 = c.mul(r2, zd); }
        Ser d2 = c.dth(zr);
        A2 = c.add(d1, d2);
    };
    Ser C1, C2;
    ckv(br, bt, C1, C2);
    Ser T1, T2;
    { Ser x = c.mul(h, C1); T1 = c.scale(x, -1.0); Ser y = c.mul(h, C2); T2 = c.scale(y, -1.0); }
    Ser z1r = c.emptyser(), z1t = c.emptyser();
    std::vector<double> keys1;
    for (auto* T : {&T1, &T2})
        for (auto& kv : T->d) {
            bool f = false;
            for (double x : keys1) f = f || x == kv.first.v;
            if (!f) keys1.push_back(kv.first.v);
        }
    std::sort(keys1.begin(), keys1.end());
    for (double gk : keys1) {
        if (gk > TOP + 1e-9) continue;
        std::vector<cd> S1(size_t(K) * NF, 0.0), S2(size_t(K) * NF, 0.0);
        { int i1 = T1.find(Key{gk, false}); if (i1 >= 0) S1 = T1.d[i1].second.a; int i2 = T2.find(Key{gk, false}); if (i2 >= 0) S2 = T2.d[i2].second.a; }
        // s1m = S1 @ Fm.T (K x nm)
        std::vector<cd> s1m(size_t(K) * gr.nm, 0.0), s2m(size_t(K) * gr.nm, 0.0);
        for (int m = 0; m < K; m++)
            for (int a = 0; a < gr.nm; a++) {
                cd s1 = 0.0, s2 = 0.0;
                for (int q = 0; q < NF; q++) { s1 += S1[size_t(m) * NF + q] * gr.Fm[size_t(a) * NF + q]; s2 += S2[size_t(m) * NF + q] * gr.Fm[size_t(a) * NF + q]; }
                s1m[size_t(m) * gr.nm + a] = s1; s2m[size_t(m) * gr.nm + a] = s2;
            }
        std::vector<cd> Am(size_t(K) * gr.nm, 0.0), Bm(size_t(K) * gr.nm, 0.0);
        const int eb = !T1.empty() ? T1.e : T2.e - 1;
        for (int a = 0; a < gr.nm; a++) {
            double mx1 = 0, mx2 = 0;
            for (int m = 0; m < K; m++) { mx1 = std::max(mx1, std::abs(s1m[size_t(m) * gr.nm + a])); mx2 = std::max(mx2, std::abs(s2m[size_t(m) * gr.nm + a])); }
            if (mx1 < 1e-15 && mx2 < 1e-15) continue;
            const double mm = gr.mk[a];
            const int K2 = 2 * K;
            std::vector<cd> Mz(size_t(K2) * K2, 0.0), rhs(K2);
            for (int r = 0; r < K; r++) {
                for (int s = 0; s < K; s++) {
                    cd Ee = (r == s ? cd(eb + n * gk, 0.0) : cd(0.0)) + (s == r + 1 ? cd(n, 0.0) : cd(0.0));
                    Mz[size_t(r) * K2 + s] = Ee;
                    Mz[size_t(K + r) * K2 + K + s] = Ee;
                }
                Mz[size_t(r) * K2 + K + r] = cd(0.0, -mm);
                Mz[size_t(K + r) * K2 + r] = cd(0.0, mm);
                rhs[r] = s1m[size_t(r) * gr.nm + a];
                rhs[K + r] = s2m[size_t(r) * gr.nm + a];
            }
            const double e0 = eb + n * gk, dz = e0 * e0 - mm * mm;
            std::vector<cd> sol = std::abs(dz) < 1e-9 ? la::lstsq_rc(Mz, K2, K2, rhs, 1, std::numeric_limits<double>::epsilon() * K2)
                                                      : la::solve(Mz, K2, rhs, 1);
            for (int m = 0; m < K; m++) { Am[size_t(m) * gr.nm + a] = sol[m]; Bm[size_t(m) * gr.nm + a] = sol[K + m]; }
        }
        // MS({gk: Am @ Einv.T}, eb + 1, TOP), Einv = exp(i outer(TH, mk))
        std::vector<cd> va(size_t(K) * NF, 0.0), vb(size_t(K) * NF, 0.0);
        for (int m = 0; m < K; m++)
            for (int q = 0; q < NF; q++) {
                cd sa = 0.0, sb = 0.0;
                for (int a = 0; a < gr.nm; a++) {
                    const cd e = std::exp(cd(0.0, TH[q] * gr.mk[a]));
                    sa += Am[size_t(m) * gr.nm + a] * e;
                    sb += Bm[size_t(m) * gr.nm + a] * e;
                }
                va[size_t(m) * NF + q] = sa; vb[size_t(m) * NF + q] = sb;
            }
        auto mk_ = [&](const std::vector<cd>& v, int e) {
            Piece p = c.zeros(1);
            p.a = v;
            std::vector<std::pair<Key, Piece>> d;
            const double gi = std::nearbyint(gk);
            d.emplace_back(Key{std::abs(gk - gi) < 1e-9 ? gi : gk, std::abs(gk - gi) < 1e-9}, std::move(p));
            return c.make(std::move(d), true, e, TOP);
        };
        Ser za = mk_(va, eb + 1), zb = mk_(vb, eb);
        z1r = c.add(z1r, za);
        z1t = c.add(z1t, zb);
    }
    Ser zr, zt;
    { Ser a = c.mul(h, br); zr = c.add(a, z1r); }
    { Ser a = bt.empty() ? c.emptyser() : c.mul(h, bt); zt = c.add(a, z1t); }
    {
        Ser A1c, A2c;
        ckv(z1r, z1t, A1c, A2c);
        Ser r1 = c.sub(A1c, T1), r2 = c.sub(A2c, T2);
        for (auto* r : {&r1, &r2})
            for (auto& kv : r->d)
                for (auto& z : kv.second.a) rep.zeta_res = std::max(rep.zeta_res, std::abs(z));
    }
    // ---- d. G (section 5)
    auto zd = [&](const Ser& f) {                                   // (zr * f.Drad() + (zt * f.Dth() if zt.d else 0)) if f.d else 0
        if (f.empty()) return c.emptyser();
        Ser fr = c.drad(f);
        Ser a = c.mul(zr, fr);
        Ser b;
        if (zt.empty()) b = c.emptyser();
        else { Ser ft = c.dth(f); b = c.mul(zt, ft); }
        return c.add(a, b);
    };
    Ser hr = c.drad(h), ht = c.dth(h);
    Ser bh;
    { Ser a = c.mul(br, hr); Ser b = bt.empty() ? c.emptyser() : c.mul(bt, ht); bh = c.add(a, b); }
    Ser ipsi = c.inv(psi2);
    Ser zdpsi = zd(psi2);
    Ser zdpsi_ip = c.mul(zdpsi, ipsi);
    Ser dlpsi;
    { Ser a = c.mul(rinv, zr); Ser ab = c.add(zdpsi_ip, a); Ser b = c.mul(zt, cotp); dlpsi = c.add(ab, b); }
    Ser divz;
    { Ser a = c.mul(rr, zr); Ser ad = c.drad(a); Ser b = c.mul(rinv, ad); Ser d2 = c.dth(zt); divz = c.add(b, d2); }
    Ser dlpb;
    { Ser a = c.scale(bh, -1.0); Ser b = c.scale(zdpsi_ip, 2.0); Ser ab = c.add(a, b); Ser zq = zd(qf); Ser d2 = c.scale(zq, 2.0); Ser abd = c.add(ab, d2); dlpb = c.add(abd, divz); }
    Ser dq;
    { Ser a = c.scale(dlpb, 0.5); dq = c.sub(a, dlpsi); }
    Ser dlPh;
    { Ser zP = zd(Phb); Ser iP = c.inv(Phb); Ser b = c.mul(zP, iP); Ser ab = c.add(bh, b); Ser ab2 = c.sub(ab, zdpsi_ip); dlPh = c.add(ab2, dlpsi); }
    Ser e2q;
    { Ser q2 = c.scale(qf, 2.0); e2q = c.expser(q2); }
    Ser a2_pb2;
    { Ser a = c.mul(Phb, Phb); Ser b = c.mul(a, ipsi); Ser d = c.mul(b, ipsi); Ser e = c.mul(d, ipsi); Ser f = c.mul(e, ipsi); Ser ie = c.inv(e2q); a2_pb2 = c.mul(f, ie); }
    Ser dbr;
    {
        Ser a = c.mul(a2_pb2, hr);
        Ser b = c.mul(br, bh);
        Ser ab = c.add(a, b);
        Ser zb = zd(br);
        Ser abz = c.add(ab, zb);
        Ser zrr = c.drad(zr);
        Ser d = c.mul(br, zrr);
        Ser abzd = c.sub(abz, d);
        Ser e;
        if (bt.empty()) e = c.emptyser();
        else { Ser zrt = c.dth(zr); e = c.mul(bt, zrt); }
        dbr = c.sub(abzd, e);
    }
    Ser dbt;
    {
        Ser a0 = c.mul(a2_pb2, rinv2);
        Ser a = c.mul(a0, ht);
        Ser b = bt.empty() ? c.emptyser() : c.mul(bt, bh);
        Ser ab = c.add(a, b);
        Ser zb = bt.empty() ? c.emptyser() : zd(bt);
        Ser abz = c.add(ab, zb);
        Ser ztr = c.drad(zt);
        Ser d = c.mul(br, ztr);
        Ser abzd = c.sub(abz, d);
        Ser e;
        if (bt.empty()) e = c.emptyser();
        else { Ser ztt = c.dth(zt); e = c.mul(bt, ztt); }
        dbt = c.sub(abzd, e);
    }
    Ser dQ;
    {
        std::vector<double> s3(NF);
        for (int q = 0; q < NF; q++) s3[q] = 6 * J * std::pow(std::sin(TH[q]), 3.0);
        Ser s3p = tser(s3);
        Ser a;
        if (Qb.empty()) a = c.emptyser();
        else { Ser qr = c.drad(Qb); a = c.mul(zr, qr); }
        Ser qt = Qb.empty() ? c.emptyser() : c.dth(Qb);
        Ser qs = c.add(qt, s3p);
        Ser b = c.mul(zt, qs);
        dQ = c.add(a, b);
    }
    G[0] = c.mul(psi2, dlpsi);
    G[1] = c.mul(Phb, dlPh);
    G[2] = dq;
    G[3] = dbr;
    G[4] = dbt;
    G[5] = dQ;
}

}  // namespace sl
}  // namespace mk

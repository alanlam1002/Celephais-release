// Mellin throat recursion, M2c (code round 463): the data rule and the conditions (mk_data.hpp).  Compiled with -ffp-contract=off.
#include "mk_data.hpp"

#include <mkl_cblas.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace mk {
namespace dt {

using ct::div;
using ct::mul;

static inline double cabs_(cd z) { return std::hypot(z.real(), z.imag()); }
static double absmax(const Vec& v) {
    double m = 0.0;
    for (auto& z : v) m = std::max(m, cabs_(z));
    return m;
}
// np.linalg.norm of a complex vector: x.real.dot(x.real) + x.imag.dot(x.imag), BLAS ddot on the strided views
static double npnorm(const Vec& v) {
    if (v.empty()) return 0.0;
    const double* p = reinterpret_cast<const double*>(v.data());
    int n = int(v.size());
    return std::sqrt(cblas_ddot(n, p, 2, p, 2) + cblas_ddot(n, p + 1, 2, p + 1, 2));
}
static inline cd rc(double r) { return cd(r, 0.0); }

Vec tg(const Vec& X, int NB, const Vec& e1, double R) {
    int N = int(e1.size());
    double a0 = R / N, a1 = R * R / N;
    Vec o(2 * size_t(NB));
    for (int k = 0; k < N; k++) {
        cd s0 = mul(rc(a0), e1[k]), s1 = mul(rc(a1), mul(e1[k], e1[k]));
        for (int j = 0; j < NB; j++) {
            cd t0 = mul(s0, X[size_t(k) * NB + j]), t1 = mul(s1, X[size_t(k) * NB + j]);
            o[j] = k == 0 ? t0 : o[j] + t0;
            o[NB + j] = k == 0 ? t1 : o[NB + j] + t1;
        }
    }
    return o;
}

// X0 + sum(c_[q] * Wq[q] for q in range(rk))
static Vec xc(const Vec& X0, const std::vector<Vec>& Wq, const Vec& c) {
    Vec S;
    for (size_t q = 0; q < c.size(); q++) {
        if (q == 0) {
            S.resize(X0.size());
            for (size_t i = 0; i < S.size(); i++) S[i] = mul(c[0], Wq[0][i]);
        } else {
            for (size_t i = 0; i < S.size(); i++) S[i] = S[i] + mul(c[q], Wq[q][i]);
        }
    }
    Vec X = X0;
    if (!S.empty())
        for (size_t i = 0; i < X.size(); i++) X[i] = X0[i] + S[i];
    return X;
}

// row-major helpers (m x n) -- small matrices; not bitwise with numpy's BLAS calls (the conditions come from the kernel anyway)
static Vec matvec(const Vec& A, int m, int n, const Vec& x) {
    Vec y(m, cd(0.0, 0.0));
    for (int i = 0; i < m; i++) {
        cd s(0.0, 0.0);
        for (int j = 0; j < n; j++) s += mul(A[size_t(i) * n + j], x[j]);
        y[i] = s;
    }
    return y;
}
static Vec hmatvec(const Vec& A, int m, int n, const Vec& x) {      // A^H x
    Vec y(n, cd(0.0, 0.0));
    for (int j = 0; j < n; j++) {
        cd s(0.0, 0.0);
        for (int i = 0; i < m; i++) s += mul(std::conj(A[size_t(i) * n + j]), x[i]);
        y[j] = s;
    }
    return y;
}
static Vec hmatmul(const Vec& A, int m, int n, const Vec& B, int p) {   // A^H B (n x p), A m x n, B m x p
    Vec C(size_t(n) * p, cd(0.0, 0.0));
    for (int a = 0; a < n; a++)
        for (int b = 0; b < p; b++) {
            cd s(0.0, 0.0);
            for (int i = 0; i < m; i++) s += mul(std::conj(A[size_t(i) * n + a]), B[size_t(i) * p + b]);
            C[size_t(a) * p + b] = s;
        }
    return C;
}
static Vec matmul(const Vec& A, int m, int n, const Vec& B, int p) {
    Vec C(size_t(m) * p, cd(0.0, 0.0));
    for (int a = 0; a < m; a++)
        for (int b = 0; b < p; b++) {
            cd s(0.0, 0.0);
            for (int i = 0; i < n; i++) s += mul(A[size_t(a) * n + i], B[size_t(i) * p + b]);
            C[size_t(a) * p + b] = s;
        }
    return C;
}

// jac(fun, c_): central differences, hc_ = 1e-3 * (1 + max|c_|); (m x rk) row-major
static Vec jac(const std::function<Vec(const Vec&)>& fun, const Vec& c, int& m) {
    int rk = int(c.size());
    double h = JAC_H * (1 + absmax(c));
    std::vector<Vec> cols(rk);
    for (int q = 0; q < rk; q++) {
        Vec cp = c, cm = c;
        cp[q] = cp[q] + rc(h);
        cm[q] = cm[q] - rc(h);
        Vec fp = fun(cp), fm = fun(cm);
        cols[q].resize(fp.size());
        for (size_t i = 0; i < fp.size(); i++) cols[q][i] = div(fp[i] - fm[i], rc(2 * h));
    }
    m = rk ? int(cols[0].size()) : 0;
    Vec J(size_t(m) * rk);
    for (int i = 0; i < m; i++)
        for (int q = 0; q < rk; q++) J[size_t(i) * rk + q] = cols[q][i];
    return J;
}

int targets(const TgtIn& in, Vec& tgtv, Vec& prevv, std::vector<double>& wt, Vec& tg0, std::vector<std::pair<int, double>>& hit) {
    int nn = in.nn, NB = nn;
    tgtv.assign(2 * NB, cd(0.0, 0.0));
    prevv.assign(2 * NB, cd(0.0, 0.0));
    for (int j = 0; j < nn; j++) {
        double t = in.g + in.kofs[j];
        for (size_t e = 0; e < in.cbb_col.size(); e++)
            if (in.cbb_col[e] == j && in.cbb_t[e] == t) { tgtv[j] = in.cbb_m0[e]; break; }
    }
    wt.assign(2 * NB, 1.0);
    if (in.notgt)
        for (int j = 0; j < nn; j++) wt[j] = wt[NB + j] = in.lc[j] > 0 ? 1.0 : 0.0;
    tg0 = in.notgt ? prevv : tgtv;
    hit.clear();
    for (auto& it : in.tgt)                                       // [(j, v) for k_, v_ in tgt_items.items() for j in range(nn) if label == k_]
        for (int j = 0; j < nn; j++)
            if (in.collab[j] + "@t" + std::to_string(int(in.g + in.kofs[j])) == it.first) hit.emplace_back(j, it.second);
    if (int(hit.size()) > in.rk) throw std::runtime_error("mk::dt::targets: more targets than residue directions (a genuine root outside the contour?)");
    if (!hit.empty()) {
        wt.assign(2 * NB, 0.0);
        tg0 = prevv;
        for (auto& h : hit) { wt[h.first] = 1.0; tg0[h.first] = h.second; }
    }
    return int(hit.size());
}

void data_rule(const DataIn& in, const Eval& ev, DataOut& o) {
    int nn = in.nn, NB = in.NB, rk = in.rk;
    o = DataOut();
    // unit residue: Wq = [w / np.linalg.norm(tg_(w)[:NB]) for w in Wq]; At = [tg_(w)].T
    o.Wq.resize(rk);
    o.At.assign(size_t(2) * NB * rk, cd(0.0, 0.0));
    for (int q = 0; q < rk; q++) {
        Vec t = tg(in.Wq[q], NB, in.e1, in.R);
        double nr = npnorm(Vec(t.begin(), t.begin() + NB));
        o.Wq[q].resize(in.Wq[q].size());
        for (size_t i = 0; i < o.Wq[q].size(); i++) o.Wq[q][i] = div(in.Wq[q][i], rc(nr));
        Vec tq = tg(o.Wq[q], NB, in.e1, in.R);
        for (int i = 0; i < 2 * NB; i++) o.At[size_t(i) * rk + q] = tq[i];
    }
    const std::vector<Vec>& Wq = o.Wq;
    Vec Wsq(2 * NB);
    for (int i = 0; i < 2 * NB; i++) Wsq[i] = rc(std::sqrt(in.wt[i]));
    auto Xc = [&](const Vec& c) { return xc(in.X0, Wq, c); };
    auto lsq_target = [&]() {                                     // lstsq(At[idx1], (tgtv - tg_(X0))[idx1])  (nb = 1: idx1 = everything)
        Vec t0 = tg(in.X0, NB, in.e1, in.R), b(2 * NB);
        for (int i = 0; i < 2 * NB; i++) b[i] = in.tgtv[i] - t0[i];
        return la::lstsq(o.At, 2 * NB, rk, b, 1);
    };
    Vec c = in.notgt ? Vec(rk, cd(0.0, 0.0)) : lsq_target();
    if (in.warm) c = in.cwarm;
    Vec F00 = ev.cond(in.X0);
    int nF = int(F00.size());
    // the tolerance: tied to the truncation (the gate rows' largest live residual) per unit of the grade's solution size
    if (in.cond_fit) {
        o.tau = 0.0; o.size = 1.0; o.condtol = std::numeric_limits<double>::infinity();
    } else {
        Vec Xm;
        if (in.notgt) {
            Xm = in.X0;
            o.tau = ev.gate(nullptr, in.g - 1);
        } else {
            Xm = Xc(lsq_target());
            o.tau = ev.gate(&Xm, in.g);
        }
        Vec tm = tg(Xm, NB, in.e1, in.R);
        o.size = std::max(absmax(Vec(tm.begin(), tm.begin() + nn)), SIZE_MIN);
        o.condtol = in.condtol_fixed != 0.0 ? in.condtol_fixed : std::max(10 * o.tau / o.size, CONDTOL_MIN);
    }
    auto dres = [&](const Vec& c_) {
        Vec tv = tg(Xc(c_), NB, in.e1, in.R), r(2 * NB);
        for (int i = 0; i < 2 * NB; i++) r[i] = mul(tv[i] - in.tg0[i], Wsq[i]);
        return r;
    };
    std::function<Vec(const Vec&)> condf = [&](const Vec& c_) { return ev.cond(Xc(c_)); };
    struct Sub { Vec U; int m = 0; Vec Vh; std::vector<double> S; int r1 = 0; Vec Jc; };
    auto csub = [&](const Vec& c_) {
        Sub s;
        s.m = nF;
        if (in.cond_fit) {
            s.Vh.assign(size_t(rk) * rk, cd(0.0, 0.0));
            for (int q = 0; q < rk; q++) s.Vh[size_t(q) * rk + q] = 1.0;
            s.Jc.assign(size_t(nF) * rk, cd(0.0, 0.0));
            return s;
        }
        int m;
        s.Jc = jac(condf, c_, m);
        s.m = m;
        la::svd(s.Jc, m, rk, s.S, &s.U, &s.Vh, true);
        for (double x : s.S) s.r1 += x > o.condtol;
        return s;
    };
    // the linearity test; the linear model when the conditions are linear in c
    if (in.cond_fit) {
        o.lin_err = 0.0;
    } else {
        int m;
        Vec Jl = jac(condf, c, m), Fl = condf(c);
        double sc = LIN_STEP * (1 + absmax(c));
        Vec dts(rk);
        for (int q = 0; q < rk; q++) dts[q] = div(mul(rc(sc), std::exp(cd(0.0, double(q)))), rc(std::sqrt(double(rk))));
        Vec cd_ = c;
        for (int q = 0; q < rk; q++) cd_[q] = c[q] + dts[q];
        Vec Fd = condf(cd_), Jd = matvec(Jl, m, rk, dts);
        double num = 0.0;
        for (int i = 0; i < m; i++) num = std::max(num, cabs_((Fd[i] - Fl[i]) - Jd[i]));
        o.lin_err = num / std::max({absmax(Jd), o.condtol * absmax(dts), 1e-300});
        if (o.lin_err < LIN_TOL && !in.cond_full) {
            Vec cl0 = c;
            condf = [Fl, Jl, cl0, m, rk](const Vec& c_) {
                Vec d(rk);
                for (int q = 0; q < rk; q++) d[q] = c_[q] - cl0[q];
                Vec y = matvec(Jl, m, rk, d);
                for (int i = 0; i < m; i++) y[i] = Fl[i] + y[i];
                return y;
            };
        }
    }
    double mu = MU0;
    int r1_0 = csub(c).r1;
    if (r1_0 == 0) {                                              // nothing constrained: the data rule alone, linear in c
        Vec d0 = dres(Vec(rk, cd(0.0, 0.0)));
        Vec Jd(size_t(2) * NB * rk);
        for (int q = 0; q < rk; q++) {
            Vec e(rk, cd(0.0, 0.0));
            e[q] = 1.0;
            Vec dq = dres(e);
            for (int i = 0; i < 2 * NB; i++) Jd[size_t(i) * rk + q] = dq[i] - d0[i];
        }
        Vec b(2 * NB);
        for (int i = 0; i < 2 * NB; i++) b[i] = -d0[i];
        c = la::lstsq(Jd, 2 * NB, rk, b, 1);
        Hist h;
        h.kind = 2;
        h.res = npnorm(dres(c));
        o.hist.push_back(h);
    }
    for (int it = 0; it < (r1_0 ? LM_ITERS : 0); it++) {         // Levenberg-Marquardt
        Sub s = csub(c);
        int r1 = s.r1, m = s.m;
        auto resid = [&](const Vec& c_) {
            if (r1 == 0) return dres(c_);
            Vec F = condf(c_), out;
            for (int a = 0; a < r1; a++) {
                cd z(0.0, 0.0);
                for (int i = 0; i < m; i++) z += mul(std::conj(s.U[size_t(i) * m + a]), F[i]);
                out.push_back(mul(rc(COND_W), z));
            }
            Vec d = dres(c_);
            out.insert(out.end(), d.begin(), d.end());
            return out;
        };
        Vec r0 = resid(c);
        int mr;
        Vec Jr = jac(resid, c, mr);
        Vec A = hmatmul(Jr, mr, rk, Jr, rk), gv = hmatvec(Jr, mr, rk, r0);
        bool acc = false;
        Vec dc;
        double nr0 = npnorm(r0);
        for (int inner = 0; inner < LM_INNER; inner++) {
            Vec M = A;
            for (int q = 0; q < rk; q++) M[size_t(q) * rk + q] = A[size_t(q) * rk + q] + mul(rc(mu), rc(std::max(A[size_t(q) * rk + q].real(), 1e-300)));
            dc = la::solve(M, rk, gv, 1);
            for (auto& z : dc) z = -z;
            Vec cn(rk);
            for (int q = 0; q < rk; q++) cn[q] = c[q] + dc[q];
            if (npnorm(resid(cn)) < nr0) {
                c = cn;
                mu = std::max(mu / 10, MU_MIN);
                acc = true;
                break;
            }
            mu *= 10;
        }
        Hist h;
        h.res = nr0; h.mu = mu; h.rank = r1;
        o.hist.push_back(h);
        if (!acc || absmax(dc) < STEP_STOP * (1 + absmax(c))) break;
    }
    for (int it = 0; it < POLISH_ITERS; it++) {                  // polish: the projected conditions inside the constrained subspace
        Sub s = csub(c);
        if (!s.r1) break;
        int r1 = s.r1, m = s.m;
        Vec Uc(size_t(m) * r1), Vc(size_t(rk) * r1);
        for (int i = 0; i < m; i++)
            for (int a = 0; a < r1; a++) Uc[size_t(i) * r1 + a] = s.U[size_t(i) * m + a];
        for (int q = 0; q < rk; q++)
            for (int a = 0; a < r1; a++) Vc[size_t(q) * r1 + a] = std::conj(s.Vh[size_t(a) * rk + q]);
        Vec A = matmul(hmatmul(Uc, m, r1, s.Jc, rk), r1, rk, Vc, r1);      // (Uc^H Jc) Vc, numpy's left-to-right
        Vec b = hmatvec(Uc, m, r1, condf(c));
        Vec da = la::lstsq(A, r1, r1, b, 1);
        for (auto& z : da) z = -z;
        Vec step = matvec(Vc, rk, r1, da);
        for (int q = 0; q < rk; q++) c[q] = c[q] + step[q];
        Hist h;
        h.kind = 1;
        h.polish = absmax(hmatvec(Uc, m, r1, condf(c)));
        o.hist.push_back(h);
        if (absmax(da) < STEP_STOP * (1 + absmax(c))) break;
    }
    // classification at the solution
    Sub s = csub(c);
    o.cond_sv = s.S;
    o.cond_rank = s.r1;
    o.c = c;
    o.F_start = F00;
    o.Xg = Xc(c);
    o.F_final = ev.cond(o.Xg);
    int gen = rk - s.r1, N = int(in.e1.size());
    double a0 = in.R / N;
    Vec res_g(nn);                                                // (R / N * e1[:, None] * Xg).sum(0)[:nn]
    for (int k = 0; k < N; k++) {
        cd s0 = mul(rc(a0), in.e1[k]);
        for (int j = 0; j < nn; j++) {
            cd t = mul(s0, o.Xg[size_t(k) * NB + j]);
            res_g[j] = k == 0 ? t : res_g[j] + t;
        }
    }
    for (int i = 0; i < gen; i++) {
        Vec z(rk);                                                // Z[:, i] = Vh[r_ + i].conj()
        for (int q = 0; q < rk; q++) z[q] = std::conj(s.Vh[size_t(s.r1 + i) * rk + q]);
        Vec d = xc(Vec(Wq[0].size(), cd(0.0, 0.0)), Wq, z);     // sum(Z[q, i] * Wq[q])  (0 + ...: the same values)
        Vec a(nn);
        for (int k = 0; k < N; k++) {
            cd s0 = mul(rc(a0), in.e1[k]);
            for (int j = 0; j < nn; j++) {
                cd t = mul(s0, d[size_t(k) * NB + j]);
                a[j] = k == 0 ? t : a[j] + t;
            }
        }
        Fitted f;
        std::vector<int> ord(nn);
        std::iota(ord.begin(), ord.end(), 0);
        std::stable_sort(ord.begin(), ord.end(), [&](int x, int y) { return cabs_(a[x]) > cabs_(a[y]); });
        for (int q = 0; q < 3 && q < nn; q++) f.cols[q] = ord[q];
        cd amp(0.0, 0.0);
        for (int q = 0; q < rk; q++) amp += mul(std::conj(z[q]), c[q]);
        f.amp = amp;
        Vec a0v;
        double l2 = 0.0;
        for (int j = 0; j < nn; j++) {
            if (in.lc[j] == 0) a0v.push_back(a[j]);
            if (in.lc[j] == 2) l2 = std::max(l2, cabs_(res_g[j]));
        }
        f.l0_share = npnorm(a0v) / std::max(npnorm(a), 1e-300);
        f.l2_content = l2;
        o.fitted.push_back(f);
    }
}

Vec data_fix(const Vec& X, const DataIn& in, const DataOut& d) {
    int NB = in.NB, rk = in.rk;
    Vec tv = tg(X, NB, in.e1, in.R), A(size_t(2) * NB * rk), b(2 * NB);
    for (int i = 0; i < 2 * NB; i++) {
        cd w = rc(std::sqrt(in.wt[i]));
        for (int q = 0; q < rk; q++) A[size_t(i) * rk + q] = mul(w, d.At[size_t(i) * rk + q]);
        b[i] = -mul(w, tv[i] - in.tg0[i]);
    }
    Vec dl = la::lstsq(A, 2 * NB, rk, b, 1);
    Vec S = xc(Vec(X.size(), cd(0.0, 0.0)), d.Wq, dl), out(X.size());
    for (size_t i = 0; i < X.size(); i++) out[i] = X[i] + S[i];
    return out;
}

void offl_start(const Vec& X0, const std::vector<Vec>& Wq_raw, int nn, int NB, const Vec& e1, double R, bool labelled, double datum,
                bool cond_fit, const Eval& ev, std::vector<Vec>& Wq, Vec& Xg, bool& has_fixed, cd& amp_fixed) {
    int rk = int(Wq_raw.size());
    Wq.assign(rk, Vec());
    for (int q = 0; q < rk; q++) {
        Vec t = tg(Wq_raw[q], NB, e1, R);
        double nr = npnorm(Vec(t.begin(), t.begin() + NB));
        Vec w(Wq_raw[q].size());
        for (size_t i = 0; i < w.size(); i++) w[i] = div(Wq_raw[q][i], rc(nr));
        Vec t2 = tg(w, NB, e1, R);
        int im = 0;
        for (int j = 1; j < nn; j++)
            if (cabs_(t2[j]) > cabs_(t2[im])) im = j;
        cd ph = std::exp(mul(cd(0.0, -1.0), rc(std::atan2(t2[im].imag(), t2[im].real()))));
        for (auto& z : w) z = mul(z, ph);
        Wq[q] = std::move(w);
    }
    has_fixed = false;
    Xg = X0;
    if (!rk) return;
    double cval = labelled ? datum : 0.0;
    for (size_t i = 0; i < Xg.size(); i++) Xg[i] = X0[i] + mul(rc(cval), Wq[0][i]);
    if (labelled && !cond_fit) {
        Vec F0 = ev.cond(X0), X1(X0.size());
        for (size_t i = 0; i < X0.size(); i++) X1[i] = X0[i] + Wq[0][i];
        Vec F1 = ev.cond(X1);
        for (size_t i = 0; i < F1.size(); i++) F1[i] = F1[i] - F0[i];
        double tau_c = 10 * std::max(absmax(F0) * 1e-12, 1e-14);
        if (absmax(F1) > std::max(tau_c, 1e-10 * absmax(F0))) {
            cd n01(0.0, 0.0), n11(0.0, 0.0);
            for (size_t i = 0; i < F1.size(); i++) { n01 += mul(std::conj(F1[i]), F0[i]); n11 += mul(std::conj(F1[i]), F1[i]); }
            amp_fixed = -div(n01, n11);
            has_fixed = true;
            for (size_t i = 0; i < Xg.size(); i++) Xg[i] = X0[i] + mul(amp_fixed, Wq[0][i]);
        }
    }
}

}  // namespace dt
}  // namespace mk

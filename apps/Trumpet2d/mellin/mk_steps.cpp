// Mellin throat recursion, M2d (code round 463): the nonlinear steps (mk_steps.hpp).  Compiled with -ffp-contract=off.
#include <cstdlib>
#include <string>
#include "mk_steps.hpp"

#include <mkl_cblas.h>

#include <algorithm>
#include <cmath>

namespace mk {
namespace st {

using ct::mul;

static inline double cabs_(cd z) { return std::hypot(z.real(), z.imag()); }
static inline cd rc(double r) { return cd(r, 0.0); }

// numpy matmul (n x n C-contiguous) @ (n), as mk_contour.cpp
static Vec gemv(const Vec& A, const cd* x, int n) {
    Vec y(n);
    const cd one(1.0, 0.0), zero(0.0, 0.0);
    cblas_zgemv(CblasColMajor, CblasTrans, n, n, &one, A.data(), n, x, 1, &zero, y.data(), 1);
    return y;
}
// tg_dc(X) = (R / N * e1[:, None] * X).sum(0)
Vec tg_dc(const Vec& X, int NB, const Vec& e1, double R) {
    int N = int(e1.size());
    Vec o(NB);
    for (int k = 0; k < N; k++) {
        cd s = mul(rc(R / N), e1[k]);
        for (int j = 0; j < NB; j++) {
            cd t = mul(s, X[size_t(k) * NB + j]);
            o[j] = k == 0 ? t : o[j] + t;
        }
    }
    return o;
}
// [-Minv[k] @ v for k, v in enumerate(nodes_of(rows, g))]
Vec correction(const Vec& rows, const std::vector<Vec>& Minv, int K, int NB, const Vec& e1, double R) {
    int N = int(e1.size());
    Vec nd = ct::nodes_of(rows, K, NB, e1, R), out(size_t(N) * NB);
    for (int k = 0; k < N; k++) {
        Vec nM(Minv[k].size());
        for (size_t i = 0; i < nM.size(); i++) nM[i] = -Minv[k][i];
        Vec y = gemv(nM, &nd[size_t(k) * NB], NB);
        std::copy(y.begin(), y.end(), out.begin() + long(k) * NB);
    }
    return out;
}
// d - sum(a[q] * Wq[q]) with a = lstsq(A_hold, tg_dc(d)[:nh]): the components along the residue directions removed
Vec tg_root(const Vec& X, int NB, const Vec& e1, double R, int Kp) {
    int N = int(e1.size());
    Vec o(NB, cd(0.0, 0.0));
    for (int k = 0; k < N; k++) {
        cd s = rc(R / N) * std::pow(e1[k], Kp + 1);
        for (int j = 0; j < NB; j++) o[j] += s * X[size_t(k) * NB + j];
    }
    return o;
}
int hold_root_K(const std::vector<Vec>& Wq, int NB, const Vec& e1, double R, int K) {
    static const bool on = std::getenv("N2A_HOLD_ROOT") && std::string(std::getenv("N2A_HOLD_ROOT")) == "1";
    if (!on || Wq.empty()) return 0;
    int N = int(e1.size());
    for (auto& W : Wq) {
        cd n01(0.0, 0.0), n00(0.0, 0.0);
        Vec t0(NB, cd(0.0, 0.0)), t1(NB, cd(0.0, 0.0));
        for (int k = 0; k < N; k++) {
            cd s0 = rc(R / N) * e1[k], s1 = rc(R * R / N) * e1[k] * e1[k];
            for (int j = 0; j < NB; j++) { t0[j] += s0 * W[size_t(k) * NB + j]; t1[j] += s1 * W[size_t(k) * NB + j]; }
        }
        for (int j = 0; j < NB; j++) { n01 += std::conj(t0[j]) * t1[j]; n00 += std::conj(t0[j]) * t0[j]; }
        if (std::abs(n00) == 0.0 || std::abs(n01 / n00) < 1e-2 * R) return 0;
    }
    return K;
}

void hold(Vec& d, const std::vector<Vec>& Wq, int nh, int NB, const Vec& e1, double R, int Kp) {
    int rk = int(Wq.size());
    if (!rk) return;
    auto fn = [&](const Vec& X) { return Kp ? tg_root(X, NB, e1, R, Kp) : tg_dc(X, NB, e1, R); };
    Vec A(size_t(nh) * rk), t = fn(d);
    for (int q = 0; q < rk; q++) {
        Vec tq = fn(Wq[q]);
        for (int i = 0; i < nh; i++) A[size_t(i) * rk + q] = tq[i];
    }
    Vec a = la::lstsq(A, nh, rk, Vec(t.begin(), t.begin() + nh), 1);
    Vec S(d.size());
    for (int q = 0; q < rk; q++)
        for (size_t i = 0; i < d.size(); i++) S[i] = q == 0 ? mul(a[0], Wq[0][i]) : S[i] + mul(a[q], Wq[q][i]);
    for (size_t i = 0; i < d.size(); i++) d[i] = d[i] - S[i];
}
static double absmax(const Vec& v) {
    double m = 0.0;
    for (auto& z : v) m = std::max(m, cabs_(z));
    return m;
}

void defect_int(const Vec& X0, const std::vector<Vec>& Minv, const std::vector<Vec>& Wq, const std::vector<char>& act, double Smax, int K,
                int NB, const Vec& e1, double R, int ndc, bool robust, const std::function<Vec(const Vec&)>& dfix, const Rows& rows, DefectOut& o,
                const std::function<void(int, const Vec&, const Vec&, const Vec&, const Vec&)>* dump) {
    o = DefectOut();
    Vec Xg = X0;
    Vec res0 = tg_dc(Xg, NB, e1, R);
    auto masked = [&](const Vec& X) {                                // galb(...) * act[None, :]  (nb = 1: kept_only removes nothing)
        Vec r = rows.gal(X);
        for (int m = 0; m < K; m++)
            for (int j = 0; j < NB; j++) {
                cd& z = r[size_t(m) * NB + j];
                z = mul(z, rc(act[j] ? 1.0 : 0.0));
            }
        return r;
    };
    double stop = DC_STOP * std::max(Smax, 1e-300);
    for (int itd = 0; itd < ndc; itd++) {
        Vec rbd = masked(Xg);
        o.hist.push_back(absmax(rbd));
        if (o.hist.back() <= stop) break;
        if (robust && o.hist.size() > 1 && o.hist.back() > HALVE * o.hist[o.hist.size() - 2]) break;
        Vec dX = correction(rbd, Minv, K, NB, e1, R);
        Vec dXr;
        if (dump) dXr = dX;
        hold(dX, Wq, NB, NB, e1, R);
        for (size_t i = 0; i < Xg.size(); i++) Xg[i] = Xg[i] + dX[i];
        if (dfix) Xg = dfix(Xg);
        if (dump) (*dump)(itd, rbd, dXr, dX, Xg);
    }
    Vec r1 = tg_dc(Xg, NB, e1, R);
    for (int j = 0; j < NB; j++) o.drift = std::max(o.drift, cabs_(r1[j] - res0[j]));
    o.rows_after = ndc ? absmax(masked(Xg)) : 0.0;
    o.X = Xg;
}

void defect_off(const Vec& X0, const std::vector<Vec>& Minv, const std::vector<Vec>& Wq, int K, int NB, const Vec& e1, double R, int ndef,
                bool robust, const Rows& rows, DefectOut& o) {
    o = DefectOut();
    Vec Xg = X0;
    for (int it = 0; it <= ndef; it++) {
        Vec Rr = rows.gal(Xg);
        o.hist.push_back(absmax(Rr));
        if (o.hist.back() < OFF_STOP || it == ndef) break;
        if (robust && o.hist.size() > 1 && o.hist.back() > HALVE * o.hist[o.hist.size() - 2]) break;
        Vec dX = correction(Rr, Minv, K, NB, e1, R);
        hold(dX, Wq, NB, NB, e1, R, hold_root_K(Wq, NB, e1, R, K));   // research round 1098: at the root's own exponent (N2A_HOLD_ROOT=1)
        for (size_t i = 0; i < Xg.size(); i++) Xg[i] = Xg[i] + dX[i];
    }
    o.conditions = rows.stored(Xg);
    o.X = Xg;
}

Vec newton_sources(const Vec& Xk, const Vec& Ms, int K, int NB, const Vec& e1, double R, const Rows& rows) {
    int N = int(e1.size());
    size_t nm = size_t(NB) * NB;
    Vec nd = ct::nodes_of(rows.gal(Xk), K, NB, e1, R), out(size_t(N) * NB);
    for (int k = 0; k < N; k++) {
        cd zz = mul(rc(R), e1[k]), z2 = mul(zz, zz);
        Vec M(nm);
        for (size_t i = 0; i < nm; i++) M[i] = (Ms[i] + mul(zz, Ms[nm + i])) + mul(z2, Ms[2 * nm + i]);
        Vec y = gemv(M, &Xk[size_t(k) * NB], NB);
        for (int j = 0; j < NB; j++) out[size_t(k) * NB + j] = nd[size_t(k) * NB + j] - y[j];
    }
    return out;
}

bool newton_continue(const std::vector<double>& h, double Smax, int newt, int newton_max, bool isint, int ndc) {
    bool stagn = h.size() > 1 && h.back() > HALVE * h[h.size() - 2];
    return isint && ndc && h.back() > NEWTON_TOL * std::max(Smax, 1e-300) && newt < newton_max - 1 && !stagn;
}

}  // namespace st
}  // namespace mk

// Mellin throat recursion (code round 463; research rounds 1074 / 1089, the forward-mode replay, STAGE 1(b)): ONE STEP at a centre on duals --
// mk_steps' defect correction (integer and off-integer centres) carried with its tangents along ND directions, iteration by iteration, as
// executed.  The DECISIONS are the value's: the number of corrections (the stops are tested on the value), the active columns, the residue
// rank, the hold's least-squares rank.  So the tangent is the derivative of the step the forward took -- not of its fixed point (the
// existing tangents, mk_tangent.hpp tan_step, linearise the fixed point and iterate to a tolerance: the two differ where the forward stopped
// short of its fixed point).
//   correction   d = -Minv nodes(rows)                     dd = -dMinv nodes(rows) - Minv nodes(drows)
//   hold         d - sum_q a_q Wq_q, a = lstsq(A, t)       (A_iq = tg_dc(Wq_q)_i, t = tg_dc(d)): mk_ad lstsq_t, the rank frozen
//   data fix     --cond-fit's dfix and its tangent (a callback pair)
// The rows reach this code as a callback returning the value and its directional derivatives (the kernel's linear response, or a test's).
// Unit-tested against central differences of mk_steps' own defect_int / defect_off with every input moved (mk_steps_dual_test.cpp).
#pragma once
#include "mk_ad.hpp"
#include "mk_contour.hpp"
#include "mk_steps.hpp"

namespace mk {
namespace sd {

using cd = std::complex<double>;
using Vec = std::vector<cd>;
using Tv = std::vector<Vec>;                    // one tangent per direction (an empty Vec = zero)

// rows(X, dX) -> value (K x NB) and its tangents (ND x (K x NB)): the total derivative along each direction (X's tangent dX[k] AND the
// rows' own dependence on the data)
using RowsT = std::function<Vec(const Vec& X, const Tv& dX, Tv& dR)>;
// the data fix and its tangent: X -> fix(X), dX -> d fix
using FixT = std::function<Vec(const Vec& X, const Tv& dX, Tv& dOut)>;

inline Vec addv(const Vec& a, const Vec& b) { Vec o(a.size()); for (size_t i = 0; i < a.size(); i++) o[i] = a[i] + b[i]; return o; }
inline Vec zerosv(size_t n) { return Vec(n, cd(0.0)); }

// the correction and its tangent: per node k, -Minv[k] nd_k (and -dMinv[k] nd_k - Minv[k] dnd_k)
inline Vec correction_t(const Vec& rows, const Tv& drows, const std::vector<Vec>& Minv, const std::vector<ad::Tans>& dMinv, int K, int NB,
                        const Vec& e1, double R, int ND, Tv& dd) {
    int N = int(e1.size());
    Vec nd = ct::nodes_of(rows, K, NB, e1, R), out(size_t(N) * NB);
    std::vector<Vec> dnd(ND);
    for (int q = 0; q < ND; q++) dnd[q] = drows[q].empty() ? zerosv(nd.size()) : ct::nodes_of(drows[q], K, NB, e1, R);
    dd.assign(ND, zerosv(out.size()));
    for (int k = 0; k < N; k++) {
        const cd* x = &nd[size_t(k) * NB];
        for (int i = 0; i < NB; i++) {
            cd s = 0.0;
            for (int j = 0; j < NB; j++) s += Minv[k][size_t(i) * NB + j] * x[j];
            out[size_t(k) * NB + i] = -s;
        }
        for (int q = 0; q < ND; q++) {
            const cd* dx = &dnd[q][size_t(k) * NB];
            const Vec* dM = (size_t(k) < dMinv.size() && size_t(q) < dMinv[k].size() && !dMinv[k][q].empty()) ? &dMinv[k][q] : nullptr;
            for (int i = 0; i < NB; i++) {
                cd s = 0.0;
                for (int j = 0; j < NB; j++) {
                    s += Minv[k][size_t(i) * NB + j] * dx[j];
                    if (dM) s += (*dM)[size_t(i) * NB + j] * x[j];
                }
                dd[q][size_t(k) * NB + i] = -s;
            }
        }
    }
    return out;
}

// the hold and its tangent (in place): d -= sum a_q Wq_q, a = lstsq(A, tg_dc(d)[:nh]); dWq[q][r]: the tangent of Wq_r along direction q
inline void hold_t(Vec& d, Tv& dd, const std::vector<Vec>& Wq, const std::vector<Tv>& dWq, int nh, int NB, const Vec& e1, double R, int ND) {
    int rk = int(Wq.size());
    if (!rk) return;
    Vec A(size_t(nh) * rk), t = st::tg_dc(d, NB, e1, R);
    for (int r = 0; r < rk; r++) {
        Vec tr = st::tg_dc(Wq[r], NB, e1, R);
        for (int i = 0; i < nh; i++) A[size_t(i) * rk + r] = tr[i];
    }
    ad::Tans dA(ND), dB(ND);
    for (int q = 0; q < ND; q++) {
        dA[q].assign(size_t(nh) * rk, 0.0);
        for (int r = 0; r < rk; r++) {
            if (size_t(q) >= dWq.size() || dWq[q].empty() || dWq[q][r].empty()) continue;
            Vec tr = st::tg_dc(dWq[q][r], NB, e1, R);
            for (int i = 0; i < nh; i++) dA[q][size_t(i) * rk + r] = tr[i];
        }
        Vec td = st::tg_dc(dd[q], NB, e1, R);
        dB[q] = Vec(td.begin(), td.begin() + nh);
    }
    ad::Tans da;
    Vec a = ad::lstsq_t(A, nh, rk, Vec(t.begin(), t.begin() + nh), 1, dA, dB, da);
    for (int q = 0; q < ND; q++)
        for (int r = 0; r < rk; r++) {
            const Vec* dW = (size_t(q) < dWq.size() && !dWq[q].empty() && !dWq[q][r].empty()) ? &dWq[q][r] : nullptr;
            for (size_t i = 0; i < d.size(); i++) {
                cd v = da[q][r] * Wq[r][i];
                if (dW) v += a[r] * (*dW)[i];
                dd[q][i] -= v;
            }
        }
    for (int r = 0; r < rk; r++)
        for (size_t i = 0; i < d.size(); i++) d[i] -= a[r] * Wq[r][i];
}

inline double absmax(const Vec& v) { double m = 0.0; for (auto& z : v) m = std::max(m, std::abs(z)); return m; }

struct DefectT {
    Vec X;
    Tv dX;
    int corrections = 0;                        // the decision: how many corrections the value took
    std::vector<double> hist;
};

// integer centre (st::defect_int on duals).  dX0: X0's tangents; dMinv[k]: Minv[k]'s tangents (ND); dWq[q][r]
inline void defect_int_t(const Vec& X0, const Tv& dX0, const std::vector<Vec>& Minv, const std::vector<ad::Tans>& dMinv,
                         const std::vector<Vec>& Wq, const std::vector<Tv>& dWq, const std::vector<char>& act, double Smax, int K, int NB,
                         const Vec& e1, double R, int ndc, bool robust, const FixT* dfix, const RowsT& rows, int ND, DefectT& o) {
    o = DefectT();
    Vec X = X0;
    Tv dX(ND);
    for (int q = 0; q < ND; q++) dX[q] = (size_t(q) < dX0.size() && !dX0[q].empty()) ? dX0[q] : zerosv(X0.size());
    auto masked = [&](const Vec& Xv, const Tv& dXv, Tv& dR) {
        Vec r = rows(Xv, dXv, dR);
        for (int m = 0; m < K; m++)
            for (int j = 0; j < NB; j++)
                if (!act[j]) {
                    r[size_t(m) * NB + j] = 0.0;
                    for (auto& t : dR) if (!t.empty()) t[size_t(m) * NB + j] = 0.0;
                }
        return r;
    };
    double stop = st::DC_STOP * std::max(Smax, 1e-300);
    for (int itd = 0; itd < ndc; itd++) {
        Tv dR;
        Vec rbd = masked(X, dX, dR);
        o.hist.push_back(absmax(rbd));
        if (o.hist.back() <= stop) break;
        if (robust && o.hist.size() > 1 && o.hist.back() > st::HALVE * o.hist[o.hist.size() - 2]) break;
        Tv dd;
        Vec d = correction_t(rbd, dR, Minv, dMinv, K, NB, e1, R, ND, dd);
        hold_t(d, dd, Wq, dWq, NB, NB, e1, R, ND);
        X = addv(X, d);
        for (int q = 0; q < ND; q++) dX[q] = addv(dX[q], dd[q]);
        if (dfix && *dfix) {
            Tv dF;
            X = (*dfix)(X, dX, dF);
            dX = dF;
        }
        o.corrections++;
    }
    o.X = X;
    o.dX = dX;
}

// off-integer centre (st::defect_off on duals; the conditions are read by the caller from the final X)
inline void defect_off_t(const Vec& X0, const Tv& dX0, const std::vector<Vec>& Minv, const std::vector<ad::Tans>& dMinv,
                         const std::vector<Vec>& Wq, const std::vector<Tv>& dWq, int K, int NB, const Vec& e1, double R, int ndef, bool robust,
                         const RowsT& rows, int ND, DefectT& o) {
    o = DefectT();
    Vec X = X0;
    Tv dX(ND);
    for (int q = 0; q < ND; q++) dX[q] = (size_t(q) < dX0.size() && !dX0[q].empty()) ? dX0[q] : zerosv(X0.size());
    for (int it = 0; it <= ndef; it++) {
        Tv dR;
        Vec Rr = rows(X, dX, dR);
        o.hist.push_back(absmax(Rr));
        if (o.hist.back() < st::OFF_STOP || it == ndef) break;
        if (robust && o.hist.size() > 1 && o.hist.back() > st::HALVE * o.hist[o.hist.size() - 2]) break;
        Tv dd;
        Vec d = correction_t(Rr, dR, Minv, dMinv, K, NB, e1, R, ND, dd);
        hold_t(d, dd, Wq, dWq, NB, NB, e1, R, ND);
        X = addv(X, d);
        for (int q = 0; q < ND; q++) dX[q] = addv(dX[q], dd[q]);
        o.corrections++;
    }
    o.X = X;
    o.dX = dX;
}

}  // namespace sd
}  // namespace mk

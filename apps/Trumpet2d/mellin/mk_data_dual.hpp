// Mellin throat recursion (code round 463; research round 1092, the forward-mode replay, STAGE 1(c), part 2): the DATA RULE on duals, in
// the production mode (--cond-fit: nothing imposed, the data rule alone, linear) -- dt::data_rule's amplitudes and start, and dt::data_fix,
// with their tangents along ND directions.  The value is dt's own (called by the caller); here only the tangents.
//   unit residue      Wn_q = W_q / nu_q, nu_q = ||tg(W_q)[:NB]||           dWn = dW / nu - W dnu / nu^2, dnu = Re(t^H dt) / nu
//   the amplitudes    c = lstsq(Jd, -d0), Jd = Ws tg(Wn) (columns), d0 = Ws (tg(X0) - tg0)   (mk_ad lstsq_t, the rank frozen)
//   the start         Xg = X0 + sum_q c_q Wn_q                              dXg = dX0 + sum (dc_q Wn_q + c_q dWn_q)
//   the data fix      X -> X + sum dl_q Wn_q, dl = lstsq(Ws At, -Ws (tg(X) - tg0))   (its tangent through X, Wn and tg0)
// tg0 (the targets) moves with a data seed: dtg0 given per direction (empty = zero).  The weights are constant.
// Unit-tested against central differences of dt::data_rule / dt::data_fix (mk_data_dual_test.cpp).
#pragma once
#include "mk_ad.hpp"
#include "mk_data.hpp"

namespace mk {
namespace ddl {

using cd = std::complex<double>;
using Vec = std::vector<cd>;

struct DataT {
    std::vector<std::vector<Vec>> dWn;    // [q][r]
    std::vector<Vec> dc, dXg;             // [q]
};

inline Vec lin(const Vec& a, cd s, const Vec& b) { Vec o(a.size()); for (size_t i = 0; i < a.size(); i++) o[i] = a[i] + s * b[i]; return o; }

// in / d: dt::data_rule's input and output (the value); dX0[q], dW[q][r] (the contour's, unnormalised), dtg0[q]
inline void data_rule_t(const dt::DataIn& in, const dt::DataOut& d, const std::vector<Vec>& dX0, const std::vector<std::vector<Vec>>& dW,
                        const std::vector<Vec>& dtg0, int ND, DataT& t) {
    int NB = in.NB, rk = in.rk;
    t = DataT();
    t.dWn.assign(ND, std::vector<Vec>(rk));
    t.dc.assign(ND, Vec(rk, 0.0));
    t.dXg.assign(ND, Vec());
    std::vector<double> ws(2 * NB);
    for (int i = 0; i < 2 * NB; i++) ws[i] = std::sqrt(in.wt[i]);
    // the value's pieces: Jd = Ws tg(Wn); d0 = Ws (tg(X0) - tg0)
    Vec Jd(size_t(2) * NB * rk), d0(2 * NB);
    for (int r = 0; r < rk; r++) {
        Vec tq = dt::tg(d.Wq[r], NB, in.e1, in.R);
        for (int i = 0; i < 2 * NB; i++) Jd[size_t(i) * rk + r] = ws[i] * tq[i];
    }
    Vec t0 = dt::tg(in.X0, NB, in.e1, in.R);
    for (int i = 0; i < 2 * NB; i++) d0[i] = ws[i] * (t0[i] - in.tg0[i]);
    ad::Tans dJ(ND), dB(ND);
    for (int q = 0; q < ND; q++) {
        dJ[q].assign(size_t(2) * NB * rk, 0.0);
        for (int r = 0; r < rk; r++) {
            Vec tr = dt::tg(in.Wq[r], NB, in.e1, in.R);           // unnormalised
            double nu = 0.0;
            for (int j = 0; j < NB; j++) nu += std::norm(tr[j]);
            nu = std::sqrt(nu);
            Vec dWr = (q < int(dW.size()) && r < int(dW[q].size()) && !dW[q][r].empty()) ? dW[q][r] : Vec(in.Wq[r].size(), 0.0);
            Vec dtr = dt::tg(dWr, NB, in.e1, in.R);
            double dnu = 0.0;
            for (int j = 0; j < NB; j++) dnu += (std::conj(tr[j]) * dtr[j]).real();
            dnu /= nu;
            Vec dWn(dWr.size());
            for (size_t i = 0; i < dWn.size(); i++) dWn[i] = dWr[i] / nu - in.Wq[r][i] * (dnu / (nu * nu));
            t.dWn[q][r] = dWn;
            Vec dtn = dt::tg(dWn, NB, in.e1, in.R);
            for (int i = 0; i < 2 * NB; i++) dJ[q][size_t(i) * rk + r] = ws[i] * dtn[i];
        }
        Vec dX0q = (q < int(dX0.size()) && !dX0[q].empty()) ? dX0[q] : Vec(in.X0.size(), 0.0);
        Vec dt0 = dt::tg(dX0q, NB, in.e1, in.R);
        dB[q].assign(2 * NB, 0.0);
        for (int i = 0; i < 2 * NB; i++) {
            cd dg = (q < int(dtg0.size()) && !dtg0[q].empty()) ? dtg0[q][i] : cd(0.0);
            dB[q][i] = -ws[i] * (dt0[i] - dg);
        }
    }
    Vec mb(2 * NB);
    for (int i = 0; i < 2 * NB; i++) mb[i] = -d0[i];
    ad::Tans dcs;
    Vec c = ad::lstsq_t(Jd, 2 * NB, rk, mb, 1, dJ, dB, dcs);
    for (int q = 0; q < ND; q++) {
        t.dc[q] = dcs[q];
        Vec dX0q = (q < int(dX0.size()) && !dX0[q].empty()) ? dX0[q] : Vec(in.X0.size(), 0.0);
        Vec X = dX0q;
        for (int r = 0; r < rk; r++) {
            X = lin(X, dcs[q][r], d.Wq[r]);
            X = lin(X, c[r], t.dWn[q][r]);
        }
        t.dXg[q] = X;
    }
}

// the data fix's tangent: X, dX[q] -> fix(X) (dt::data_fix, the value) and its tangents, given dWn[q][r] and dtg0[q]
inline Vec data_fix_t(const Vec& X, const std::vector<Vec>& dX, const dt::DataIn& in, const dt::DataOut& d,
                      const std::vector<std::vector<Vec>>& dWn, const std::vector<Vec>& dtg0, int ND, std::vector<Vec>& dOut) {
    int NB = in.NB, rk = in.rk;
    std::vector<double> ws(2 * NB);
    for (int i = 0; i < 2 * NB; i++) ws[i] = std::sqrt(in.wt[i]);
    Vec tv = dt::tg(X, NB, in.e1, in.R), A(size_t(2) * NB * rk), b(2 * NB);
    for (int i = 0; i < 2 * NB; i++) {
        for (int r = 0; r < rk; r++) A[size_t(i) * rk + r] = ws[i] * d.At[size_t(i) * rk + r];
        b[i] = -ws[i] * (tv[i] - in.tg0[i]);
    }
    ad::Tans dA(ND), dB(ND);
    for (int q = 0; q < ND; q++) {
        dA[q].assign(size_t(2) * NB * rk, 0.0);
        for (int r = 0; r < rk; r++) {
            Vec dtn = dt::tg(dWn[q][r], NB, in.e1, in.R);
            for (int i = 0; i < 2 * NB; i++) dA[q][size_t(i) * rk + r] = ws[i] * dtn[i];
        }
        Vec dtv = dt::tg(dX[q], NB, in.e1, in.R);
        dB[q].assign(2 * NB, 0.0);
        for (int i = 0; i < 2 * NB; i++) {
            cd dg = (q < int(dtg0.size()) && !dtg0[q].empty()) ? dtg0[q][i] : cd(0.0);
            dB[q][i] = -ws[i] * (dtv[i] - dg);
        }
    }
    ad::Tans ddl;
    Vec dl = ad::lstsq_t(A, 2 * NB, rk, b, 1, dA, dB, ddl);
    Vec out = X;
    for (int r = 0; r < rk; r++) out = lin(out, dl[r], d.Wq[r]);
    dOut.assign(ND, Vec());
    for (int q = 0; q < ND; q++) {
        Vec o = dX[q];
        for (int r = 0; r < rk; r++) {
            o = lin(o, ddl[q][r], d.Wq[r]);
            o = lin(o, dl[r], dWn[q][r]);
        }
        dOut[q] = o;
    }
    return out;
}

// research round 1093 (the replay's stage 2, part 1): the OFF-INTEGER START (dt::offl_start, --cond-fit: nothing fixed by the conditions) on
// duals.  Each direction normalised (nu = ||tg(W)[:NB]||) and phased (ph = exp(-i arg t_im), im = the largest |tg(w)_j|, j < nn: a decision,
// the value's), the start Xg = X0 + datum Wq_0:
//   dw = dW / nu - W dnu / nu^2;  d arg t = Im(dt / t);  dph = -i ph d arg t;  dWq = dw ph + w dph;  dXg = dX0 + ddatum Wq_0 + datum dWq_0
// Wq / Xg (the value) from dt::offl_start (called by the caller); here the tangents.  Unit-tested against central differences of it.
struct OfflT {
    std::vector<std::vector<Vec>> dWq;    // [q][r]
    std::vector<Vec> dXg;                 // [q]
};
inline void offl_start_t(const Vec& X0, const std::vector<Vec>& Wq_raw, int nn, int NB, const Vec& e1, double R, bool labelled, double datum,
                         const std::vector<Vec>& dX0, const std::vector<std::vector<Vec>>& dW, const std::vector<double>& ddatum, int ND,
                         const std::vector<Vec>& Wq, OfflT& t) {
    const int rk = int(Wq_raw.size());
    t = OfflT();
    t.dWq.assign(ND, std::vector<Vec>(rk));
    t.dXg.assign(ND, Vec());
    for (int q = 0; q < ND; q++) {
        for (int r = 0; r < rk; r++) {
            Vec tr = dt::tg(Wq_raw[r], NB, e1, R);
            double nu = 0.0;
            for (int j = 0; j < NB; j++) nu += std::norm(tr[j]);
            nu = std::sqrt(nu);
            Vec dWr = (q < int(dW.size()) && r < int(dW[q].size()) && !dW[q][r].empty()) ? dW[q][r] : Vec(Wq_raw[r].size(), 0.0);
            Vec dtr = dt::tg(dWr, NB, e1, R);
            double dnu = 0.0;
            for (int j = 0; j < NB; j++) dnu += (std::conj(tr[j]) * dtr[j]).real();
            dnu /= nu;
            Vec w(Wq_raw[r].size()), dw(Wq_raw[r].size());
            for (size_t i = 0; i < w.size(); i++) { w[i] = Wq_raw[r][i] / nu; dw[i] = dWr[i] / nu - Wq_raw[r][i] * (dnu / (nu * nu)); }
            Vec t2 = dt::tg(w, NB, e1, R), dt2 = dt::tg(dw, NB, e1, R);
            int im = 0;
            for (int j = 1; j < nn; j++)
                if (std::abs(t2[j]) > std::abs(t2[im])) im = j;
            cd ph = std::exp(cd(0.0, -1.0) * std::atan2(t2[im].imag(), t2[im].real()));
            double darg = (dt2[im] / t2[im]).imag();
            cd dph = cd(0.0, -1.0) * ph * darg;
            Vec o(w.size());
            for (size_t i = 0; i < w.size(); i++) o[i] = dw[i] * ph + w[i] * dph;
            t.dWq[q][r] = std::move(o);
        }
        Vec X = (q < int(dX0.size()) && !dX0[q].empty()) ? dX0[q] : Vec(X0.size(), 0.0);
        if (rk) {
            double cval = labelled ? datum : 0.0, dc = (labelled && q < int(ddatum.size())) ? ddatum[q] : 0.0;
            for (size_t i = 0; i < X.size(); i++) X[i] += dc * Wq[0][i] + cval * t.dWq[q][0][i];
        }
        t.dXg[q] = std::move(X);
    }
}

}  // namespace ddl
}  // namespace mk

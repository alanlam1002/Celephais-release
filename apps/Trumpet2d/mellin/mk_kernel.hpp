// Mellin throat kernel (code round 463 M1, research round 1013): the row assembly and its analytic linear response (the C ABI: mk_kernel.cpp).
//
//  * the TAPE (spinning_trumpet/scripts/m1_tape.py): the throat equations' assembly as executed by the Python engine (finiteJ_system.TH2 with
//    the C3 edits, run with MS), recorded in execution order -- every op replayed here, none pruned (products register off-lattice centres);
//  * fields_from: stage B's columns -> the six fields (n2a_nl.fields_from: psi2 = r^-1 exp(2U), Phibar = K / r x P_shape x ..., beta^r = r b,
//    q, Qbar, beta^theta), the cylinder's grid values and constants given;
//  * the LINEAR RESPONSE (forward mode through the same tape): d(a b) = da b + a db, d(1/a) = -(1/a)^2 da, d exp a = exp a da,
//    d log a = da / a; each derivative truncated to its value's validity;
//  * the C ABI (mk_*): a context holding the configuration, the tape and the column data; results kept in the context and copied out.
#pragma once
#include "mk_series.hpp"

#include <climits>
#include <cstring>
#include <map>

namespace mk {

enum Op { IN = 0, EMPTY, C0, ADD, SUB, MUL, SCALE, DRAD, DTH, INV, EXP, LOG };
static const char* FAMS[6] = {"QF", "PS", "PH", "BR", "BT", "QB"};   // n2a_nl.FAMS (column families)

struct Tape {
    std::vector<int> op, a, b;
    std::vector<cd> sc;
    std::vector<int> ce;                       // constant r-power (INT_MIN: None)
    std::vector<std::vector<cd>> cv;           // constant values on the grid
    int inputs[6];                             // psi2 Phb qf br bt Qb
    int outputs[6];                            // H_c E_sigma L_K E_sh^r E_sh^th E_Q
};

struct Cols {                                  // per column family: the angular bases on the grid, (ncol, NF)
    std::vector<int> ncol;
    std::vector<std::vector<double>> B;
};

struct Cyl {                                   // the cylinder (lattice 0) on the grid and its constants
    std::vector<double> U, P, b, q, Qb, bt;
    double bnorm = 1.0;
    int e_inv_r = -1, e_Ph_r = -1, e_r = 1;    // C0(1/r), C0(Ph / r), C0(r): r-powers and values (n2a_nl.C0's)
    std::vector<cd> v_inv_r, v_Ph_r, v_r;
};

struct Entry {                                 // one column's coefficient: column j, centre, moments (nb, K)
    int j;
    Key t;
    int nb;
    std::vector<cd> m;
};

struct Kernel {
    Ctx c;
    Tape tp;
    Cols cols;
    std::vector<int> colfam, colidx;           // column j -> family index, position within the family
    Cyl cyl;
    Ser out[6], dout[6];
    std::string err;

    // n2a_nl.fields_from's series(f): sum over the family's columns of moments x basis, keyed by centre (first-insertion order)
    std::vector<std::pair<Key, Piece>> series(int f, const std::vector<Entry>& C, double top) {
        std::vector<std::pair<Key, Piece>> d;
        // Python iterates the family's columns in order, each column's centres in its dict order: entries are given in that order
        for (int jj = 0; jj < cols.ncol[f]; jj++)
            for (auto& en : C) {
                if (colfam[en.j] != f || colidx[en.j] != jj) continue;
                if (en.t.v > top) continue;
                bool nz = false;
                for (auto& z : en.m)
                    if (z != 0.0) { nz = true; break; }
                if (!nz) continue;
                Piece p = c.zeros(en.nb);
                const double* Bv = &cols.B[f][size_t(jj) * c.NF];
                for (int b = 0; b < en.nb; b++)
                    for (int k = 0; k < c.K; k++)
                        for (int q = 0; q < c.NF; q++) p.a[(size_t(b) * c.K + k) * c.NF + q] = en.m[size_t(b) * c.K + k] * Bv[q];
                bool found = false;
                for (auto& o : d)
                    if (keq(o.first, en.t)) { o.second = c.padd(o.second, p); found = true; break; }
                if (!found) d.emplace_back(en.t, std::move(p));
            }
        return d;
    }
    void add0(std::vector<std::pair<Key, Piece>>& d, const std::vector<double>& vals) {
        bool nz = false;
        for (double v : vals)
            if (v != 0.0) { nz = true; break; }
        if (!nz) return;
        Piece p = c.zeros(1);
        for (int q = 0; q < c.NF; q++) p.a[q] = vals[q];
        for (auto& o : d)
            if (keq(o.first, Key{0.0, true})) { o.second = c.padd(o.second, p); return; }
        d.emplace_back(Key{0.0, true}, std::move(p));
    }
    static int fam(const char* s) {
        for (int i = 0; i < 6; i++)
            if (!std::strcmp(FAMS[i], s)) return i;
        return -1;
    }
    // the six fields (psi2, Phb, qf, br, bt, Qb); with lin = true the LINEAR part only (dC: no cylinder) and dU for psi2's chain rule
    void fields(const std::vector<Entry>& C, double top, Ser F[6], Ser* U2 = nullptr, bool lin = false) {
        auto U = series(fam("PS"), C, top);
        if (!lin) add0(U, cyl.U);
        for (auto& kv : U)
            for (auto& z : kv.second.a) z *= 2.0;
        Ser U2s = c.make(std::move(U), true, 0, top);
        if (U2) *U2 = U2s;
        Ser cinv = c.const_ser(cyl.e_inv_r, cyl.v_inv_r);
        F[0] = lin ? Ser() : c.mul(cinv, c.expser(U2s));
        auto p = series(fam("PH"), C, top);
        if (!lin) add0(p, std::vector<double>(c.NF, 1.0));
        std::vector<std::pair<Key, Piece>> pp;
        for (auto& kv : p) {
            Piece w = kv.second;
            for (int b = 0; b < w.B; b++)
                for (int k = 0; k < c.K; k++)
                    for (int q = 0; q < c.NF; q++) w.a[(size_t(b) * c.K + k) * c.NF + q] *= cyl.P[q];
            pp.emplace_back(Key{kv.first.v + 1, kv.first.isint}, std::move(w));
        }
        F[1] = c.mul(c.const_ser(cyl.e_Ph_r, cyl.v_Ph_r), c.make(std::move(pp), true, 0, top + 1));
        auto bs = series(fam("BR"), C, top);
        for (auto& kv : bs)
            for (auto& z : kv.second.a) z *= cyl.bnorm;
        if (!lin) add0(bs, cyl.b);
        F[3] = c.mul(c.const_ser(cyl.e_r, cyl.v_r), c.make(std::move(bs), true, 0, top));
        auto qf = series(fam("QF"), C, top);
        if (!lin) add0(qf, cyl.q);
        auto Qb = series(fam("QB"), C, top);
        if (!lin) add0(Qb, cyl.Qb);
        auto bt = series(fam("BT"), C, top);
        if (!lin) add0(bt, cyl.bt);
        F[2] = c.make(std::move(qf), true, 0, top);
        F[4] = c.make(std::move(bt), true, 0, top);
        F[5] = c.make(std::move(Qb), true, 0, top);
    }

    Ser Fin[6];                                // fields given directly (mk_rows_fields)
    bool use_fin = false;
    std::vector<Ser>* keep_v = nullptr;        // diagnostic (jvp_check): every op's value / derivative kept
    std::vector<Ser>* keep_dv = nullptr;
    // diagnostic: each op's linear response against the central difference of its values at C +- h dC; the first ops that differ reported
    void jvp_check(const std::vector<Entry>& C, const std::vector<Entry>& dC, double top, double h, int nrep = 8) {
        auto shifted = [&](double e) {
            std::vector<Entry> Cx = C;
            for (auto& d : dC) {
                int at = -1;
                for (size_t q = 0; q < Cx.size(); q++)
                    if (Cx[q].j == d.j && keq(Cx[q].t, d.t)) { at = int(q); break; }
                std::vector<cd> m(c.K);
                for (int k = 0; k < c.K; k++) m[k] = (at >= 0 ? Cx[at].m[k] : cd(0.0, 0.0)) + e * d.m[k];
                if (at >= 0) Cx[at].m = m;
                else Cx.push_back(Entry{d.j, d.t, 1, m});
            }
            return Cx;
        };
        std::vector<Ser> vp, vm, v0, dv;
        keep_v = &vp; run(shifted(h), nullptr, top);
        keep_v = &vm; run(shifted(-h), nullptr, top);
        keep_v = &v0; keep_dv = &dv; run(C, &dC, top);
        keep_v = nullptr; keep_dv = nullptr;
        static const char* ON[] = {"IN", "EMPTY", "C0", "ADD", "SUB", "MUL", "SCALE", "DRAD", "DTH", "INV", "EXP", "LOG"};
        int rep = 0;
        for (size_t i = 0; i < dv.size() && rep < nrep; i++) {
            std::vector<Key> keys;
            for (auto* S : {&vp[i], &vm[i], &dv[i]})
                for (auto& kv : S->d) {
                    bool f = false;
                    for (auto& k : keys) if (keq(k, kv.first)) { f = true; break; }
                    if (!f) keys.push_back(kv.first);
                }
            double sc = 0.0, worst = 0.0;
            Key wk{0.0, true};
            double wfd = 0.0, wjv = 0.0;
            for (auto& k : keys) {
                int ip = vp[i].find(k), im = vm[i].find(k), id = dv[i].find(k);
                size_t P = c.psz();
                for (size_t q = 0; q < P; q++) {
                    cd a = ip >= 0 ? vp[i].d[ip].second.a[q] : cd(0.0), b = im >= 0 ? vm[i].d[im].second.a[q] : cd(0.0);
                    cd fd = (a - b) / (2 * h), jv = id >= 0 ? dv[i].d[id].second.a[q] : cd(0.0);
                    sc = std::max(sc, std::max(std::abs(fd), std::abs(jv)));
                    if (std::abs(fd - jv) > worst) { worst = std::abs(fd - jv); wk = k; wfd = std::abs(fd); wjv = std::abs(jv); }
                }
            }
            if (worst > 1e-5 * std::max(sc, 1e-300) && sc > 1e-20) {
                std::fprintf(stderr, "jvp_check op %zu %s (a %d b %d): worst at key %.6f%s |FD| %.3e |JVP| %.3e diff %.3e (scale %.3e); tops v %.3f dv %.3f; pmin v %.3f\n",
                             i, ON[tp.op[i]], tp.a[i], tp.b[i], wk.v, wk.isint ? "i" : "", wfd, wjv, worst, sc, v0[i].top, dv[i].top, v0[i].pmin());
                rep++;
                if (std::getenv("MK_JVP_KEYS")) {
                    auto pk = [&](const char* nm, const Ser& S) {
                        std::fprintf(stderr, "   %s (e %d top %.3f):", nm, S.e, S.top);
                        for (auto& kv : S.d) std::fprintf(stderr, " %.6f", kv.first.v);
                        std::fprintf(stderr, "\n");
                    };
                    pk("value", v0[i]); pk("value+h", vp[i]); pk("jvp", dv[i]);
                    if (tp.a[i] >= 0 && size_t(tp.a[i]) < dv.size()) { pk("input value", v0[tp.a[i]]); pk("input jvp", dv[tp.a[i]]); }
                    std::fprintf(stderr, "   registry now:");
                    for (double x : c.cent) std::fprintf(stderr, " %.6f", x);
                    std::fprintf(stderr, "\n");
                }
            }
        }
    }
    void run(const std::vector<Entry>& C, const std::vector<Entry>* dC, double top) {
        Ser F[6], dF[6], U2, dU2;
        if (use_fin) {
            for (int k = 0; k < 6; k++) F[k] = Fin[k];
        } else {
            fields(C, top, F, &U2);
        }
        if (dC) {
            fields(*dC, top, dF, &dU2, true);
            // psi2 = r^-1 exp(2U): d psi2 = psi2 x 2 dU (U2 carries the factor 2)
            dF[0] = c.trunc(c.mul(F[0], dU2), F[0].top);
        }
        size_t N = tp.op.size();
        std::vector<Ser> v(N), dv(dC ? N : 0);
        auto lin = [&](size_t i, Ser s) { dv[i] = c.trunc(s, v[i].top); };
        for (size_t i = 0; i < N; i++) {
            int o = tp.op[i], A = tp.a[i], Bq = tp.b[i];
            switch (o) {
            case IN:
                v[i] = F[A];
                if (dC) dv[i] = dF[A];
                break;
            case EMPTY:
                v[i] = c.emptyser();
                if (dC) dv[i] = c.emptyser();
                break;
            case C0:
                v[i] = tp.ce[A] == INT_MIN ? c.emptyser() : c.const_ser(tp.ce[A], tp.cv[A]);
                if (dC) dv[i] = c.emptyser();
                break;
            case ADD:
                v[i] = c.add(v[A], v[Bq]);
                if (dC) lin(i, c.add(dv[A], dv[Bq]));
                break;
            case SUB:
                v[i] = c.sub(v[A], v[Bq]);
                if (dC) lin(i, c.sub(dv[A], dv[Bq]));
                break;
            case MUL:
                v[i] = c.mul(v[A], v[Bq]);
                if (dC) {                               // dA b + a dB in Python's order (products register centres; C++ argument order is unspecified)
                    Ser p1 = c.mul(dv[A], v[Bq]);
                    Ser p2 = c.mul(v[A], dv[Bq]);
                    lin(i, c.add(p1, p2));
                }
                break;
            case SCALE:
                v[i] = c.scale(v[A], tp.sc[i]);
                if (dC) lin(i, c.scale(dv[A], tp.sc[i]));
                break;
            case DRAD:
                v[i] = c.drad(v[A]);
                if (dC) lin(i, c.drad(dv[A]));
                break;
            case DTH:
                v[i] = c.dth(v[A]);
                if (dC) lin(i, c.dth(dv[A]));
                break;
            case INV:
                v[i] = c.inv(v[A]);
                if (dC) lin(i, c.scale(c.mul(c.mul(v[i], v[i]), dv[A]), -1.0));
                break;
            case EXP:
                v[i] = c.expser(v[A]);
                if (dC) lin(i, c.mul(v[i], dv[A]));
                break;
            case LOG:
                v[i] = c.logser(v[A]);
                if (dC) lin(i, dv[A].empty() ? c.emptyser() : c.mul(dv[A], c.inv(v[A])));
                break;
            default:
                throw std::runtime_error("mk: unknown op");
            }
        }
        for (int k = 0; k < 6; k++) {
            out[k] = v[tp.outputs[k]];
            if (dC) dout[k] = dv[tp.outputs[k]];
        }
        if (keep_v) *keep_v = v;
        if (keep_dv && dC) *keep_dv = dv;
    }

    // ---- G1 (NOTES_M4_DESIGN.md section 19; research round 1049): the SECOND-ORDER forward pass -- d1 a series direction dC (as run's), d2
    // a direction of the cylinder (grade 0: its grid values, K, bnorm, n), d12 the mixed derivative.  The operator build reads d1 (M); d12 is
    // its derivative with respect to the cylinder (dM / d theta).  n enters the ops only through drad (e + n key, the log moment n v[k + 1])
    // and logser (e / n); the fields carry the rest ----
    struct DCyl {                              // d / d theta of the cylinder: grid values, K (the Ph / r constant), bnorm, n
        std::vector<double> U, P, b, q, Qb, bt;
        double dK = 0.0, dbnorm = 0.0, dn = 0.0;
    };
    Ser d2out[6], d12out[6];
    Ser cser(const std::vector<double>& vals, double top) {     // a grid function as a key-0 series (empty if zero)
        std::vector<std::pair<Key, Piece>> d;
        if (!vals.empty()) add0(d, vals);
        return d.empty() ? c.emptyser(top) : c.make(std::move(d), true, 0, top);
    }
    // the shifted P x series(f) product of fields() (Phb's), with the grid factor w
    std::vector<std::pair<Key, Piece>> pshift(const std::vector<std::pair<Key, Piece>>& p, const std::vector<double>& w) {
        std::vector<std::pair<Key, Piece>> pp;
        for (auto& kv : p) {
            Piece x = kv.second;
            for (int b = 0; b < x.B; b++)
                for (int k = 0; k < c.K; k++)
                    for (int q = 0; q < c.NF; q++) x.a[(size_t(b) * c.K + k) * c.NF + q] *= w[q];
            pp.emplace_back(Key{kv.first.v + 1, kv.first.isint}, std::move(x));
        }
        return pp;
    }
    // the fields' d2 (F2) and d12 (F12) at the series C in the series direction dC (F, dF, dU2: run's)
    void fields2(const std::vector<Entry>& C, const std::vector<Entry>& dC, double top, const DCyl& d, const Ser F[6], const Ser dF[6],
                 const Ser& dU2, Ser F2[6], Ser F12[6]) {
        for (int k = 0; k < 6; k++) { F2[k] = c.emptyser(); F12[k] = c.emptyser(); }
        // psi2 = r^-1 exp(2 U): d2 = psi2 2 dU, d12 = psi2 (2 dU_C) (2 dU)
        if (!d.U.empty()) {
            std::vector<double> u2(d.U.size());
            for (size_t i = 0; i < u2.size(); i++) u2[i] = 2.0 * d.U[i];
            Ser du = cser(u2, top);
            if (!du.empty()) {
                F2[0] = c.trunc(c.mul(F[0], du), F[0].top);
                if (!dU2.empty()) F12[0] = c.trunc(c.mul(c.mul(F[0], dU2), du), F[0].top);
            }
        }
        // Phbar = (K / r) x [P (1 + series(PH))] shifted: d2 = (dK / K) Phb + (K / r) [dP (1 + series)], d12 the same on series(dC)
        {
            Ser cP = c.const_ser(cyl.e_Ph_r, cyl.v_Ph_r);
            double Kv = cyl.v_Ph_r.empty() ? 1.0 : cyl.v_Ph_r[0].real();
            Ser a2 = d.dK != 0.0 && !F[1].empty() ? c.scale(F[1], d.dK / Kv) : c.emptyser();
            Ser a12 = d.dK != 0.0 && !dF[1].empty() ? c.scale(dF[1], d.dK / Kv) : c.emptyser();
            if (!d.P.empty()) {
                auto p = series(fam("PH"), C, top);
                add0(p, std::vector<double>(c.NF, 1.0));
                Ser t2 = c.mul(cP, c.make(pshift(p, d.P), true, 0, top + 1));
                a2 = a2.empty() ? t2 : c.add(a2, t2);
                auto pd = series(fam("PH"), dC, top);
                if (!pd.empty()) {
                    Ser t12 = c.mul(cP, c.make(pshift(pd, d.P), true, 0, top + 1));
                    a12 = a12.empty() ? t12 : c.add(a12, t12);
                }
            }
            F2[1] = a2;
            F12[1] = a12;
        }
        // beta^r = r (b + bnorm series(BR)): d2 = r (db + dbnorm series), d12 = r dbnorm series(dC)
        {
            Ser cr = c.const_ser(cyl.e_r, cyl.v_r);
            auto bs = series(fam("BR"), C, top);
            for (auto& kv : bs)
                for (auto& z : kv.second.a) z *= d.dbnorm;
            if (!d.b.empty()) add0(bs, d.b);
            if (!bs.empty()) F2[3] = c.mul(cr, c.make(std::move(bs), true, 0, top));
            if (d.dbnorm != 0.0) {
                auto bd = series(fam("BR"), dC, top);
                for (auto& kv : bd)
                    for (auto& z : kv.second.a) z *= d.dbnorm;
                if (!bd.empty()) F12[3] = c.mul(cr, c.make(std::move(bd), true, 0, top));
            }
        }
        F2[2] = cser(d.q, top);                                    // qf, beta^theta, Qbar: additive in the cylinder
        F2[4] = cser(d.bt, top);
        F2[5] = cser(d.Qb, top);
    }
    // d / dn of drad at fixed keys: (e + n key) v + n v[k + 1]  ->  key v + v[k + 1]
    Ser drad_dn(const Ser& a) {
        if (a.empty()) return c.emptyser(a.top);
        std::vector<std::pair<Key, Piece>> d;
        for (auto& kv : a.d) {
            const Piece& v = kv.second;
            Piece w = c.zeros(v.B);
            for (int b = 0; b < v.B; b++)
                for (int k = 0; k < c.K; k++)
                    for (int q = 0; q < c.NF; q++) {
                        size_t i = (size_t(b) * c.K + k) * c.NF + q;
                        w.a[i] = kv.first.v * v.a[i];
                        if (k < c.K - 1) w.a[i] += v.a[i + c.NF];
                    }
            d.emplace_back(kv.first, std::move(w));
        }
        return c.make(std::move(d), true, a.e - 1, a.top);
    }
    // d / dn of logser at fixed keys: its key-0 moment-1 term e / n  ->  - e / n^2
    Ser log_dn(const Ser& a) {
        if (a.empty() || c.K < 2 || a.e == 0) return c.emptyser();
        Piece v = c.zeros(1);
        for (int q = 0; q < c.NF; q++) v.a[size_t(c.NF) + q] = -double(a.e) / (c.n * c.n);
        std::vector<std::pair<Key, Piece>> d;
        d.emplace_back(Key{0.0, true}, std::move(v));
        return c.make(std::move(d), true, 0, BIG);
    }
    void run2(const std::vector<Entry>& C, const std::vector<Entry>& dC, const DCyl& dcy, double top) {
        Ser F[6], dF[6], U2, dU2, F2[6], F12[6];
        fields(C, top, F, &U2);
        fields(dC, top, dF, &dU2, true);
        dF[0] = c.trunc(c.mul(F[0], dU2), F[0].top);
        fields2(C, dC, top, dcy, F, dF, dU2, F2, F12);
        size_t N = tp.op.size();
        std::vector<Ser> v(N), d1(N), d2(N), d12(N);
        auto T = [&](size_t i, const Ser& s_) { return c.trunc(s_, v[i].top); };
        auto A2 = [&](const Ser& x, const Ser& y) { return x.empty() ? y : y.empty() ? x : c.add(x, y); };
        auto M2 = [&](const Ser& x, const Ser& y) { return (x.empty() || y.empty()) ? c.emptyser() : c.mul(x, y); };
        auto S2 = [&](const Ser& x, cd f) { return x.empty() ? x : c.scale(x, f); };
        for (size_t i = 0; i < N; i++) {
            int o = tp.op[i], A = tp.a[i], Bq = tp.b[i];
            switch (o) {
            case IN:
                v[i] = F[A]; d1[i] = dF[A]; d2[i] = F2[A]; d12[i] = F12[A];
                break;
            case EMPTY:
                v[i] = c.emptyser(); d1[i] = d2[i] = d12[i] = c.emptyser();
                break;
            case C0:
                v[i] = tp.ce[A] == INT_MIN ? c.emptyser() : c.const_ser(tp.ce[A], tp.cv[A]);
                d1[i] = d2[i] = d12[i] = c.emptyser();
                break;
            case ADD:
            case SUB: {
                cd sg = o == ADD ? 1.0 : -1.0;
                v[i] = o == ADD ? c.add(v[A], v[Bq]) : c.sub(v[A], v[Bq]);
                d1[i] = T(i, A2(d1[A], S2(d1[Bq], sg)));
                d2[i] = T(i, A2(d2[A], S2(d2[Bq], sg)));
                d12[i] = T(i, A2(d12[A], S2(d12[Bq], sg)));
                break;
            }
            case MUL:
                v[i] = c.mul(v[A], v[Bq]);
                d1[i] = T(i, A2(M2(d1[A], v[Bq]), M2(v[A], d1[Bq])));
                d2[i] = T(i, A2(M2(d2[A], v[Bq]), M2(v[A], d2[Bq])));
                d12[i] = T(i, A2(A2(M2(d12[A], v[Bq]), M2(d1[A], d2[Bq])), A2(M2(d2[A], d1[Bq]), M2(v[A], d12[Bq]))));
                break;
            case SCALE:
                v[i] = c.scale(v[A], tp.sc[i]);
                d1[i] = T(i, S2(d1[A], tp.sc[i])); d2[i] = T(i, S2(d2[A], tp.sc[i])); d12[i] = T(i, S2(d12[A], tp.sc[i]));
                break;
            case DRAD: {
                v[i] = c.drad(v[A]);
                d1[i] = T(i, d1[A].empty() ? d1[A] : c.drad(d1[A]));
                Ser n2 = dcy.dn != 0.0 ? S2(drad_dn(v[A]), dcy.dn) : c.emptyser();
                Ser n12 = dcy.dn != 0.0 ? S2(drad_dn(d1[A]), dcy.dn) : c.emptyser();
                d2[i] = T(i, A2(d2[A].empty() ? d2[A] : c.drad(d2[A]), n2));
                d12[i] = T(i, A2(d12[A].empty() ? d12[A] : c.drad(d12[A]), n12));
                break;
            }
            case DTH:
                v[i] = c.dth(v[A]);
                d1[i] = T(i, c.dth(d1[A])); d2[i] = T(i, c.dth(d2[A])); d12[i] = T(i, c.dth(d12[A]));
                break;
            case INV: {
                v[i] = c.inv(v[A]);
                Ser v2 = c.mul(v[i], v[i]);
                d1[i] = T(i, S2(M2(v2, d1[A]), -1.0));
                d2[i] = T(i, S2(M2(v2, d2[A]), -1.0));
                d12[i] = T(i, A2(S2(M2(M2(c.mul(v2, v[i]), d1[A]), d2[A]), 2.0), S2(M2(v2, d12[A]), -1.0)));
                break;
            }
            case EXP:
                v[i] = c.expser(v[A]);
                d1[i] = T(i, M2(v[i], d1[A]));
                d2[i] = T(i, M2(v[i], d2[A]));
                d12[i] = T(i, M2(v[i], A2(d12[A], M2(d1[A], d2[A]))));
                break;
            case LOG: {
                v[i] = c.logser(v[A]);
                Ser ia = (d1[A].empty() && d2[A].empty() && d12[A].empty()) ? c.emptyser() : c.inv(v[A]);
                d1[i] = T(i, M2(d1[A], ia));
                Ser n2 = dcy.dn != 0.0 ? S2(log_dn(v[A]), dcy.dn) : c.emptyser();
                d2[i] = T(i, A2(M2(d2[A], ia), n2));
                d12[i] = T(i, A2(M2(d12[A], ia), S2(M2(M2(d1[A], d2[A]), M2(ia, ia)), -1.0)));
                break;
            }
            default:
                throw std::runtime_error("mk: unknown op");
            }
        }
        for (int k = 0; k < 6; k++) {
            out[k] = v[tp.outputs[k]];
            dout[k] = d1[tp.outputs[k]];
            d2out[k] = d2[tp.outputs[k]];
            d12out[k] = d12[tp.outputs[k]];
        }
    }
};

}  // namespace mk

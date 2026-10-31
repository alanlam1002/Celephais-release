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
    }
};

}  // namespace mk

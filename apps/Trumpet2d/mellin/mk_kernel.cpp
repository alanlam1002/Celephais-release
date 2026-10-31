// Mellin throat kernel (code round 463 M1, research round 1013): the row assembly and its analytic linear response, with a C ABI.
//
//  * the TAPE (spinning_trumpet/scripts/m1_tape.py): the throat equations' assembly as executed by the Python engine (finiteJ_system.TH2 with
//    the C3 edits, run with MS), recorded in execution order -- every op replayed here, none pruned (products register off-lattice centres);
//  * fields_from: stage B's columns -> the six fields (n2a_nl.fields_from: psi2 = r^-1 exp(2U), Phibar = K / r x P_shape x ..., beta^r = r b,
//    q, Qbar, beta^theta), the cylinder's grid values and constants given;
//  * the LINEAR RESPONSE (forward mode through the same tape): d(a b) = da b + a db, d(1/a) = -(1/a)^2 da, d exp a = exp a da,
//    d log a = da / a; each derivative truncated to its value's validity;
//  * the C ABI (mk_*): a context holding the configuration, the tape and the column data; results kept in the context and copied out.
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
                if (dC) lin(i, c.add(c.mul(dv[A], v[Bq]), c.mul(v[A], dv[Bq])));
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

using mk::cd;

// ------------------------------------------------------------------------------------------------------------------------------------------
// the C ABI (complex arrays as interleaved doubles)
// ------------------------------------------------------------------------------------------------------------------------------------------
extern "C" {

static std::vector<cd> cvec(const double* p, size_t n) {
    std::vector<cd> v(n);
    for (size_t i = 0; i < n; i++) v[i] = cd(p[2 * i], p[2 * i + 1]);
    return v;
}

void* mk_new(int K, int NF, double n, double Rm, const double* D) {
    auto* k = new mk::Kernel();
    k->c.init(K, NF, n, Rm, cvec(D, size_t(NF) * NF));
    return k;
}
void mk_free(void* h) { delete static_cast<mk::Kernel*>(h); }
const char* mk_error(void* h) { return static_cast<mk::Kernel*>(h)->err.c_str(); }

void mk_set_cent(void* h, const double* c, int nc) {
    auto* k = static_cast<mk::Kernel*>(h);
    k->c.cent.assign(c, c + nc);
}
int mk_get_cent(void* h, double* out, int cap) {
    auto* k = static_cast<mk::Kernel*>(h);
    int n = int(k->c.cent.size());
    for (int i = 0; i < std::min(n, cap); i++) out[i] = k->c.cent[i];
    return n;
}

// ops: nops x 3 ints (op, a, b); sc: nops complex; consts: e (INT_MIN = None) and NF complex values each; io: 6 inputs, 6 outputs
void mk_set_tape(void* h, int nops, const int* ops, const double* sc, int nconst, const int* ce, const double* cv, const int* io) {
    auto* k = static_cast<mk::Kernel*>(h);
    auto& t = k->tp;
    t.op.resize(nops); t.a.resize(nops); t.b.resize(nops);
    for (int i = 0; i < nops; i++) { t.op[i] = ops[3 * i]; t.a[i] = ops[3 * i + 1]; t.b[i] = ops[3 * i + 2]; }
    t.sc = cvec(sc, nops);
    t.ce.assign(ce, ce + nconst);
    t.cv.clear();
    for (int i = 0; i < nconst; i++) t.cv.push_back(cvec(cv + 2 * size_t(i) * k->c.NF, k->c.NF));
    for (int i = 0; i < 6; i++) { t.inputs[i] = io[i]; t.outputs[i] = io[6 + i]; }
}

// columns: ncol per family (FAMS order), the bases (concatenated, ncol x NF each, family order), column j -> (family, position)
void mk_set_cols(void* h, const int* ncol, const double* B, int nn, const int* colfam, const int* colidx) {
    auto* k = static_cast<mk::Kernel*>(h);
    k->cols.ncol.assign(ncol, ncol + 6);
    k->cols.B.assign(6, {});
    size_t off = 0;
    for (int f = 0; f < 6; f++) {
        size_t m = size_t(ncol[f]) * k->c.NF;
        k->cols.B[f].assign(B + off, B + off + m);
        off += m;
    }
    k->colfam.assign(colfam, colfam + nn);
    k->colidx.assign(colidx, colidx + nn);
}

// the cylinder: grid values U, P, b, q, Qb, bt (6 x NF), bnorm, and the three constants' r-powers and values (3 x NF complex)
void mk_set_cyl(void* h, const double* vals, double bnorm, const int* ce, const double* cv) {
    auto* k = static_cast<mk::Kernel*>(h);
    int NF = k->c.NF;
    auto sl = [&](int i) { return std::vector<double>(vals + size_t(i) * NF, vals + size_t(i + 1) * NF); };
    k->cyl.U = sl(0); k->cyl.P = sl(1); k->cyl.b = sl(2); k->cyl.q = sl(3); k->cyl.Qb = sl(4); k->cyl.bt = sl(5);
    k->cyl.bnorm = bnorm;
    k->cyl.e_inv_r = ce[0]; k->cyl.e_Ph_r = ce[1]; k->cyl.e_r = ce[2];
    k->cyl.v_inv_r = cvec(cv, NF);
    k->cyl.v_Ph_r = cvec(cv + 2 * NF, NF);
    k->cyl.v_r = cvec(cv + 4 * NF, NF);
}

static std::vector<mk::Entry> entries(int nent, const int* col, const double* key, const int* isint, const int* nb, const double* mom, int K) {
    std::vector<mk::Entry> C(nent);
    size_t off = 0;
    for (int i = 0; i < nent; i++) {
        C[i].j = col[i];
        C[i].t = mk::Key{key[i], isint[i] != 0};
        C[i].nb = nb[i];
        C[i].m = cvec(mom + 2 * off, size_t(nb[i]) * K);
        off += size_t(nb[i]) * K;
    }
    return C;
}

// the rows (and, with dnent >= 0, their linear response along dC); returns 0 on success
int mk_rows(void* h, double top, int nent, const int* col, const double* key, const int* isint, const int* nb, const double* mom,
            int dnent, const int* dcol, const double* dkey, const int* disint, const int* dnb, const double* dmom) {
    auto* k = static_cast<mk::Kernel*>(h);
    try {
        auto C = entries(nent, col, key, isint, nb, mom, k->c.K);
        if (dnent >= 0) {
            auto dC = entries(dnent, dcol, dkey, disint, dnb, dmom, k->c.K);
            k->run(C, &dC, top);
        } else {
            k->run(C, nullptr, top);
        }
        k->err.clear();
        return 0;
    } catch (const std::exception& ex) {
        k->err = ex.what();
        return 1;
    }
}

// the rows from FIELDS given directly (psi2 Phb qf br bt Qb, each: np pieces, keys, isint, nb, has_e, e, top, data B x K x NF complex) --
// the gauge flow's fields (n2a_nl: run_assembly(Fgauge)); the pieces taken as given (no canonicalisation: they are canonical already)
int mk_rows_fields(void* h, const int* np, const double* keys, const int* isint, const int* nb, const int* has_e, const int* e,
                   const double* top, const double* data) {
    auto* k = static_cast<mk::Kernel*>(h);
    try {
        size_t ip = 0, off = 0, P = k->c.psz();
        for (int f = 0; f < 6; f++) {
            mk::Ser s;
            for (int i = 0; i < np[f]; i++, ip++) {
                mk::Piece p{nb[ip], std::vector<cd>(size_t(nb[ip]) * P)};
                for (size_t q = 0; q < p.a.size(); q++, off++) p.a[q] = cd(data[2 * off], data[2 * off + 1]);
                s.d.emplace_back(mk::Key{keys[ip], isint[ip] != 0}, std::move(p));
            }
            s.has_e = has_e[f] != 0 && !s.d.empty();
            s.e = e[f];
            s.top = top[f];
            k->Fin[f] = std::move(s);
        }
        k->use_fin = true;
        std::vector<mk::Entry> none;
        k->run(none, nullptr, 0.0);
        k->use_fin = false;
        k->err.clear();
        return 0;
    } catch (const std::exception& ex) {
        k->use_fin = false;
        k->err = ex.what();
        return 1;
    }
}

// an output family's shape: number of pieces, r-power (has_e), validity, total doubles of its data
void mk_out_info(void* h, int f, int deriv, int* np, int* has_e, int* e, double* top, long* ndata) {
    auto* k = static_cast<mk::Kernel*>(h);
    const mk::Ser& s = deriv ? k->dout[f] : k->out[f];
    *np = int(s.d.size());
    *has_e = s.has_e;
    *e = s.e;
    *top = s.top;
    long n = 0;
    for (auto& kv : s.d) n += long(kv.second.a.size()) * 2;
    *ndata = n;
}
void mk_out_get(void* h, int f, int deriv, double* keys, int* isint, int* nb, double* data) {
    auto* k = static_cast<mk::Kernel*>(h);
    const mk::Ser& s = deriv ? k->dout[f] : k->out[f];
    size_t off = 0;
    for (size_t i = 0; i < s.d.size(); i++) {
        keys[i] = s.d[i].first.v;
        isint[i] = s.d[i].first.isint;
        nb[i] = s.d[i].second.B;
        for (auto& z : s.d[i].second.a) { data[off++] = z.real(); data[off++] = z.imag(); }
    }
}

}  // extern "C"

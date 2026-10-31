// Mellin throat kernel: the C ABI (M1: the rows and their linear response; M2a: the operator at a centre).
#include "mk_kernel.hpp"
#include "mk_operator.hpp"
#include "mk_la.hpp"
#include "mk_bundle.hpp"

#include <climits>
#include <cstring>
#include <map>


using mk::cd;

// ------------------------------------------------------------------------------------------------------------------------------------------
// the C ABI (complex arrays as interleaved doubles)
// ------------------------------------------------------------------------------------------------------------------------------------------
static std::map<void*, mk::Proj> g_proj;                  // the projection tables per kernel handle

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

// ---- M2a: the operator at a centre ----

// gfam / dE: per Galerkin row (family in the kernel's row order); W: nn x NF complex (P.I^T P.T[i]); likewise the condition rows; kofs per column
void mk_set_proj(void* h, int nn, int nc, int nb, const int* gfam, const int* dE, const double* W, const int* cfam, const int* cdE,
                 const double* Wc, const int* kofs) {
    auto* k = static_cast<mk::Kernel*>(h);
    mk::Proj& P = g_proj[h];
    int NF = k->c.NF;
    P.nn = nn; P.nc = nc; P.nb = nb;
    P.gfam.assign(gfam, gfam + nn); P.dE.assign(dE, dE + nn);
    P.W = cvec(W, size_t(nn) * NF);
    P.cfam.assign(cfam, cfam + nc); P.cdE.assign(cdE, cdE + nc);
    P.Wc = cvec(Wc, size_t(nc) * NF);
    P.kofs.assign(kofs, kofs + nn);
}

// the operator at centre g (value, isint) about the series given: mode 0 FD (h, chunk), 1 JVP; Ms (3 NB NB), Mc (3 nc NB) complex out
int mk_meff(void* h, double g, double top, int mode, double hfd, int chunk, int nent, const int* col, const double* key, const int* isint,
            const int* nb, const double* mom, double* Ms_out, double* Mc_out) {
    auto* k = static_cast<mk::Kernel*>(h);
    try {
        auto C = entries(nent, col, key, isint, nb, mom, k->c.K);
        std::vector<cd> Ms, Mc;
        mk::meff(*k, g_proj.at(h), C, g, top, mode, hfd, chunk, Ms, Mc);
        for (size_t i = 0; i < Ms.size(); i++) { Ms_out[2 * i] = Ms[i].real(); Ms_out[2 * i + 1] = Ms[i].imag(); }
        for (size_t i = 0; i < Mc.size(); i++) { Mc_out[2 * i] = Mc[i].real(); Mc_out[2 * i + 1] = Mc[i].imag(); }
        k->err.clear();
        return 0;
    } catch (const std::exception& ex) {
        k->err = ex.what();
        return 1;
    }
}

// the re-centred operator: M0 (3 x nm complex) about g0 -> M (3 x nm) about g0 + d
void mk_recentre(const double* M0, long nm, double d, double* M_out) {
    std::vector<cd> a = cvec(M0, size_t(3 * nm)), o;
    mk::recentre(a, o, size_t(nm), d);
    for (size_t i = 0; i < o.size(); i++) { M_out[2 * i] = o[i].real(); M_out[2 * i + 1] = o[i].imag(); }
}

// a kernel configured from a setup bundle directory (m2_bundle.py), its projections installed: no Python setup involved
void* mk_new_from_bundle(const char* dir) {
    auto* k = new mk::Kernel();
    try {
        mk::Bundle B;
        mk::load_bundle(dir, *k, g_proj[k], B);
    } catch (const std::exception& ex) {
        k->err = ex.what();
    }
    return k;
}

// ---- the linear-algebra wrapper's test entries (m2_la_test.py: against numpy / scipy on the same matrices) ----
static std::vector<cd> cm(const double* p, size_t n) { return cvec(p, n); }
static void put(const std::vector<cd>& v, double* o) { for (size_t i = 0; i < v.size(); i++) { o[2 * i] = v[i].real(); o[2 * i + 1] = v[i].imag(); } }
int mk_la_inv(const double* A, int n, double* out) { try { put(mk::la::inv(cm(A, size_t(n) * n), n), out); return 0; } catch (...) { return 1; } }
int mk_la_svd(const double* A, int m, int n, int full, double* s, double* U, double* Vh) {
    try {
        std::vector<double> sv; std::vector<cd> u, v;
        mk::la::svd(cm(A, size_t(m) * n), m, n, sv, U ? &u : nullptr, U ? &v : nullptr, full != 0);
        for (size_t i = 0; i < sv.size(); i++) s[i] = sv[i];
        if (U) { put(u, U); put(v, Vh); }
        return 0;
    } catch (...) { return 1; }
}
int mk_la_lstsq(const double* A, int m, int n, const double* B, int nrhs, double* X, int* rank) {
    try { int rk; put(mk::la::lstsq(cm(A, size_t(m) * n), m, n, cm(B, size_t(m) * nrhs), nrhs, nullptr, &rk), X); *rank = rk; return 0; }
    catch (...) { return 1; }
}
int mk_la_qrpiv(const double* A, int m, int n, int* p) {
    try { auto q = mk::la::qrpiv(cm(A, size_t(m) * n), m, n); for (int i = 0; i < n; i++) p[i] = q[i]; return 0; } catch (...) { return 1; }
}
int mk_la_eiggen(const double* A, const double* B, int n, double* ev) {
    try { put(mk::la::eiggen(cm(A, size_t(n) * n), cm(B, size_t(n) * n), n), ev); return 0; } catch (...) { return 1; }
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

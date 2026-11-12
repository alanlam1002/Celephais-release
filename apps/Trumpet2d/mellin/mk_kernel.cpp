// Mellin throat kernel: the C ABI (M1: the rows and their linear response; M2a: the operator at a centre).
#include "mk_kernel.hpp"
#include "mk_operator.hpp"
#include "mk_la.hpp"
#include "mk_bundle.hpp"
#include "mk_data.hpp"
#include "mk_steps.hpp"
#include "mk_families.hpp"

#include <climits>
#include <cstring>
#include <map>
#include <filesystem>


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

void mk_set_exact(void* h, int on) {                    // research round 1031: the exact merge (stage (i)) on / off
    static_cast<mk::Kernel*>(h)->c.exact = on != 0;
}
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

// ---- M2c: the data rule and the conditions (mk_data.hpp); the kernel-backed evaluators are built here (the kernel's translation unit) ----
namespace {
void putv(const std::vector<cd>& v, double* p) {
    for (size_t i = 0; i < v.size(); i++) { p[2 * i] = v[i].real(); p[2 * i + 1] = v[i].imag(); }
}
// the conditions [S0 (nc), G0 (xrows), S1, G1] and the gate rows' largest live residual, about the series `base` at aligned g
mk::dt::Eval make_eval(mk::Kernel& k, const mk::Proj& P, const std::vector<mk::Entry>& base, double g, double top,
                       const std::vector<int>& xrows, const std::vector<cd>& e1, double R) {
    mk::dt::Eval ev;
    ev.cond = [&k, &P, base, g, top, xrows, e1, R](const std::vector<cd>& X) {
        auto Cy = mk::with_grade(k.c, base, X, g, P.kofs, e1, R);
        k.run(Cy, nullptr, top);
        int K = k.c.K, nc = P.nc, nn = P.nn;
        auto S = mk::project(k.c, k.out, g, 1, P.cfam, P.cdE, P.Wc, true);
        std::vector<cd> G;
        if (!xrows.empty()) G = mk::project(k.c, k.out, g, 1, P.gfam, P.dE, P.W, false);
        std::vector<cd> r;
        for (int m = 0; m < 2 && m < K; m++) {
            for (int i = 0; i < nc; i++) r.push_back(S[size_t(m) * nc + i]);
            for (int x : xrows) r.push_back(G[size_t(m) * nn + x]);
        }
        return r;
    };
    ev.gate = [&k, &P, base, g, top, e1, R](const std::vector<cd>* X, double gg) {
        if (P.gtfam.empty()) throw std::runtime_error("mk: the gate rows are not set (mk_set_gate)");
        if (X) k.run(mk::with_grade(k.c, base, *X, g, P.kofs, e1, R), nullptr, top);
        else k.run(base, nullptr, top);
        auto Sg = mk::project(k.c, k.out, gg, 1, P.gtfam, P.gtdE, P.Wg, true);
        double m = 0.0;
        for (size_t q = 0; q < P.gtfam.size(); q++)
            if (P.live[q]) m = std::max(m, std::abs(Sg[q]));
        return m;
    };
    return ev;
}
}  // namespace

extern "C" {

void mk_set_gate(void* h, int ng, const int* gtfam, const int* gtdE, const double* Wg, const int* live) {
    auto* k = static_cast<mk::Kernel*>(h);
    mk::Proj& P = g_proj[h];
    P.gtfam.assign(gtfam, gtfam + ng);
    P.gtdE.assign(gtdE, gtdE + ng);
    P.Wg = cvec(Wg, size_t(ng) * k->c.NF);
    P.live.assign(live, live + ng);
}

// the targets and the data rule at integer centre g.  base: the series (entries); X0 (N x nn), Wq (rk x N x nn) from the contour step;
// targets: the target series (cbb_*), the column labels ('\n'-separated), the --tgt items ('LAB=V;...'), lc.  Outputs: tgtv / wt / tg0 (2 nn),
// nhit; c (rk), At (2nn x rk), F_start / F_final (nF), cond_sv (<= rk), cond_rank, lin_err, tol (tau, size, condtol); hist (5 per entry:
// kind, res, mu, rank, polish; cap 200); fitted (8 per entry: amp re, im, cols 3, l0_share, l2_content, -); Xg (N x nn); Wqn (rk x N x nn)
int mk_data_rule(void* h, double g, double top, int nent, const int* col, const double* key, const int* isint, const int* nbs, const double* mom,
                 int N, double R, const double* e1, int rk, const double* X0, const double* Wq, int nx, const int* xrows, int ncbb,
                 const int* cbb_col, const double* cbb_t, const double* cbb_m0, const char* collab, const char* tgt, const int* lc, int notgt,
                 int cond_fit, int cond_full, double condtol_fixed, int warm, const double* cwarm, double* tgtv_o, double* wt_o, double* tg0_o,
                 int* nhit, double* c_o, double* At_o, int* nF, double* Fs_o, double* Ff_o, double* sv_o, int* cond_rank, double* lin_err,
                 double* tol_o, int* nhist, double* hist_o, int* nfit, double* fit_o, double* Xg_o, double* Wqn_o) {
    auto* k = static_cast<mk::Kernel*>(h);
    try {
        const mk::Proj& P = g_proj.at(h);
        int nn = P.nn;
        auto base = entries(nent, col, key, isint, nbs, mom, k->c.K);
        mk::dt::TgtIn ti;
        ti.nn = nn; ti.rk = rk; ti.g = g; ti.kofs = P.kofs; ti.lc.assign(lc, lc + nn); ti.notgt = notgt != 0;
        {
            std::string s(collab), tok;
            size_t a = 0;
            while (a <= s.size()) {
                size_t b = s.find('\n', a);
                if (b == std::string::npos) b = s.size();
                ti.collab.push_back(s.substr(a, b - a));
                a = b + 1;
            }
            std::string t(tgt);
            a = 0;
            while (a < t.size()) {
                size_t b = t.find(';', a);
                if (b == std::string::npos) b = t.size();
                std::string it = t.substr(a, b - a);
                size_t e = it.find('=');
                if (e != std::string::npos) {
                    std::string kk = it.substr(0, e), vv = it.substr(e + 1);
                    kk.erase(0, kk.find_first_not_of(' '));
                    kk.erase(kk.find_last_not_of(' ') + 1);
                    ti.tgt.emplace_back(kk, std::stod(vv));
                }
                a = b + 1;
            }
        }
        ti.cbb_col.assign(cbb_col, cbb_col + ncbb);
        ti.cbb_t.assign(cbb_t, cbb_t + ncbb);
        ti.cbb_m0 = cvec(cbb_m0, ncbb);
        mk::dt::DataIn in;
        std::vector<std::pair<int, double>> hit;
        *nhit = mk::dt::targets(ti, in.tgtv, in.prevv, in.wt, in.tg0, hit);
        in.nn = nn; in.NB = nn; in.rk = rk; in.R = R; in.g = g;
        in.e1 = cvec(e1, N);
        in.X0 = cvec(X0, size_t(N) * nn);
        for (int q = 0; q < rk; q++) in.Wq.push_back(cvec(Wq + size_t(q) * 2 * N * nn, size_t(N) * nn));
        in.notgt = notgt != 0; in.cond_fit = cond_fit != 0; in.cond_full = cond_full != 0; in.condtol_fixed = condtol_fixed;
        in.warm = warm != 0;
        if (warm) in.cwarm = cvec(cwarm, rk);
        in.lc.assign(lc, lc + nn);
        auto ev = make_eval(*k, P, base, g, top, std::vector<int>(xrows, xrows + nx), in.e1, R);
        mk::dt::DataOut o;
        mk::dt::data_rule(in, ev, o);
        putv(in.tgtv, tgtv_o);
        for (int i = 0; i < 2 * nn; i++) wt_o[i] = in.wt[i];
        putv(in.tg0, tg0_o);
        putv(o.c, c_o);
        putv(o.At, At_o);
        *nF = int(o.F_start.size());
        putv(o.F_start, Fs_o);
        putv(o.F_final, Ff_o);
        for (size_t i = 0; i < o.cond_sv.size(); i++) sv_o[i] = o.cond_sv[i];
        *cond_rank = o.cond_rank;
        *lin_err = o.lin_err;
        tol_o[0] = o.tau; tol_o[1] = o.size; tol_o[2] = o.condtol;
        *nhist = int(std::min<size_t>(o.hist.size(), 200));
        for (int i = 0; i < *nhist; i++) {
            auto& hh = o.hist[i];
            double* p = hist_o + 5 * i;
            p[0] = hh.kind; p[1] = hh.res; p[2] = hh.mu; p[3] = hh.rank; p[4] = hh.polish;
        }
        *nfit = int(o.fitted.size());
        for (int i = 0; i < *nfit; i++) {
            auto& f = o.fitted[i];
            double* p = fit_o + 8 * i;
            p[0] = f.amp.real(); p[1] = f.amp.imag(); p[2] = f.cols[0]; p[3] = f.cols[1]; p[4] = f.cols[2]; p[5] = f.l0_share; p[6] = f.l2_content;
            p[7] = 0.0;
        }
        putv(o.Xg, Xg_o);
        for (int q = 0; q < rk; q++) putv(o.Wq[q], Wqn_o + size_t(q) * 2 * N * nn);
        k->err.clear();
        return 0;
    } catch (const std::exception& ex) {
        k->err = ex.what();
        return 1;
    }
}

// --cond-fit's data fix on X (N x nn): the data rule's At, normalised Wq, weights and tg0 given
int mk_data_fix(int N, int nn, int rk, double R, const double* e1, const double* X, const double* At, const double* Wqn, const double* wt,
                const double* tg0, double* out) {
    try {
        mk::dt::DataIn in;
        in.nn = nn; in.NB = nn; in.rk = rk; in.R = R;
        in.e1 = cvec(e1, N);
        in.wt.assign(wt, wt + 2 * nn);
        in.tg0 = cvec(tg0, 2 * nn);
        mk::dt::DataOut d;
        d.At = cvec(At, size_t(2) * nn * rk);
        for (int q = 0; q < rk; q++) d.Wq.push_back(cvec(Wqn + size_t(q) * 2 * N * nn, size_t(N) * nn));
        putv(mk::dt::data_fix(cvec(X, size_t(N) * nn), in, d), out);
        return 0;
    } catch (...) { return 1; }
}

// the off-integer start: base (the series STRIPPED at g), X0, Wq (raw); labelled / datum; cond_fit.  Outputs Wqn (rk x N x nn), Xg, fixed, amp
int mk_offl_start(void* h, double g, double top, int nent, const int* col, const double* key, const int* isint, const int* nbs, const double* mom,
                  int N, double R, const double* e1, int rk, const double* X0, const double* Wq, int nx, const int* xrows, int labelled, double datum,
                  int cond_fit, double* Wqn_o, double* Xg_o, int* fixed, double* amp_o) {
    auto* k = static_cast<mk::Kernel*>(h);
    try {
        const mk::Proj& P = g_proj.at(h);
        int nn = P.nn;
        auto base = entries(nent, col, key, isint, nbs, mom, k->c.K);
        auto e = cvec(e1, N);
        std::vector<std::vector<cd>> W;
        for (int q = 0; q < rk; q++) W.push_back(cvec(Wq + size_t(q) * 2 * N * nn, size_t(N) * nn));
        auto ev = make_eval(*k, P, base, g, top, std::vector<int>(xrows, xrows + nx), e, R);
        std::vector<std::vector<cd>> Wn;
        std::vector<cd> Xg;
        bool hf = false;
        cd amp(0.0, 0.0);
        mk::dt::offl_start(cvec(X0, size_t(N) * nn), W, nn, nn, e, R, labelled != 0, datum, cond_fit != 0, ev, Wn, Xg, hf, amp);
        for (int q = 0; q < rk; q++) putv(Wn[q], Wqn_o + size_t(q) * 2 * N * nn);
        putv(Xg, Xg_o);
        *fixed = hf;
        amp_o[0] = amp.real(); amp_o[1] = amp.imag();
        k->err.clear();
        return 0;
    } catch (const std::exception& ex) {
        k->err = ex.what();
        return 1;
    }
}

}  // extern "C"

// ---- M2d: the nonlinear steps (mk_steps.hpp); the rows callbacks built here ----
namespace {
mk::st::Rows make_rows(mk::Kernel& k, const mk::Proj& P, const std::vector<mk::Entry>& base, double g, double top, const std::vector<cd>& e1,
                       double R) {
    mk::st::Rows r;
    r.gal = [&k, &P, base, g, top, e1, R](const std::vector<cd>& X) {
        k.run(mk::with_grade(k.c, base, X, g, P.kofs, e1, R), nullptr, top);
        return mk::project(k.c, k.out, g, 1, P.gfam, P.dE, P.W, false);
    };
    r.stored = [&k, &P, base, g, top, e1, R](const std::vector<cd>& X) {
        k.run(mk::with_grade(k.c, base, X, g, P.kofs, e1, R), nullptr, top);
        return mk::project(k.c, k.out, g, 1, P.cfam, P.cdE, P.Wc, true);
    };
    return r;
}
}  // namespace

extern "C" {

// the defect correction at centre g about `base` (the series stripped at g).  isint: the integer loop (ndc = --defect; act; Smax; the data fix
// when dfix: At (2nn x rk), wt, tg0), else the off-integer loop (ndc = max(--defect, 4)).  Minv: N x nn x nn; Wqn: rk x N x nn (normalised);
// outputs X (N x nn), hist (cap 64), drift, rows_after (integer), cond (K x nc, off-integer)
int mk_defect(void* h, double g, double top, int nent, const int* col, const double* key, const int* isint_, const int* nbs, const double* mom,
              int N, double R, const double* e1, int isint, int rk, const double* Minv, const double* Wqn, const double* X0, const int* act,
              double Smax, int ndc, int robust, int dfix, const double* At, const double* wt, const double* tg0, double* X_o, double* hist_o,
              int* nhist, double* drift, double* rows_after, double* cond_o) {
    auto* k = static_cast<mk::Kernel*>(h);
    try {
        const mk::Proj& P = g_proj.at(h);
        int nn = P.nn, K = k->c.K;
        auto base = entries(nent, col, key, isint_, nbs, mom, K);
        auto e = cvec(e1, N);
        std::vector<std::vector<cd>> Mi, W;
        for (int q = 0; q < N; q++) Mi.push_back(cvec(Minv + size_t(q) * 2 * nn * nn, size_t(nn) * nn));
        for (int q = 0; q < rk; q++) W.push_back(cvec(Wqn + size_t(q) * 2 * N * nn, size_t(N) * nn));
        auto rows = make_rows(*k, P, base, g, top, e, R);
        mk::st::DefectOut o;
        if (isint) {
            std::function<std::vector<cd>(const std::vector<cd>&)> fix;
            mk::dt::DataIn in;
            mk::dt::DataOut d;
            if (dfix && rk) {
                in.nn = nn; in.NB = nn; in.rk = rk; in.R = R; in.e1 = e;
                in.wt.assign(wt, wt + 2 * nn);
                in.tg0 = cvec(tg0, 2 * nn);
                d.At = cvec(At, size_t(2) * nn * rk);
                d.Wq = W;
                fix = [in, d](const std::vector<cd>& X) { return mk::dt::data_fix(X, in, d); };
            }
            mk::st::defect_int(cvec(X0, size_t(N) * nn), Mi, W, std::vector<char>(act, act + nn), Smax, K, nn, e, R, ndc, robust != 0, fix, rows, o);
        } else {
            mk::st::defect_off(cvec(X0, size_t(N) * nn), Mi, W, K, nn, e, R, ndc, robust != 0, rows, o);
            putv(o.conditions, cond_o);
        }
        putv(o.X, X_o);
        *nhist = int(std::min<size_t>(o.hist.size(), 64));
        for (int i = 0; i < *nhist; i++) hist_o[i] = o.hist[i];
        *drift = o.drift;
        *rows_after = o.rows_after;
        k->err.clear();
        return 0;
    } catch (const std::exception& ex) {
        k->err = ex.what();
        return 1;
    }
}

// Newton's sources at the nodes about X_k (base: the series stripped at g; Ms: the re-linearised operator, 3 x nn x nn)
int mk_newton_sources(void* h, double g, double top, int nent, const int* col, const double* key, const int* isint_, const int* nbs, const double* mom,
                      int N, double R, const double* e1, const double* Ms, const double* Xk, double* Sn_o) {
    auto* k = static_cast<mk::Kernel*>(h);
    try {
        const mk::Proj& P = g_proj.at(h);
        int nn = P.nn;
        auto base = entries(nent, col, key, isint_, nbs, mom, k->c.K);
        auto e = cvec(e1, N);
        auto rows = make_rows(*k, P, base, g, top, e, R);
        putv(mk::st::newton_sources(cvec(Xk, size_t(N) * nn), cvec(Ms, size_t(3) * nn * nn), k->c.K, nn, e, R, rows), Sn_o);
        k->err.clear();
        return 0;
    } catch (const std::exception& ex) {
        k->err = ex.what();
        return 1;
    }
}

}  // extern "C"

// ---- M2e: the families and the registry (mk_families.hpp) ----
extern "C" {

// the centres the rows of `base` (the series stripped at g) announce above g: ann (key, magnitude; cap 1024), the keys added to the queue
int mk_announce(void* h, double g, double top, double gmax, int nent, const int* col, const double* key, const int* isint, const int* nbs,
                const double* mom, int* nann, double* ann, int* nadd, double* add_v, int* add_int) {
    auto* k = static_cast<mk::Kernel*>(h);
    try {
        const mk::Proj& P = g_proj.at(h);
        k->run(entries(nent, col, key, isint, nbs, mom, k->c.K), nullptr, top);
        std::vector<mk::fm::Ann> a;
        std::vector<mk::Key> ad;
        mk::fm::announce(k->c, k->out, P, g, gmax, a, ad);
        *nann = int(std::min<size_t>(a.size(), 1024));
        for (int i = 0; i < *nann; i++) { ann[2 * i] = a[i].key; ann[2 * i + 1] = a[i].mag; }
        *nadd = int(std::min<size_t>(ad.size(), 4096));
        for (int i = 0; i < *nadd; i++) { add_v[i] = ad[i].v; add_int[i] = ad[i].isint; }
        k->err.clear();
        return 0;
    } catch (const std::exception& ex) {
        k->err = ex.what();
        return 1;
    }
}

// the lapse families from the bordered pencil at aligned g (Ms: 3 x NB x NB; bord: nbord (row, column) pairs, full indices)
int mk_lapse_roots(void* h, int NB, const double* Ms, const double* rw, const double* cw, int nbord, const int* bord, double g, double gmax, int lmax,
                   double* eig, int* nev, double* evr, int* nevr, int* nl, int* lapse_l, double* lapse_key, int* nreg, int* reg_l, double* reg_guess) {
    auto* k = static_cast<mk::Kernel*>(h);
    try {
        std::vector<std::pair<int, int>> b;
        for (int q = 0; q < nbord; q++) b.emplace_back(bord[2 * q], bord[2 * q + 1]);
        mk::fm::Lapse o;
        mk::fm::lapse_roots(k->c, cvec(Ms, size_t(3) * NB * NB), NB, std::vector<double>(rw, rw + NB), std::vector<double>(cw, cw + NB), b, g, k->c.n,
                            gmax, lmax, o);
        *nev = int(o.eig_all.size());
        putv(o.eig_all, eig);
        *nevr = int(o.evr.size());
        for (int i = 0; i < *nevr; i++) evr[i] = o.evr[i];
        *nl = int(o.lapse.size());
        for (int i = 0; i < *nl; i++) { lapse_l[i] = o.lapse[i].first; lapse_key[i] = o.lapse[i].second.v; }
        *nreg = int(o.registry.size());
        for (int i = 0; i < *nreg; i++) { reg_l[i] = o.registry[i].first; reg_guess[i] = o.registry[i].second; }
        k->err.clear();
        return 0;
    } catch (const std::exception& ex) {
        k->err = ex.what();
        return 1;
    }
}

int mk_towers(void* h, int nf, const double* fv, double gmax, double* out_v, int* out_int, int cap) {
    auto* k = static_cast<mk::Kernel*>(h);
    std::vector<mk::Key> f;
    for (int i = 0; i < nf; i++) f.push_back(mk::Key{fv[i], false});
    auto t = mk::fm::towers(k->c, f, gmax);
    int n = int(std::min<size_t>(t.size(), size_t(cap)));
    for (int i = 0; i < n; i++) { out_v[i] = t[i].v; out_int[i] = t[i].isint; }
    return int(t.size());
}

// the pending queue (the driver's; exposed for the unit test)
void* mk_q_new() { return new mk::fm::Queue(); }
void mk_q_free(void* q) { delete static_cast<mk::fm::Queue*>(q); }
void mk_q_push(void* q, int n, const double* v, const int* isint) {
    std::vector<mk::Key> ks;
    for (int i = 0; i < n; i++) ks.push_back(mk::Key{v[i], isint[i] != 0});
    static_cast<mk::fm::Queue*>(q)->push(ks);
}
int mk_q_pop(void* q, double* v, int* isint) {
    mk::Key g;
    if (!static_cast<mk::fm::Queue*>(q)->pop(g)) return 0;
    *v = g.v;
    *isint = g.isint;
    return 1;
}
int mk_q_after(void* q, double* out, int cap) {
    auto a = static_cast<mk::fm::Queue*>(q)->after();
    for (int i = 0; i < int(a.size()) && i < cap; i++) out[i] = a[i];
    return int(a.size());
}

// the gauge Lie flow of six fields given as mk_rows_fields takes them; the result in the kernel's output slots (mk_out_info / mk_out_get, deriv 0)
int mk_lie_flow(void* h, const int* np, const double* keys, const int* isint, const int* nb, const int* has_e, const int* e, const double* top,
                const double* data, int ncs, const int* ls, const double* cs, double J, const double* TH, int steps) {
    auto* k = static_cast<mk::Kernel*>(h);
    try {
        size_t ip = 0, off = 0, P = k->c.psz();
        mk::Ser F[6];
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
            F[f] = std::move(s);
        }
        std::vector<std::pair<int, double>> c;
        for (int i = 0; i < ncs; i++) c.emplace_back(ls[i], cs[i]);
        mk::fm::lie_flow(k->c, F, c, J, std::vector<double>(TH, TH + k->c.NF), steps);
        for (int f = 0; f < 6; f++) k->out[f] = F[f];
        k->err.clear();
        return 0;
    } catch (const std::exception& ex) {
        k->err = ex.what();
        return 1;
    }
}

}  // extern "C"

// ---- M2f: the driver (mk_engine.hpp) ----
#include "mk_engine.hpp"
extern "C" {
// the whole recursion from a setup bundle; the decision log, the final series and the flowed fields written to OUT
int mk_run_bundle(const char* dir, const char* out, char* err, int errcap) {
    try {
        mk::Kernel k;
        mk::Proj P;
        mk::en::Setup S;
        mk::en::Cfg c;
        mk::en::load_engine(dir, k, P, S, c);
        mk::en::Engine E(k, P, S, c, out);
        E.run();
        return 0;
    } catch (const std::exception& ex) {
        std::snprintf(err, size_t(errcap), "%s", ex.what());
        return 1;
    }
}
// M3: the forward run, then the tangents.  opts: 'key=value;...' -- which (';'-list as n2a_nl's --tangent, '|'-separated here), tol, passes,
// ctol, iters, h, margin, threads, top_local, no_stall, mode (fd / jvp)
// G3 (i): the forward, then the matching keys' exact n-derivative on the final series against the Richardson differences with n moved by +-h,
// +-2h and the series frozen -> OUT/mk_dn.json (per key: value, exact, D(h), D(2h)); opts 'h=VALUE' (default 1e-5)
int mk_mk_dn(const char* dir, const char* out, const char* opts, char* err, int errcap) {
    try {
        mk::Kernel k;
        mk::Proj P;
        mk::en::Setup S;
        mk::en::Cfg c;
        mk::en::load_engine(dir, k, P, S, c);
        double h = 1e-5;
        std::string o(opts ? opts : "");
        if (o.rfind("h=", 0) == 0) h = std::stod(o.substr(2));
        mk::en::Engine E(k, P, S, c, out);
        E.run();
        E.mk_dn_check(h);
        return 0;
    } catch (const std::exception& ex) {
        std::snprintf(err, size_t(errcap), "%s", ex.what());
        return 1;
    }
}

int mk_run_bundle_tan(const char* dir, const char* out, const char* opts, char* err, int errcap) {
    try {
        mk::Kernel k;
        mk::Proj P;
        mk::en::Setup S;
        mk::en::Cfg c;
        mk::en::load_engine(dir, k, P, S, c);
        mk::en::TanOpt T;
        T.on = true;
        std::string o(opts);
        size_t a = 0;
        while (a < o.size()) {
            size_t b = o.find(';', a);
            if (b == std::string::npos) b = o.size();
            std::string it = o.substr(a, b - a);
            size_t e = it.find('=');
            if (e != std::string::npos) {
                std::string kk = it.substr(0, e), v = it.substr(e + 1);
                if (kk == "which") { for (auto& ch : v) if (ch == '|') ch = ';'; T.which = v; }
                else if (kk == "tol") T.tol = std::stod(v);
                else if (kk == "passes") T.passes = std::stoi(v);
                else if (kk == "ctol") T.ctol = std::stod(v);
                else if (kk == "iters") T.iters = std::stoi(v);
                else if (kk == "h") T.h = std::stod(v);
                else if (kk == "margin") T.margin = std::stoi(v);
                else if (kk == "threads") T.threads = std::stoi(v);
                else if (kk == "top_local") T.top_local = v == "1";
                else if (kk == "no_stall") T.no_stall = v == "1";
                else if (kk == "mode") T.jvp = v == "jvp";
                else if (kk == "direct") T.direct = std::stoi(v);
                else if (kk == "direct_chunk") T.direct_chunk = std::stoi(v);
                else throw std::runtime_error("mk: unknown tangent option " + kk);
            }
            a = b + 1;
        }
        mk::en::Engine E(k, P, S, c, out);
        E.run();
        E.tangents(T);
        return 0;
    } catch (const std::exception& ex) {
        std::snprintf(err, size_t(errcap), "%s", ex.what());
        return 1;
    }
}
}  // extern "C"

// ---- M3.5: the operator about the cylinder from the kernel's linearisation (mk_opbuild.hpp) ----
#include "mk_opbuild.hpp"
extern "C" {
// opts 'key=value;...': e0 (1.3137), rows (a file of tall row labels fam@P_D; default: every degree of the family's parity up to the largest
// column l + 14, kept above rowtol (1e-10) of the family's largest entry; H_c, L_K on E_sigma's degrees), lkdiv (exact | rowwise), edge (12), chunk (34), ng (200);
// levels_out=PATH / levels_in=PATH (M4e: the structure frozen per J step) -- the build's discrete decisions written / reused: the tall rows
// (PATH.rows, as 'rows'), the square, condition and gate rows' leading levels and the gate rows' liveness (PATH)
// G1 (NOTES_M4_DESIGN.md section 19): the operator's derivative with respect to the cylinder parameters (n3b names without 'cyl ': uJ, P_shapeJ,
// qJ, QJ, bJ, btK, c, n, K) at the bundle DIR's cylinder, on the frozen rows of opts 'levels_in=FILE' (its FILE.rows) -- the layout of
// mk_op_build's A / B / C.  Writes OUT/dA.npy, dB.npy, dC.npy (ntheta, np, nr, nn), OUT/thetas.txt and, for the check against mk_op_build,
// OUT/A2.npy, B2.npy, C2.npy (the second-order pass's own M).  opts: levels_in (required), thetas=NAME,NAME,..., e0, chunk, ng
int mk_op_build_dtheta(const char* dir, const char* out, const char* opts, char* err, int errcap) {
    try {
        using namespace mk;
        Kernel k;
        Proj P;
        en::Setup S;
        en::Cfg cf;
        en::load_engine(dir, k, P, S, cf);
        Bundle Bd;
        {
            std::ifstream f(std::string(dir) + "/config.txt");
            std::string line;
            while (std::getline(f, line)) {
                size_t sp = line.find(' ');
                if (sp != std::string::npos) Bd.cfg[line.substr(0, sp)] = line.substr(sp + 1);
            }
        }
        double e0 = 1.3137;
        int chunk = 34, ng = 200;
        std::string lvin, ths;
        std::string o(opts ? opts : "");
        size_t a = 0;
        while (a < o.size()) {
            size_t b = o.find(';', a);
            if (b == std::string::npos) b = o.size();
            std::string it = o.substr(a, b - a);
            size_t e = it.find('=');
            if (e != std::string::npos) {
                std::string kk = it.substr(0, e), v = it.substr(e + 1);
                if (kk == "e0") e0 = std::stod(v);
                else if (kk == "chunk") chunk = std::stoi(v);
                else if (kk == "ng") ng = std::stoi(v);
                else if (kk == "levels_in") lvin = v;
                else if (kk == "thetas") ths = v;
                else throw std::runtime_error("mk: unknown op-build-dtheta option " + kk);
            }
            a = b + 1;
        }
        if (lvin.empty()) throw std::runtime_error("mk: op-build-dtheta needs levels_in (the J step's frozen rows)");
        std::vector<int> rf, rD;
        {
            std::ifstream f(lvin + ".rows");
            if (!f) throw std::runtime_error("mk: op-build-dtheta: cannot read " + lvin + ".rows");
            std::string l;
            while (std::getline(f, l)) {
                if (l.empty()) continue;
                size_t at = l.find("@P");
                int fi = ob::rowfam(l.substr(0, at));
                if (fi < 0) continue;
                rf.push_back(fi);
                rD.push_back(std::stoi(l.substr(at + 2)));
            }
        }
        std::vector<std::string> names;
        {
            size_t p0 = 0;
            while (p0 < ths.size()) {
                size_t p1 = ths.find(',', p0);
                if (p1 == std::string::npos) p1 = ths.size();
                if (p1 > p0) names.push_back(ths.substr(p0, p1 - p0));
                p0 = p1 + 1;
            }
        }
        if (names.empty()) throw std::runtime_error("mk: op-build-dtheta: no thetas");
        int nn = int(S.collab.size());
        ob::Quad Q;
        Q.init(k.c.NF, ng);
        std::vector<int> PL = {-2, -1, 0, 1, 2};
        double top = Bd.getd("TOP", 9.0);
        const auto& th = S.TH;
        size_t NF = th.size();
        auto grid = [&](auto f) { std::vector<double> v(NF); for (size_t g = 0; g < NF; g++) v[g] = f(th[g]); return v; };
        std::vector<double> dA, dB, dC, A2, B2, C2;
        for (size_t t = 0; t < names.size(); t++) {
            const std::string& key = names[t];
            Kernel::DCyl d;
            if (key == "n") d.dn = 1.0;
            else if (key == "K") d.dK = 1.0;
            else if (key == "c") { d.b = grid([](double) { return 1.0; }); d.dbnorm = 1.0; }
            else {
                size_t dg = key.find_first_of("0123456789");
                if (dg == std::string::npos) throw std::runtime_error("mk: op-build-dtheta: theta " + key);
                std::string fam = key.substr(0, dg);
                double j = std::stod(key.substr(dg));
                if (fam == "u") d.U = grid([&](double x) { return std::cos(2.0 * j * x); });
                else if (fam == "P_shape") d.P = grid([&](double x) { return std::cos(2.0 * j * x); });
                else if (fam == "q") d.q = grid([&](double x) { double s_ = std::sin(x); return s_ * s_ * std::cos(2.0 * j * x); });
                else if (fam == "Q") d.Qb = grid([&](double x) { double s_ = std::sin(x); return s_ * s_ * s_ * s_ * std::cos((2.0 * j + 1.0) * x); });
                else if (fam == "b") d.b = grid([&](double x) { return std::cos(2.0 * j * x); });
                else if (fam == "bt") d.bt = grid([&](double x) { return std::sin(2.0 * (j + 1.0) * x); });
                else throw std::runtime_error("mk: op-build-dtheta: theta " + key);
            }
            auto o2 = ob::build2(k, nn, rf, rD, PL, e0, top, chunk, Q, d);
            dA.insert(dA.end(), o2.dA.begin(), o2.dA.end());
            dB.insert(dB.end(), o2.dB.begin(), o2.dB.end());
            dC.insert(dC.end(), o2.dC.begin(), o2.dC.end());
            if (t == 0) { A2 = o2.A; B2 = o2.B; C2 = o2.C; }
        }
        std::string od(out);
        std::filesystem::create_directories(od);
        long nth = long(names.size()), np_ = long(PL.size()), nr = long(rf.size());
        en::npy_write(od + "/dA.npy", dA.data(), "<f8", {nth, np_, nr, long(nn)}, 8);
        en::npy_write(od + "/dB.npy", dB.data(), "<f8", {nth, np_, nr, long(nn)}, 8);
        en::npy_write(od + "/dC.npy", dC.data(), "<f8", {nth, np_, nr, long(nn)}, 8);
        en::npy_write(od + "/A2.npy", A2.data(), "<f8", {np_, nr, long(nn)}, 8);
        en::npy_write(od + "/B2.npy", B2.data(), "<f8", {np_, nr, long(nn)}, 8);
        en::npy_write(od + "/C2.npy", C2.data(), "<f8", {np_, nr, long(nn)}, 8);
        {
            // G3: the square rows' projection (mk_op_build's 'tri' pairing: column (cl, l) on its family's rows, test function P_l -- QF P_l, BT
            // P_(l+1), QB (1 - z^2)^2 P_l -- int P_D T on 200 Gauss nodes) and kE (s = n (sigma + kE)): the families' roots are the level-0
            // square pencil's
            std::vector<std::string> cl(nn);
            std::vector<int> lc(nn);
            for (int j = 0; j < nn; j++) {
                size_t p = S.collab[j].find_first_of("0123456789");
                cl[j] = S.collab[j].substr(0, p);
                lc[j] = std::stoi(S.collab[j].substr(p));
            }
            std::vector<double> gx, gw;
            ob::gauss_legendre(200, gx, gw);
            int Dmx = 0;
            for (int r = 0; r < nr; r++) Dmx = std::max(Dmx, rD[r]);
            for (int j = 0; j < nn; j++) Dmx = std::max(Dmx, lc[j] + 1);
            std::vector<std::vector<double>> PLg(200);
            for (int g = 0; g < 200; g++) ob::legendre_upto(Dmx, gx[g], PLg[g]);
            static const char* cfm[6] = {"QF", "PS", "PH", "BR", "BT", "QB"};
            std::vector<double> Psq(size_t(nn) * nr, 0.0);
            std::vector<int> kE(nn);
            for (int j = 0; j < nn; j++) {
                int fi = -1;
                for (int i = 0; i < 6; i++)
                    if (cl[j] == cfm[i]) fi = i;
                int lt = cl[j] == "BT" ? lc[j] + 1 : lc[j];
                for (int r = 0; r < nr; r++) {
                    if (rf[r] != fi) continue;
                    double s_ = 0.0;
                    for (int g = 0; g < 200; g++) {
                        double T = PLg[g][lt];
                        if (cl[j] == "QB") T *= (1 - gx[g] * gx[g]) * (1 - gx[g] * gx[g]);
                        s_ += gw[g] * PLg[g][rD[r]] * T;
                    }
                    Psq[size_t(j) * nr + r] = s_;
                }
                kE[j] = (cl[j] == "QF" || cl[j] == "PH" || (cl[j] == "BR" && lc[j] == 0)) ? -1 : 0;
            }
            en::npy_write(od + "/Psq.npy", Psq.data(), "<f8", {long(nn), nr}, 8);
            en::npy_write(od + "/kE_sq.npy", kE.data(), "<i4", {long(nn)}, 4);
            double nv = k.c.n;
            en::npy_write(od + "/n.npy", &nv, "<f8", {1}, 8);
        }
        std::ofstream f(od + "/thetas.txt");
        for (auto& nm : names) f << nm << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::snprintf(err, size_t(errcap), "%s", ex.what());
        return 1;
    }
}

int mk_op_build(const char* dir, const char* out, const char* opts, char* err, int errcap) {
    try {
        using namespace mk;
        Kernel k;
        Proj P;
        en::Setup S;
        en::Cfg cf;
        en::load_engine(dir, k, P, S, cf);
        Bundle Bd;
        {
            std::ifstream f(std::string(dir) + "/config.txt");
            std::string line;
            while (std::getline(f, line)) {
                size_t sp = line.find(' ');
                if (sp != std::string::npos) Bd.cfg[line.substr(0, sp)] = line.substr(sp + 1);
            }
        }
        double e0 = 1.3137;
        std::string rowsf, lkdiv = "exact", lvin, lvout;
        int edge = 12, chunk = 34, ng = 200;
        double rowtol = 1e-10;
        std::string o(opts ? opts : "");
        size_t a = 0;
        while (a < o.size()) {
            size_t b = o.find(';', a);
            if (b == std::string::npos) b = o.size();
            std::string it = o.substr(a, b - a);
            size_t e = it.find('=');
            if (e != std::string::npos) {
                std::string kk = it.substr(0, e), v = it.substr(e + 1);
                if (kk == "e0") e0 = std::stod(v);
                else if (kk == "rows") rowsf = v;
                else if (kk == "lkdiv") lkdiv = v;
                else if (kk == "edge") edge = std::stoi(v);
                else if (kk == "chunk") chunk = std::stoi(v);
                else if (kk == "ng") ng = std::stoi(v);
                else if (kk == "rowtol") rowtol = std::stod(v);
                else if (kk == "levels_in") lvin = v;
                else if (kk == "levels_out") lvout = v;
                else throw std::runtime_error("mk: unknown op-build option " + kk);
            }
            a = b + 1;
        }
        if (lkdiv != "exact" && lkdiv != "rowwise") throw std::runtime_error("mk: lkdiv exact | rowwise");
        // the frozen structure (levels_in): its rows as 'rows', its levels and liveness below
        std::map<std::string, std::vector<int>> LV;
        if (!lvin.empty()) {
            if (rowsf.empty()) rowsf = lvin + ".rows";
            std::ifstream f(lvin);
            if (!f) throw std::runtime_error("mk: op-build: cannot read " + lvin);
            std::string key;
            size_t cnt;
            while (f >> key >> cnt) {
                std::vector<int> v(cnt);
                for (auto& x : v) f >> x;
                LV[key] = v;
            }
            for (const char* key_ : {"D0", "DC", "DG", "LIVE"})
                if (!LV.count(key_)) throw std::runtime_error(std::string("mk: op-build: ") + lvin + " without " + key_);
        }
        auto fix = [&](const char* key_) { return lvin.empty() ? nullptr : &LV[key_]; };
        int nn = int(S.collab.size());
        std::vector<std::string> cl(nn);
        std::vector<int> lc(nn);
        for (int j = 0; j < nn; j++) {
            size_t p = S.collab[j].find_first_of("0123456789");
            cl[j] = S.collab[j].substr(0, p);
            lc[j] = std::stoi(S.collab[j].substr(p));
        }
        int lmaxc = *std::max_element(lc.begin(), lc.end());
        ob::Quad Q;
        Q.init(k.c.NF, ng);
        std::vector<int> PL = {-2, -1, 0, 1, 2};
        double top = Bd.getd("TOP", 9.0);
        std::vector<int> rf, rD;
        bool rule = rowsf.empty();
        if (!rule) {
            std::ifstream f(rowsf);
            std::string l;
            while (std::getline(f, l)) {
                if (l.empty()) continue;
                size_t at = l.find("@P");
                int fi = ob::rowfam(l.substr(0, at));
                if (fi < 0) continue;                          // M (maximality) has no kernel row
                rf.push_back(fi);
                rD.push_back(std::stoi(l.substr(at + 2)));
            }
        } else {
            for (int fi = 0; fi < 6; fi++)
                for (int D = (fi >= 4 ? 1 : 0); D <= lmaxc + 14; D += 2) { rf.push_back(fi); rD.push_back(D); }
        }
        auto op = ob::build(k, nn, rf, rD, PL, e0, top, chunk, lkdiv == "rowwise", Q);
        if (rule) {
            // the builder's rule: degrees above rowtol (1e-10) of the family's largest entry; H_c and L_K on E_sigma's degrees (n2a_lk forms
            // them on every degree of E_q, E_sigma, E_Phi -- not by their own magnitudes)
            std::vector<double> fmax(6, 0.0), rmax(rf.size(), 0.0);
            for (size_t r = 0; r < rf.size(); r++)
                for (size_t ip = 0; ip < PL.size(); ip++)
                    for (int j = 0; j < nn; j++) {
                        size_t q = op.at(int(ip), int(r), j);
                        rmax[r] = std::max({rmax[r], std::abs(op.A[q]), std::abs(op.B[q]), std::abs(op.C[q])});
                    }
            for (size_t r = 0; r < rf.size(); r++) fmax[rf[r]] = std::max(fmax[rf[r]], rmax[r]);
            ob::Op o2;
            o2.PL = PL; o2.nn = nn; o2.imag = op.imag; o2.bg_ephi = op.bg_ephi;
            std::vector<size_t> keep;
            std::vector<char> sig(rf.size(), 0);
            for (size_t r = 0; r < rf.size(); r++) sig[r] = rmax[r] > rowtol * fmax[rf[r]];
            for (size_t r = 0; r < rf.size(); r++) {
                bool k_ = sig[r];
                if (rf[r] == 0 || rf[r] == 2)
                    for (size_t r2 = 0; r2 < rf.size(); r2++)
                        if (rf[r2] == 1 && rD[r2] == rD[r]) k_ = sig[r2];
                if (k_) { keep.push_back(r); o2.rf.push_back(rf[r]); o2.rD.push_back(rD[r]); }
            }
            size_t sz = PL.size() * keep.size() * size_t(nn);
            o2.A.assign(sz, 0.0); o2.B = o2.A; o2.C = o2.A;
            for (size_t ip = 0; ip < PL.size(); ip++)
                for (size_t r = 0; r < keep.size(); r++)
                    for (int j = 0; j < nn; j++) {
                        o2.A[o2.at(int(ip), int(r), j)] = op.A[op.at(int(ip), int(keep[r]), j)];
                        o2.B[o2.at(int(ip), int(r), j)] = op.B[op.at(int(ip), int(keep[r]), j)];
                        o2.C[o2.at(int(ip), int(r), j)] = op.C[op.at(int(ip), int(keep[r]), j)];
                    }
            op = std::move(o2);
        }
        int nr = int(op.rf.size());
        auto rowidx = [&](int fi, int D) {
            for (int r = 0; r < nr; r++)
                if (op.rf[r] == fi && op.rD[r] == D) return r;
            return -1;
        };
        // the square rows: n2a_rec.build_ops 'tri' -- column (cl, l) paired with its family's rows (PS E_sigma, PH L_K, BR E_sh^r, QF H_c,
        // BT E_sh^th, QB E_Q), test function P_l (QF: P_l, BT: P_(l+1), QB: (1 - z^2)^2 P_l), int P_D T on 200 Gauss nodes
        std::vector<double> gx, gw;
        ob::gauss_legendre(200, gx, gw);
        int Dmx = 0;
        for (int r = 0; r < nr; r++) Dmx = std::max(Dmx, op.rD[r]);
        Dmx = std::max(Dmx, lmaxc + 1);
        std::vector<std::vector<double>> PLg(200);
        for (int g = 0; g < 200; g++) ob::legendre_upto(Dmx, gx[g], PLg[g]);
        auto pairfam = [](const std::string& c) {
            static const char* cf[6] = {"QF", "PS", "PH", "BR", "BT", "QB"};
            static const int rf[6] = {0, 1, 2, 3, 4, 5};             // QF H_c, PS E_sigma, PH L_K, BR E_sh^r, BT E_sh^th, QB E_Q
            for (int i = 0; i < 6; i++)
                if (c == cf[i]) return rf[i];
            return -1;
        };
        std::vector<std::vector<double>> Psq(nn, std::vector<double>(nr, 0.0));
        std::vector<int> kE(nn);
        for (int j = 0; j < nn; j++) {
            int fi = pairfam(cl[j]);
            int lt = cl[j] == "BT" ? lc[j] + 1 : lc[j];
            for (int r = 0; r < nr; r++) {
                if (op.rf[r] != fi) continue;
                double s = 0.0;
                for (int g = 0; g < 200; g++) {
                    double T = PLg[g][lt];
                    if (cl[j] == "QB") T *= (1 - gx[g] * gx[g]) * (1 - gx[g] * gx[g]);
                    s += gw[g] * PLg[g][op.rD[r]] * T;
                }
                Psq[j][r] = s;
            }
            kE[j] = (cl[j] == "QF" || cl[j] == "PH" || (cl[j] == "BR" && lc[j] == 0)) ? -1 : 0;
        }
        auto sq = ob::select(op, Psq);
        auto d0 = ob::derive(sq, PL, kE, k.c.n, nullptr, fix("D0"));
        std::vector<int> dEu = d0.dE;
        bool dEbt0 = Bd.cfg.count("dEbt0") && (Bd.cfg["dEbt0"] == "True" || Bd.cfg["dEbt0"] == "1" || Bd.cfg["dEbt0"] == "true");
        if (dEbt0)
            for (int j = 0; j < nn; j++)
                if (cl[j] == "BT") dEu[j] = 0;
        auto dsq = ob::derive(sq, PL, kE, k.c.n, &dEu, fix("D0"));
        // the condition rows H_c@P0, E_sh^th@P1; the gate rows (the engine's families, degree <= edge, E_sh^th <= edge + 1)
        auto unit = [&](const std::vector<int>& rs) {
            std::vector<std::vector<double>> U(rs.size(), std::vector<double>(nr, 0.0));
            for (size_t i = 0; i < rs.size(); i++) U[i][rs[i]] = 1.0;
            return U;
        };
        std::vector<int> crow = {rowidx(0, 0), rowidx(4, 1)};
        if (crow[0] < 0 || crow[1] < 0) throw std::runtime_error("mk: op-build without the condition rows");
        auto dc = ob::derive(ob::select(op, unit(crow)), PL, kE, k.c.n, nullptr, fix("DC"));
        std::vector<int> grow;
        for (int r = 0; r < nr; r++)
            if (op.rD[r] <= (op.rf[r] == 4 ? edge + 1 : edge)) grow.push_back(r);
        auto gs = ob::select(op, unit(grow));
        auto dg = ob::derive(gs, PL, kE, k.c.n, nullptr, fix("DG"));
        std::vector<int> live(grow.size(), 0);
        {
            std::vector<double> rmx(grow.size(), 0.0), fm(6, 0.0);
            for (size_t i = 0; i < grow.size(); i++) {
                for (size_t ip = 0; ip < PL.size(); ip++)
                    for (int j = 0; j < nn; j++) {
                        size_t q = (ip * grow.size() + i) * nn + j;
                        rmx[i] = std::max({rmx[i], std::abs(gs.A[q]), std::abs(gs.B[q]), std::abs(gs.C[q])});
                    }
                fm[op.rf[grow[i]]] = std::max(fm[op.rf[grow[i]]], rmx[i]);
            }
            for (size_t i = 0; i < grow.size(); i++) live[i] = rmx[i] > 1e-8 * fm[op.rf[grow[i]]] && dg.dE[i] < 99;
            if (!lvin.empty()) {
                if (LV["LIVE"].size() != live.size()) throw std::runtime_error("mk: op-build: frozen liveness for a different gate-row set");
                live = LV["LIVE"];
            }
        }
        // output
        std::string od(out);
        mkdir(od.c_str(), 0755);
        long np_ = long(PL.size());
        en::npy_write(od + "/A.npy", op.A.data(), "<f8", {np_, long(nr), long(nn)}, 8);
        en::npy_write(od + "/B.npy", op.B.data(), "<f8", {np_, long(nr), long(nn)}, 8);
        en::npy_write(od + "/C.npy", op.C.data(), "<f8", {np_, long(nr), long(nn)}, 8);
        en::npy_write(od + "/PL.npy", PL.data(), "<i4", {np_}, 4);
        auto wi = [&](const std::string& nm, const std::vector<int>& v) { en::npy_write(od + "/" + nm + ".npy", v.data(), "<i4", {long(v.size())}, 4); };
        auto wd = [&](const std::string& nm, const std::vector<double>& v) { en::npy_write(od + "/" + nm + ".npy", v.data(), "<f8", {long(v.size())}, 8); };
        wi("dE", dEu); wd("rw", dsq.rw); wd("cw", dsq.cw); wd("rw_out", dsq.rw_out); wi("kE", kE);
        wi("cdE", dc.dE); wd("rwc", dc.rw); wi("gt_dE", dg.dE); wi("live", live);
        std::ofstream rl(od + "/rlab.txt");
        for (int r = 0; r < nr; r++) rl << ob::ROWS[op.rf[r]] << "@P" << op.rD[r] << "\n";
        std::ofstream gl(od + "/glab.txt");
        for (int r : grow) gl << ob::ROWS[op.rf[r]] << "@P" << op.rD[r] << "\n";
        if (!lvout.empty()) {                                           // the structure for the J step's later builds
            std::ofstream lf(lvout);
            auto wv = [&](const char* key_, const std::vector<int>& v) { lf << key_ << " " << v.size(); for (int x : v) lf << " " << x; lf << "\n"; };
            wv("D0", d0.dE); wv("DC", dc.dE); wv("DG", dg.dE); wv("LIVE", live);
            std::ofstream rf(lvout + ".rows");
            for (int r = 0; r < nr; r++) rf << ob::ROWS[op.rf[r]] << "@P" << op.rD[r] << "\n";
        }
        std::ofstream inf(od + "/info.txt");
        inf.precision(17);
        inf << "e0 " << e0 << "\nn " << k.c.n << "\nlkdiv " << lkdiv << "\nimag " << op.imag << "\nrows " << nr << "\ncols " << nn
            << "\nbg_ephi " << op.bg_ephi[0] << " " << op.bg_ephi[1] << " " << op.bg_ephi[2] << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::snprintf(err, size_t(errcap), "%s", ex.what());
        return 1;
    }
}
}  // extern "C"

// ---- M4a: the setup bundle at a cylinder (mk_cylinder.hpp) ----
#include "mk_cylinder.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
extern "C" {
// base: a setup bundle of the same set-up (any cylinder); cyl: an n2a_nl cylinder file; K: the lapse normalisation; out: the new bundle;
// opts: mk_op_build's (lkdiv defaults to exact -- the method's convention, research round 1036).  The cylinder's arrays and scalars replaced,
// the operator built natively about the new cylinder (out/opbuild) and its derived quantities copied into the bundle.
int mk_bundle_at(const char* base, const char* cyl, double K, const char* out, const char* opts, char* err, int errcap) {
    try {
        namespace fs = std::filesystem;
        using namespace mk;
        std::string od(out), bd(base);
        fs::create_directories(od);
        for (const auto& e : fs::directory_iterator(bd))
            if (e.is_regular_file()) fs::copy_file(e.path(), fs::path(od) / e.path().filename(), fs::copy_options::overwrite_existing);
        auto C = cy::cylinder_from(cy::read_json_numbers(cyl));
        // M4c: 'perturb=KEY:DELTA' in opts -- one grade-0 parameter moved (n3b_linthroat's names without 'cyl ': uJ, P_shapeJ, qJ, QJ, bJ, btJ, c, n,
        // K; cyl_dict's rule: c moves b_0 with it), for the Jacobian's grade-0 columns; the remaining options go to the operator build
        std::string oo(opts ? opts : "");
        {
            size_t at = oo.find("perturb=");
            if (at != std::string::npos) {
                size_t e = oo.find(';', at);
                std::string it = oo.substr(at + 8, (e == std::string::npos ? oo.size() : e) - at - 8);
                oo.erase(at, (e == std::string::npos ? oo.size() : e + 1) - at);
                std::string key = it.substr(0, it.find(':'));
                double dv = std::stod(it.substr(it.find(':') + 1));
                size_t dg = key.find_first_of("0123456789");
                std::string fam = key.substr(0, dg);
                auto bump = [&](std::vector<double>& v, int j) { if (int(v.size()) <= j) v.resize(size_t(j) + 1, 0.0); v[size_t(j)] += dv; };
                if (key == "c") { C.c += dv; if (!C.b.empty()) C.b[0] = C.c; }
                else if (key == "n") C.n += dv;
                else if (key == "K") K += dv;
                else if (dg == std::string::npos) throw std::runtime_error("mk: bundle-at: perturb " + key);
                else {
                    int j = std::stoi(key.substr(dg));
                    if (fam == "u") bump(C.u, j);
                    else if (fam == "P_shape") bump(C.P, j);
                    else if (fam == "q") bump(C.q, j);
                    else if (fam == "Q") bump(C.Q, j);
                    else if (fam == "b") bump(C.b, j);
                    else if (fam == "bt") bump(C.bt, j);
                    else throw std::runtime_error("mk: bundle-at: perturb " + key);
                }
            }
        }
        auto th = npy_as<double>(npy_read(bd + "/TH.npy"));
        long NF = long(th.size());
        auto vals = cy::cyl_vals(C, th);
        en::npy_write(od + "/cyl_vals.npy", vals.data(), "<f8", {6, NF}, 8);
        // (3, NF): 1 / r, Ph / r, r -- Ph / r's constant piece is K, as the oracle forms it: n2a_nl.C0 passes K through sympy at 15 significant
        // digits (J = 0.10: 1.11164503925951 for K = 1.1116450392595139, 3.4e-15 relative) -- mirrored, so the forward can be gated bitwise
        char k15[40];
        std::snprintf(k15, sizeof k15, "%.15g", K);
        auto c0 = npy_cplx(npy_read(bd + "/cyl_c0.npy"));
        for (long g = 0; g < NF; g++) c0[size_t(NF + g)] = std::complex<double>(std::stod(k15), 0.0);
        en::npy_write(od + "/cyl_c0.npy", c0.data(), "<c16", {3, NF}, 16);
        {                                                               // config.txt: n, bnorm (= c), Ph (= K)
            std::ifstream f(bd + "/config.txt");
            std::ostringstream o;
            o.precision(17);
            std::string line;
            while (std::getline(f, line)) {
                std::string k_ = line.substr(0, line.find(' '));
                auto rp = [](double v) { char b_[32]; auto r_ = std::to_chars(b_, b_ + 32, v); return std::string(b_, r_.ptr); };   // repr's digits
                if (k_ == "n") o << "n " << rp(C.n) << "\n";
                else if (k_ == "bnorm") o << "bnorm " << rp(C.c) << "\n";
                else if (k_ == "Ph") o << "Ph " << rp(K) << "\n";
                else if (k_ == "lapse_sl") continue;
                else o << line << "\n";
            }
            // research round 1078 (N2A_LAPSE_GUESS=sl): the lapse families' Sturm-Liouville exponents on THIS cylinder
            if (const char* lg = std::getenv("N2A_LAPSE_GUESS"); lg && std::string(lg) == "sl") {
                auto sl = mk::fm::lapse_sl_sigma(C.u, C.P, C.n, 12);
                o << "lapse_sl ";
                bool first = true;
                for (auto& kv : sl) { if (kv.first < 4) continue; char b_[40]; std::snprintf(b_, sizeof b_, "%.17g", kv.second); o << (first ? "" : ";") << kv.first << ":" << b_; first = false; }
                o << "\n";
            }
            std::ofstream(od + "/config.txt") << o.str();
        }
        if (oo.find("lkdiv=") == std::string::npos) oo += (oo.empty() ? "" : ";") + std::string("lkdiv=exact");
        char e2[1024] = {0};
        std::string ob = od + "/opbuild";
        if (mk_op_build(od.c_str(), ob.c_str(), oo.c_str(), e2, sizeof e2)) throw std::runtime_error(std::string("op-build: ") + e2);
        for (const char* nm : {"dE", "rw", "cw", "rw_out", "kE", "cdE", "rwc"})
            fs::copy_file(fs::path(ob) / (std::string(nm) + ".npy"), fs::path(od) / (std::string(nm) + ".npy"), fs::copy_options::overwrite_existing);
        // the gate rows in the bundle's order -- the operator file's row order (n3b_linthroat's packing: E_sigma, E_sh^r, E_sh^th, E_Q, L_K, H_c, each
        // by degree), the build's own is the kernel's family order; the families checked against the base bundle's gt_fam
        {
            std::vector<std::pair<int, int>> gl;
            std::ifstream f(ob + "/glab.txt");
            std::string l;
            while (std::getline(f, l))
                if (!l.empty()) gl.emplace_back(ob::rowfam(l.substr(0, l.find("@P"))), std::stoi(l.substr(l.find("@P") + 2)));
            static const int forder[6] = {5, 0, 4, 1, 2, 3};          // kernel family (H_c, E_sigma, L_K, E_sh^r, E_sh^th, E_Q) -> its place in the file
            std::vector<size_t> idx(gl.size());
            for (size_t i = 0; i < idx.size(); i++) idx[i] = i;
            std::stable_sort(idx.begin(), idx.end(), [&](size_t x, size_t y) {
                return std::make_pair(forder[gl[x].first], gl[x].second) < std::make_pair(forder[gl[y].first], gl[y].second);
            });
            auto gd = npy_as<int>(npy_read(ob + "/gt_dE.npy")), lv = npy_as<int>(npy_read(ob + "/live.npy"));
            auto gf = npy_as<int>(npy_read(bd + "/gt_fam.npy"));
            if (gf.size() != idx.size()) throw std::runtime_error("mk: bundle-at: " + std::to_string(idx.size()) + " gate rows against the base's " + std::to_string(gf.size()));
            std::vector<int> gd2(idx.size()), lv2(idx.size());
            for (size_t i = 0; i < idx.size(); i++) {
                if (gl[idx[i]].first != gf[i]) throw std::runtime_error("mk: bundle-at: gate row " + std::to_string(i) + "'s family differs from the base's");
                gd2[i] = gd[idx[i]];
                lv2[i] = lv[idx[i]];
            }
            en::npy_write(od + "/gt_dE.npy", gd2.data(), "<i4", {long(gd2.size())}, 4);
            en::npy_write(od + "/live.npy", lv2.data(), "<i4", {long(lv2.size())}, 4);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::snprintf(err, size_t(errcap), "%s", ex.what());
        return 1;
    }
}
}  // extern "C"

// ---- M4b: the Mellin parameters written into a bundle (the stage-B unknowns p; n3a_fit's names) ----
#include <regex>
extern "C" {
// params: a file of lines 'NAME VALUE' (NAME may contain spaces; the value is the last field) -- 'FAML@tT' a grade-T datum (the initial series
// C0, moment 0, in the file's order; the config's set_data), 'tgt X' a residue target (tgt), 'lapse lL' / 'spurious ...' a family amplitude
// (set_offl), 'gauge lL' a gauge amplitude (gauge) -- as n3a_fit.Problem.command writes the engine's flags
int mk_set_params(const char* dir, const char* params, char* err, int errcap) {
    try {
        using namespace mk;
        std::string d(dir);
        std::vector<std::string> collab;
        {
            std::ifstream f(d + "/collab.txt");
            std::string l;
            while (std::getline(f, l))
                if (!l.empty()) collab.push_back(l);
        }
        auto rp = [](double v) { char b_[32]; auto r_ = std::to_chars(b_, b_ + 32, v); return std::string(b_, r_.ptr); };
        std::string setd, tgt, offl, gauge;
        std::vector<int> col, isint;
        std::vector<double> key;
        std::vector<std::complex<double>> mom;
        int K = 0;
        {
            std::ifstream f(d + "/config.txt");
            std::string l;
            while (std::getline(f, l))
                if (l.rfind("K ", 0) == 0) K = std::stoi(l.substr(2));
        }
        if (K <= 0) throw std::runtime_error("mk: set-params: no K in the config");
        std::ifstream f(params);
        if (!f) throw std::runtime_error(std::string("mk: set-params: cannot read ") + params);
        std::string l;
        static const std::regex dat("([A-Z]+)([0-9]+)@t([0-9]+)");
        while (std::getline(f, l)) {
            size_t sp = l.find_last_of(" \t");
            if (l.empty() || sp == std::string::npos) continue;
            std::string nm = l.substr(0, sp), vs = l.substr(sp + 1);
            while (!nm.empty() && (nm.back() == ' ' || nm.back() == '\t')) nm.pop_back();
            double v = std::stod(vs);
            auto add = [&](std::string& s, const std::string& item, char sep) { s += (s.empty() ? "" : std::string(1, sep)) + item; };
            std::smatch m;
            if (nm.rfind("tgt ", 0) == 0) add(tgt, nm.substr(4) + "=" + rp(v), ';');
            else if (nm.rfind("lapse ", 0) == 0 || nm.rfind("spurious ", 0) == 0) add(offl, nm + "=" + rp(v), ';');
            else if (nm.rfind("gauge ", 0) == 0) add(gauge, nm.substr(6) + "=" + rp(v), ';');
            else if (std::regex_match(nm, m, dat)) {
                std::string cl = m[1].str() + m[2].str();
                auto it = std::find(collab.begin(), collab.end(), cl);
                if (it == collab.end()) throw std::runtime_error("mk: set-params: no column " + cl);
                add(setd, nm + "=" + rp(v), ',');
                col.push_back(int(it - collab.begin()));
                key.push_back(std::stod(m[3].str()));
                isint.push_back(1);
                for (int q = 0; q < K; q++) mom.emplace_back(q == 0 ? v : 0.0, 0.0);
            } else throw std::runtime_error("mk: set-params: unknown parameter '" + nm + "'");
        }
        en::npy_write(d + "/C0_col.npy", col.data(), "<i4", {long(col.size())}, 4);
        en::npy_write(d + "/C0_key.npy", key.data(), "<f8", {long(key.size())}, 8);
        en::npy_write(d + "/C0_isint.npy", isint.data(), "<i4", {long(isint.size())}, 4);
        en::npy_write(d + "/C0_mom.npy", mom.data(), "<c16", {long(col.size()), long(K)}, 16);
        std::ifstream cf(d + "/config.txt");
        std::ostringstream o;
        while (std::getline(cf, l)) {
            std::string k_ = l.substr(0, l.find(' '));
            if (k_ == "set_data") o << "set_data " << setd << "\n";
            else if (k_ == "tgt") o << "tgt " << tgt << "\n";
            else if (k_ == "set_offl") o << "set_offl " << offl << "\n";
            else if (k_ == "gauge") o << "gauge " << gauge << "\n";
            else o << l << "\n";
        }
        cf.close();
        std::ofstream(d + "/config.txt") << o.str();
        return 0;
    } catch (const std::exception& ex) {
        std::snprintf(err, size_t(errcap), "%s", ex.what());
        return 1;
    }
}
}  // extern "C"

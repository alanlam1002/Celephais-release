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

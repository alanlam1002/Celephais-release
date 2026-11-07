// Mellin throat recursion, M2f (code round 463; research rounds 1018-1026): the DRIVER -- the whole recursion from the setup bundle.
//
// spinning_trumpet/scripts/n2a_nl.py main, 'the recursion' (route a, nb = 1, opmode engine; the ported paths, NOTES_M2_DESIGN.md section 0):
// per pass the queue (the lattice gdata + 1 .. gmax and the families' towers); per centre: the sources (the rows of the series stripped at g),
// Newton iterations of { the operator (fresh / the cache / re-centred off-integer), the contour step (M2b), the announced centres, the lapse
// families (pass 1, centre 2), the off-integer branch (start, defect loop, conditions) or the data rule (M2c) and the defect correction (M2d) };
// the series updated, the rows of the kept grade reported, the cache; the pass's change and the pass-level stop; after the passes the gauge
// families' Lie flow of the converged series.  Every call goes to the components M1-M2e, with ONE registry (the kernel's Ctx) threaded
// through in the engine's order.
//
// Output (the decision log, design section 5, schema v1): OUT/declog.json with the arrays as OUT/arrays/*.npy ({"npy": "arrays/..."}), the
// final series OUT/series_*.npy and the flowed fields OUT/fields_gauge_*.npy.  Diffed against the Python engine's by
// spinning_trumpet/scripts/m2_dlcompare.py.
#pragma once
#include "mk_bundle.hpp"
#include "mk_families.hpp"
#include "mk_steps.hpp"
#include "mk_tangent.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <atomic>
#include <chrono>
#include <thread>

namespace mk {
namespace en {

using cd = std::complex<double>;
using Vec = std::vector<cd>;

// ---- a minimal .npy writer and JSON builder ----
inline void npy_write(const std::string& path, const void* data, const std::string& descr, const std::vector<long>& shape, size_t item) {
    std::ostringstream h;
    h << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); i++) h << shape[i] << (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
    h << "), }";
    std::string hs = h.str();
    size_t tot = 10 + hs.size() + 1;
    hs += std::string((64 - tot % 64) % 64, ' ') + "\n";
    std::ofstream f(path, std::ios::binary);
    f.write("\x93NUMPY\x01\x00", 8);
    unsigned short hl = static_cast<unsigned short>(hs.size());
    f.write(reinterpret_cast<const char*>(&hl), 2);
    f.write(hs.data(), long(hs.size()));
    size_t n = 1;
    for (long s : shape) n *= size_t(s);
    f.write(static_cast<const char*>(data), long(n * item));
}

struct Json {
    std::ostringstream s;
    std::vector<bool> first{true};
    void sep() {
        if (!first.back()) s << ",";
        first.back() = false;
    }
    void key(const std::string& k) { sep(); s << "\"" << k << "\":"; first.back() = true; first.push_back(true); first.pop_back(); }
    void open(char c) { s << c; first.push_back(true); }
    void close(char c) { s << c; first.pop_back(); }
    // object members / array items
    void kv(const std::string& k, const std::string& raw) { sep(); s << "\"" << k << "\":" << raw; }
    void item(const std::string& raw) { sep(); s << raw; }
    void okey(const std::string& k) { sep(); s << "\"" << k << "\":"; }
    static std::string num(double v) {
        if (std::isnan(v)) return "\"nan\"";
        if (std::isinf(v)) return v > 0 ? "\"inf\"" : "\"-inf\"";
        char b[40];
        std::snprintf(b, sizeof b, "%.17g", v);
        return b;
    }
    static std::string str(const std::string& v) { return "\"" + v + "\""; }
    static std::string nums(const std::vector<double>& v) {
        std::string r = "[";
        for (size_t i = 0; i < v.size(); i++) r += (i ? "," : "") + num(v[i]);
        return r + "]";
    }
};

struct Cfg {
    double R = 0.015, pass_tol = 1e-10, condtol = 0.0, meff_h = 1e-2, top = 9, J = 0.0;
    double pass_stop = 0.0;                                         // research round 1049: n2a_nl --pass-stop (0: off)
    bool newton_floor_memo = false;                                 // n2a_nl --newton-floor-memo
    int N = 96, gmax = 8, gdata = 1, passes = 1, newton = 6, defect = 8, offl_lmax = 8, meff_mode = 0, meff_chunk = 34, gauge_steps = 2;
    bool cond_fit = false, notgt = false, cond_full = false, robust = false, offl_auto = false;
    std::string tgt;
    std::vector<std::pair<std::string, double>> set_offl;
    std::vector<std::pair<int, double>> gauge;
};

struct Setup {
    std::vector<double> rw, cw, TH;
    Vec e1;
    std::vector<int> lc;
    std::vector<std::string> collab;
    int j_br0 = -1;
    std::vector<int> cbb_col;
    std::vector<double> cbb_t;
    Vec cbb_m0;
    std::vector<Entry> C0;
    std::string set_data;             // --set-data (the data directions' order; absent in older bundles: C0's order)
    // M3: the fit's output map (bundles from round 1027 on): the condition rows' scale, the matching data (--match-data), the flags
    std::vector<double> rw_out, rwc;
    std::vector<std::string> condlab;         // the condition rows' labels (bundles from round 1028 on: the fit outputs' labels)
    bool has_md = false, g0rows = false;
    std::string eval_r;
    double md_rm = 0.0, tan_h = 1e-3;
    int md_nthm = 0;
    Vec md_I;                         // nthm x NF
    std::vector<int> md_key;          // nkeys x 3: field, d, mode
    std::vector<double> md_p;         // nkeys x nthm
};

// the driver's inputs from a setup bundle (spinning_trumpet/scripts/m2_bundle.py): the kernel and projections (load_bundle), the gate rows,
// the target series, the column labels, the configuration
inline void load_engine(const std::string& dir, Kernel& k, Proj& P, Setup& S, Cfg& c) {
    Bundle B;
    load_bundle(dir, k, P, B);
    auto A = [&](const std::string& n) { return npy_read(dir + "/" + n + ".npy"); };
    P.gtfam = npy_as<int>(A("gt_fam"));
    P.gtdE = npy_as<int>(A("gt_dE"));
    P.Wg = npy_cplx(A("Wg"));
    auto lv = npy_as<int>(A("live"));
    P.live.assign(lv.begin(), lv.end());
    S.rw = npy_as<double>(A("rw"));
    S.cw = npy_as<double>(A("cw"));
    S.TH = npy_as<double>(A("TH"));
    S.e1 = npy_cplx(A("e1"));
    S.lc = npy_as<int>(A("lc"));
    S.cbb_col = npy_as<int>(A("cbb_col"));
    S.cbb_t = npy_as<double>(A("cbb_t"));
    S.cbb_m0 = npy_cplx(A("cbb_m0"));
    {
        std::ifstream f(dir + "/collab.txt");
        std::string l;
        while (std::getline(f, l))
            if (!l.empty()) S.collab.push_back(l);
    }
    for (size_t e = 0; e < B.C0_col.size(); e++) {
        std::vector<cd> m(B.C0_mom.begin() + long(e) * k.c.K, B.C0_mom.begin() + long(e + 1) * k.c.K);
        S.C0.push_back(Entry{B.C0_col[e], Key{B.C0_key[e], B.C0_isint[e] != 0}, 1, m});
    }
    auto cent0 = npy_as<double>(A("cent0"));
    k.c.cent = cent0;
    if (k.c.exact) {                                    // research round 1031: the interned exponents at the recursion's start (n2a_nl CF.xk)
        std::ifstream xf(dir + "/xk0.npy");
        if (!xf) throw std::runtime_error("mk: an exact-merge bundle without xk0.npy");
        k.c.xk = npy_as<double>(A("xk0"));
    }
    auto gs = [&](const std::string& key) { auto i = B.cfg.find(key); return i == B.cfg.end() ? std::string() : i->second; };
    c.R = B.getd("R", 0.015); c.N = B.geti("N", 96); c.gmax = B.geti("gmax", 8); c.gdata = B.geti("gdata", 1); c.top = B.getd("TOP", 10);
    c.pass_stop = B.getd("pass_stop", 0.0); c.newton_floor_memo = B.geti("newton_floor_memo", 0) != 0;
    c.passes = B.geti("passes", 1); c.pass_tol = B.getd("pass_tol", 1e-10); c.newton = B.geti("newton", 6); c.defect = B.geti("defect", 8);
    c.condtol = B.getd("condtol", 0.0); c.meff_chunk = B.geti("meff_chunk", 34); c.offl_lmax = B.geti("offl_lmax", 8); c.J = B.getd("J", 0.0);
    c.gauge_steps = B.geti("gauge_steps", 2); c.cond_fit = B.geti("cond_fit", 0); c.notgt = B.geti("notarget", 0);
    c.cond_full = B.geti("cond_full", 0); c.robust = B.geti("defect_stop_robust", 0); c.offl_auto = gs("offl") == "auto";
    c.meff_mode = gs("meff_mode") == "jvp" ? 1 : 0;
    c.tgt = gs("tgt");
    S.set_data = gs("set_data");
    S.j_br0 = B.geti("j_br0", -1);
    auto exists = [&](const std::string& n) { return bool(std::ifstream(dir + "/" + n + ".npy")); };
    S.rw_out = npy_as<double>(A("rw_out"));
    if (exists("rwc")) S.rwc = npy_as<double>(A("rwc"));
    {
        std::ifstream f(dir + "/condlab.txt");
        std::string l;
        while (std::getline(f, l))
            if (!l.empty()) S.condlab.push_back(l);
    }
    S.g0rows = B.geti("g0rows", 0);
    S.eval_r = gs("eval_r");
    S.tan_h = B.getd("tan_h", 1e-3);
    if (exists("md_I")) {
        S.has_md = true;
        S.md_I = npy_cplx(A("md_I"));
        S.md_key = npy_as<int>(A("md_key"));
        S.md_p = npy_as<double>(A("md_p"));
        S.md_rm = B.getd("md_rm", 0.0);
        S.md_nthm = int(S.md_I.size() / size_t(k.c.NF));
    }
    auto items = [](const std::string& s) {
        std::vector<std::pair<std::string, double>> v;
        size_t a = 0;
        while (a < s.size()) {
            size_t b = s.find(';', a);
            if (b == std::string::npos) b = s.size();
            std::string it = s.substr(a, b - a);
            size_t e = it.find('=');
            if (e != std::string::npos) {
                std::string k_ = it.substr(0, e);
                k_.erase(0, k_.find_first_not_of(' '));
                k_.erase(k_.find_last_not_of(' ') + 1);
                v.emplace_back(k_, std::stod(it.substr(e + 1)));
            }
            a = b + 1;
        }
        return v;
    };
    c.set_offl = items(gs("set_offl"));
    for (auto& kv : items(gs("gauge"))) c.gauge.emplace_back(std::stoi(kv.first.substr(kv.first.find('l') + 1)), kv.second);
}

// M3: the tangents' options (n2a_nl's --tangent, --tan-tol, --tan-passes, --tan-ctol, --tan-iters, --tan-h, --tan-top-local, --top-margin,
// --tan-no-stall; the stall measure absolute), the rows' derivative (fd: the oracle's four-point stencil; jvp: the kernel's linear response)
struct TanOpt {
    bool on = false, top_local = false, no_stall = false, jvp = false;
    std::string which = "all";
    double tol = 1e-10, ctol = tn::TAN_CTOL, h = 1e-3;
    int passes = 20, iters = tn::TAN_ITERS, margin = 2, threads = 1;
    // research rounds 1030-1031: the direct solve at the family centres (n2a_nl's default; --tan-richardson there = direct=0 here); -1: on
    // for an exact-merge bundle (the oracle_v2 configuration), off otherwise
    int direct = -1, direct_chunk = 34;
};

// n2a_nl's strip: the series without the aligned-g columns (data grades, t < 1, never stripped)
inline std::vector<Entry> strip_g(Ctx& c, const std::vector<Entry>& C, double g, const std::vector<int>& kofs) {
    std::vector<Entry> o;
    std::vector<Key> ks;
    for (size_t j = 0; j < kofs.size(); j++) ks.push_back(ckey(c, g + kofs[j]));
    for (auto& en : C) {
        const Key& t = ks[en.j];
        if (t.v >= 1 && keq(en.t, t)) continue;
        o.push_back(en);
    }
    return o;
}

class Engine {
  public:
    Engine(Kernel& k, Proj& P, const Setup& S, const Cfg& c, const std::string& out) : k_(k), P_(P), S_(S), c_(c), out_(out) {
        mkdir(out.c_str(), 0755);
        mkdir((out + "/arrays").c_str(), 0755);
    }
    void run();
    void tangents(const TanOpt& T);
    // n2a_nl fit_out's fvec: the fit's equations and outputs on a series (the gauge families' amplitudes given)
    std::vector<cd> fvec(Kernel& kc, const std::vector<Entry>& Cx, const std::vector<std::pair<int, double>>& gauge, const std::vector<double>& cents);

  private:
    Kernel& k_;
    Proj& P_;
    const Setup& S_;
    Cfg c_;
    std::string out_;
    Json J_;
    int narr_ = 0;

    // ---- helpers ----
    std::string arr(const std::string& nm, const Vec& v, const std::vector<long>& shape) {
        std::string p = "arrays/" + std::to_string(narr_++) + "_" + nm + ".npy";
        npy_write(out_ + "/" + p, v.data(), "<c16", shape, 16);
        return "{\"npy\":\"" + p + "\"}";
    }
    std::string arrd(const std::string& nm, const std::vector<double>& v) {
        std::string p = "arrays/" + std::to_string(narr_++) + "_" + nm + ".npy";
        npy_write(out_ + "/" + p, v.data(), "<f8", {long(v.size())}, 8);
        return "{\"npy\":\"" + p + "\"}";
    }
    std::string cent() {
        std::vector<double> v = k_.c.cent;
        return Json::nums(v);
    }
    std::vector<Entry> strip(const std::vector<Entry>& C, double g) {
        std::vector<Entry> o;
        std::vector<Key> ks;
        for (int j = 0; j < P_.nn; j++) ks.push_back(ckey(k_.c, g + P_.kofs[j]));
        for (auto& en : C) {
            const Key& t = ks[en.j];
            if (t.v >= 1 && keq(en.t, t)) continue;
            o.push_back(en);
        }
        return o;
    }
    Vec galb(double g) { return project(k_.c, k_.out, g, 1, P_.gfam, P_.dE, P_.W, false); }   // K x nn of the kernel's current rows
    double maxabs(const Vec& v) {
        double m = 0.0;
        for (auto& z : v) m = std::max(m, std::abs(z));
        return m;
    }
    Vec moments(const Vec& X) { return ct::moments_of(X, k_.c.K, P_.nn, S_.e1, c_.R); }
    dt::Eval eval(const std::vector<Entry>& base, double g) {
        dt::Eval ev;
        Kernel* k = &k_;
        const Proj* P = &P_;
        std::vector<int> xr = xrows_;
        Vec e1 = S_.e1;
        double R = c_.R, top = c_.top;
        ev.cond = [k, P, base, g, top, xr, e1, R](const Vec& X) {
            k->run(with_grade(k->c, base, X, g, P->kofs, e1, R), nullptr, top);
            int K = k->c.K, nc = P->nc, nn = P->nn;
            auto S = project(k->c, k->out, g, 1, P->cfam, P->cdE, P->Wc, true);
            Vec G;
            if (!xr.empty()) G = project(k->c, k->out, g, 1, P->gfam, P->dE, P->W, false);
            Vec r;
            for (int m = 0; m < 2 && m < K; m++) {
                for (int i = 0; i < nc; i++) r.push_back(S[size_t(m) * nc + i]);
                for (int x : xr) r.push_back(G[size_t(m) * nn + x]);
            }
            return r;
        };
        ev.gate = [k, P, base, g, top, e1, R](const Vec* X, double gg) {
            if (X) k->run(with_grade(k->c, base, *X, g, P->kofs, e1, R), nullptr, top);
            else k->run(base, nullptr, top);
            auto Sg = project(k->c, k->out, gg, 1, P->gtfam, P->gtdE, P->Wg, true);
            double m = 0.0;
            for (size_t q = 0; q < P->gtfam.size(); q++)
                if (P->live[q]) m = std::max(m, std::abs(Sg[q]));
            return m;
        };
        return ev;
    }
    st::Rows rowsfn(const std::vector<Entry>& base, double g) {
        st::Rows r;
        Kernel* k = &k_;
        const Proj* P = &P_;
        Vec e1 = S_.e1;
        double R = c_.R, top = c_.top;
        r.gal = [k, P, base, g, top, e1, R](const Vec& X) {
            k->run(with_grade(k->c, base, X, g, P->kofs, e1, R), nullptr, top);
            return project(k->c, k->out, g, 1, P->gfam, P->dE, P->W, false);
        };
        r.stored = [k, P, base, g, top, e1, R](const Vec& X) {
            k->run(with_grade(k->c, base, X, g, P->kofs, e1, R), nullptr, top);
            return project(k->c, k->out, g, 1, P->cfam, P->cdE, P->Wc, true);
        };
        return r;
    }
    std::vector<int> xrows_;
    // M3: the forward's last pass -- each centre's step state, the centre order, the final series, the families
    std::map<double, tn::StepState> steps_;
    std::vector<double> order_last_;
    std::vector<Entry> C_final_;
    std::vector<std::pair<std::string, Key>> fams_;
    static double kf(double g) { return std::nearbyint(g * 1e9) / 1e9; }
};

inline void Engine::run() {
    if (const char* t = std::getenv("MK_TRACE_CANON")) k_.c.trace_above = std::atof(t);
    int nn = P_.nn, K = k_.c.K, N = c_.N, NB = nn, nc = P_.nc;
    size_t nm = size_t(NB) * NB, nmc = size_t(nc) * NB;
    std::vector<Entry> C = S_.C0;
    std::map<double, std::pair<Vec, Vec>> cache;                    // meff_cache: the converged operator per integer centre
    std::map<double, double> nfloor;                                // --newton-floor-memo: per centre, the floor a futile Newton step found
    std::vector<double> rel_hist;                                   // --pass-stop: the pass changes relative to the series' scale
    std::map<double, int> rk_fix;
    std::map<double, Vec> c_warm;
    std::vector<std::pair<std::string, Key>> fams;                  // fams_offl, in insertion order
    bool have_mref = false;
    double mref_g = 0.0;
    Vec mref_s, mref_c;
    std::map<std::string, double> offl_set(c_.set_offl.begin(), c_.set_offl.end());
    J_.open('{');
    J_.okey("passes");
    J_.open('[');
    std::string families_json = "null";
    int ps = 0;
    for (ps = 0; ps < c_.passes; ps++) {
        std::vector<Entry> Cprev = C;
        J_.sep();
        J_.open('{');
        J_.kv("cent_start", cent());
        J_.okey("centres");
        J_.open('[');
        fm::Queue Q;
        std::vector<Key> lat;
        for (int g = c_.gdata + 1; g <= c_.gmax; g++) lat.push_back(Key{double(g), true});
        std::vector<Key> fk;
        for (auto& f : fams) fk.push_back(f.second);
        Q.push(lat);
        Q.push(fm::towers(k_.c, fk, c_.gmax));
        std::vector<double> order;
        Key gk;
        while (Q.pop(gk)) {
            double g = gk.v;
            bool isint = gk.isint;
            order.push_back(g);
            J_.sep();
            J_.open('{');
            J_.kv("g", Json::num(g));
            J_.kv("isint", isint ? "true" : "false");
            J_.kv("passno", std::to_string(ps));
            // the sources: the rows of the series stripped at g
            std::vector<Entry> Cs = strip(C, g);
            k_.c.trace_tag = "pass " + std::to_string(ps) + " g " + std::to_string(g) + " sources";
            k_.run(Cs, nullptr, c_.top);
            Vec S = galb(g);
            double Smax = maxabs(S);
            auto cand = c_.offl_auto ? fm::announce_candidates(k_.out, P_, g, c_.gmax) : std::vector<std::pair<double, double>>{};
            J_.okey("newton");
            J_.open('[');
            Vec Ms, Mc, Xk, Xg;
            tn::StepState stc;                                      // M3: the step's linear data (n2a_nl step_state; the last pass's kept)
            stc.isint = isint;
            stc.kept_only = isint;                                  // nb = 1, no --dc-predictor
            std::vector<double> newton_hist;
            bool memo_used = false;
            bool offint_done = false;
            std::vector<char> act;
            for (int newt = 0; newt < c_.newton; newt++) {
                bool from_cache = newt == 0 && cache.count(g);
                k_.c.trace_tag = "pass " + std::to_string(ps) + " g " + std::to_string(g) + " newton " + std::to_string(newt) + " operator";
                std::string opname;
                if (isint || !have_mref) {
                    opname = "engine-integer";
                    if (newt) meff(k_, P_, with_grade(k_.c, Cs, Xk, g, P_.kofs, S_.e1, c_.R), g, c_.top, c_.meff_mode, c_.meff_h, c_.meff_chunk, Ms, Mc);
                    else if (from_cache) { Ms = cache[g].first; Mc = cache[g].second; }
                    else { meff(k_, P_, Cs, g, c_.top, c_.meff_mode, c_.meff_h, c_.meff_chunk, Ms, Mc); cache[g] = {Ms, Mc}; }
                    have_mref = true; mref_g = g; mref_s = Ms; mref_c = Mc;
                } else {
                    opname = "engine-recentred";
                    recentre(mref_s, Ms, nm, g - mref_g);
                    recentre(mref_c, Mc, nmc, g - mref_g);
                }
                k_.c.trace_tag = "pass " + std::to_string(ps) + " g " + std::to_string(g) + " contour/sources";
                // the contour step
                Vec Sn = newt ? st::newton_sources(Xk, Ms, K, NB, S_.e1, c_.R, rowsfn(Cs, g)) : ct::nodes_of(S, K, NB, S_.e1, c_.R);
                act = ct::active(g, P_.kofs);
                xrows_.clear();
                for (int j = 0; j < NB; j++)
                    if (!act[j]) xrows_.push_back(j);
                double kf = std::nearbyint(g * 1e9) / 1e9;
                ct::Contour o;
                ct::contour(Ms, NB, S_.rw, S_.cw, S_.e1, c_.R, act, Sn, rk_fix.count(kf) ? rk_fix[kf] : -1, o);
                if (!rk_fix.count(kf)) rk_fix[kf] = o.rk_thr;
                int rk = o.rk;
                stc.Minv = o.Minv;
                stc.act = act;
                stc.rk = rk;
                stc.Wq.clear();
                stc.has_src = false;
                J_.sep();
                J_.open('{');
                J_.kv("operator", Json::str(opname));
                J_.kv("from_cache", from_cache && isint ? "true" : "false");
                J_.kv("Ms", arr("Ms", Ms, {3, NB, NB}));
                J_.kv("Mc", arr("Mc", Mc, {3, nc, NB}));
                J_.kv("recentred_from", opname == "engine-recentred" ? Json::num(mref_g) : "null");
                std::string a = "[";
                for (int j = 0; j < NB; j++) a += std::string(j ? "," : "") + (act[j] ? "true" : "false");
                J_.kv("act", a + "]");
                J_.kv("kn", std::to_string(o.kn));
                std::string bd = "[";
                for (size_t q = 0; q < o.bord.size(); q++) bd += (q ? "," : "") + std::string("[") + std::to_string(o.bord[q].first) + "," + std::to_string(o.bord[q].second) + "]";
                J_.kv("bord", bd + "]");
                J_.kv("S", arr("S", S, {K, NB}));
                J_.kv("Sn", arr("Sn", Sn, {N, NB}));
                J_.kv("X0", arr("X0", moments(o.X0), {K, NB}));
                J_.kv("X0_nodes", arr("X0_nodes", o.X0, {N, NB}));
                J_.kv("Res", arr("Res", o.Res, {NB, NB}));
                J_.kv("Sr", arrd("Sr", o.Sr));
                J_.kv("rref", Json::num(o.rref));
                J_.kv("rk_threshold", std::to_string(o.rk_thr));
                J_.kv("rk", std::to_string(rk));
                J_.kv("residue_dirs", rk ? arr("residue_dirs", o.rdirs, {rk, 2 * NB}) : "null");
                // the announced centres (--offl auto), registered here as the engine does
                std::vector<Key> add;
                if (c_.offl_auto) {
                    std::vector<fm::Ann> ann;
                    fm::announce_register(k_.c, cand, ann, add);
                    std::string an = "[";
                    for (size_t q = 0; q < ann.size(); q++) an += (q ? "," : "") + std::string("[") + Json::num(ann[q].key) + "," + Json::num(ann[q].mag) + "]";
                    J_.kv("announce", an + "]");
                    Q.push(add);
                    if (ps == 0 && isint && g == 2 && fams.empty()) {
                        fm::Lapse L;
                        fm::lapse_roots(k_.c, Ms, NB, S_.rw, S_.cw, o.bord, g, k_.c.n, c_.gmax, c_.offl_lmax, L);
                        for (auto& l : L.lapse) fams.emplace_back("lapse l" + std::to_string(l.first), l.second);
                        std::vector<Key> fk2;
                        for (auto& f : fams) fk2.push_back(f.second);
                        Q.push(fm::towers(k_.c, fk2, c_.gmax));
                        Json F;
                        F.open('{');
                        F.kv("block_roots_real", Json::nums(L.evr));
                        F.kv("eig_all", arr("eig_all", L.eig_all, {long(L.eig_all.size())}));
                        std::string la = "{";
                        for (size_t q = 0; q < L.lapse.size(); q++) la += (q ? "," : "") + Json::str("lapse l" + std::to_string(L.lapse[q].first)) + ":" + Json::num(L.lapse[q].second.v);
                        F.kv("lapse", la + "}");
                        std::string rg = "[";
                        for (size_t q = 0; q < L.registry.size(); q++) rg += (q ? "," : "") + std::string("[") + std::to_string(L.registry[q].first) + "," + Json::num(L.registry[q].second) + "]";
                        F.kv("registry", rg + "]");
                        F.close('}');
                        families_json = F.s.str();
                    }
                }
                if (!isint) {
                    // the off-integer centre: the family's datum, the defect loop, the conditions (not imposed)
                    std::string lab;
                    for (auto& f : fams)
                        if (keq(f.second, gk)) { lab = f.first; break; }
                    double datum = (!lab.empty() && offl_set.count(lab)) ? offl_set[lab] : 0.0;
                    std::vector<Vec> Wn;
                    Vec Xs = o.X0;
                    bool hf = false;
                    cd amp(0.0, 0.0);
                    if (rk) dt::offl_start(o.X0, o.Wq, nn, NB, S_.e1, c_.R, !lab.empty(), datum, c_.cond_fit, eval(Cs, g), Wn, Xs, hf, amp);
                    st::DefectOut d;
                    st::defect_off(Xs, o.Minv, Wn, K, NB, S_.e1, c_.R, std::max(c_.defect, 4), c_.robust, rowsfn(Cs, g), d);
                    C = with_grade(k_.c, C, d.X, g, P_.kofs, S_.e1, c_.R);
                    stc.Wq = Wn;
                    stc.X = d.X;
                    stc.has_label = !lab.empty();
                    stc.label = lab;
                    J_.close('}');
                    J_.close(']');
                    J_.kv("offlattice", "true");
                    J_.kv("label", lab.empty() ? "null" : Json::str(lab));
                    J_.kv("datum", lab.empty() ? "null" : Json::num(datum));
                    J_.kv("amplitude_fixed", hf ? "[" + Json::num(amp.real()) + "," + Json::num(amp.imag()) + "]" : "null");
                    J_.kv("defect", Json::nums(d.hist));
                    J_.kv("conditions", arr("conditions", d.conditions, {K, nc}));
                    J_.kv("X", arr("X", moments(d.X), {K, NB}));
                    J_.kv("cent_after", cent());
                    offint_done = true;
                    break;
                }
                // the data rule (integer centre with residues), then the defect correction
                dt::DataIn din;
                dt::DataOut dout;
                std::function<Vec(const Vec&)> dfix;
                Xg = o.X0;
                std::vector<Vec> Wn;
                k_.c.trace_tag = "pass " + std::to_string(ps) + " g " + std::to_string(g) + " data rule";
                if (rk) {
                    dt::TgtIn ti;
                    ti.nn = nn; ti.rk = rk; ti.g = g; ti.kofs = P_.kofs; ti.lc = S_.lc; ti.collab = S_.collab; ti.notgt = c_.notgt;
                    ti.cbb_col = S_.cbb_col; ti.cbb_t = S_.cbb_t; ti.cbb_m0 = S_.cbb_m0;
                    std::string t = c_.tgt;
                    size_t a0 = 0;
                    while (a0 < t.size()) {
                        size_t b = t.find(';', a0);
                        if (b == std::string::npos) b = t.size();
                        std::string it = t.substr(a0, b - a0);
                        size_t e = it.find('=');
                        if (e != std::string::npos) ti.tgt.emplace_back(it.substr(0, e), std::stod(it.substr(e + 1)));
                        a0 = b + 1;
                    }
                    std::vector<std::pair<int, double>> hit;
                    dt::targets(ti, din.tgtv, din.prevv, din.wt, din.tg0, hit);
                    stc.has_src = true;
                    stc.src_tgt.assign(2 * size_t(NB), std::string());
                    for (auto& h : hit) stc.src_tgt[h.first] = S_.collab[h.first] + "@t" + std::to_string(int(g + P_.kofs[h.first]));
                    din.nn = nn; din.NB = NB; din.rk = rk; din.R = c_.R; din.e1 = S_.e1; din.X0 = o.X0; din.Wq = o.Wq; din.g = g;
                    din.notgt = c_.notgt; din.cond_fit = c_.cond_fit; din.cond_full = c_.cond_full; din.condtol_fixed = c_.condtol; din.lc = S_.lc;
                    din.j_br0 = S_.j_br0;
                    din.warm = c_warm.count(kf) && int(c_warm[kf].size()) == rk;
                    if (din.warm) din.cwarm = c_warm[kf];
                    dt::data_rule(din, eval(C, g), dout);
                    c_warm[kf] = dout.c;
                    Xg = dout.Xg;
                    Wn = dout.Wq;
                    stc.Wq = dout.Wq;
                    stc.wt = din.wt;
                    if (c_.cond_fit) dfix = [din, dout](const Vec& X) { return dt::data_fix(X, din, dout); };
                    Json D;
                    D.open('{');
                    D.kv("target", arr("target", din.tgtv, {2 * NB}));
                    D.kv("tg0", arr("tg0", din.tg0, {2 * NB}));
                    D.kv("weights", arrd("weights", din.wt));
                    D.kv("At", arr("At", dout.At, {2 * NB, rk}));
                    D.kv("hit", std::to_string(hit.size()));
                    D.kv("notarget", c_.notgt ? "true" : "false");
                    D.kv("warm", din.warm ? "true" : "false");
                    D.kv("c", arr("c", dout.c, {rk}));
                    D.kv("F_start", arr("F_start", dout.F_start, {long(dout.F_start.size())}));
                    D.kv("F_final", arr("F_final", dout.F_final, {long(dout.F_final.size())}));
                    D.kv("cond_sv", Json::nums(dout.cond_sv));
                    D.kv("cond_rank", std::to_string(dout.cond_rank));
                    D.kv("lin_err", Json::num(dout.lin_err));
                    D.kv("tol", "{\"tau\":" + Json::num(dout.tau) + ",\"size\":" + Json::num(dout.size) + ",\"condtol\":" + Json::num(dout.condtol) + "}");
                    std::string hs = "[";
                    for (size_t q = 0; q < dout.hist.size(); q++) {
                        auto& h = dout.hist[q];
                        hs += std::string(q ? "," : "") + (h.kind == 2 ? "{\"direct\":true,\"res\":" + Json::num(h.res) + "}"
                                                           : h.kind == 1 ? "{\"polish\":" + Json::num(h.polish) + "}"
                                                                         : "{\"res\":" + Json::num(h.res) + ",\"mu\":" + Json::num(h.mu) + ",\"rank\":" + std::to_string(h.rank) + "}");
                    }
                    D.kv("hist", hs + "]");
                    std::string fs = "[";
                    for (size_t q = 0; q < dout.fitted.size(); q++) {
                        auto& f = dout.fitted[q];
                        fs += std::string(q ? "," : "") + "{\"amp\":[" + Json::num(f.amp.real()) + "," + Json::num(f.amp.imag()) + "],\"cols\":[" +
                              Json::str(S_.collab[f.cols[0]]) + "," + Json::str(S_.collab[f.cols[1]]) + "," + Json::str(S_.collab[f.cols[2]]) +
                              "],\"l0_share\":" + Json::num(f.l0_share) + ",\"l2_content\":" + Json::num(f.l2_content) + "}";
                    }
                    D.kv("fitted", fs + "]");
                    if (dout.has_alt) {
                        std::string ra = "[";
                        for (size_t q = 0; q < dout.rows_at_alt.size(); q++) ra += (q ? "," : "") + Json::nums(dout.rows_at_alt[q]);
                        D.kv("alt", "{\"poly_deg\":" + std::to_string(dout.poly_deg) + ",\"alt_b1\":" + Json::num(dout.alt_b1) + ",\"poly\":" + Json::nums(dout.poly) + ",\"alt_tau\":" +
                                        Json::nums(dout.alt_tau) + ",\"b_alt\":" + Json::nums(dout.b_alt) + ",\"tau_fit\":" + Json::nums(dout.tau_fit) +
                                        ",\"rows_at_alt\":" + ra + "]}");
                    }
                    D.close('}');
                    J_.kv("data", D.s.str());
                }
                k_.c.trace_tag = "pass " + std::to_string(ps) + " g " + std::to_string(g) + " defect";
                st::DefectOut d;
                st::defect_int(Xg, o.Minv, Wn, act, Smax, K, NB, S_.e1, c_.R, c_.defect, c_.robust, dfix, rowsfn(Cs, g), d);
                Xg = d.X;
                stc.X = Xg;
                newton_hist.push_back(d.rows_after);
                J_.kv("defect", Json::nums(d.hist));
                J_.kv("defect_drift", Json::num(d.drift));
                J_.kv("rows_after", Json::num(d.rows_after));
                J_.kv("X", arr("X", moments(Xg), {K, NB}));
                J_.kv("X_nodes", arr("X_nodes", Xg, {N, NB}));
                J_.close('}');
                {
                    // n2a_nl: Newton at its floor (a step that does not halve the rows) records the floor; --newton-floor-memo: a centre whose
                    // Newton step was futile at an earlier pass is not re-linearised while its rows stay within twice that floor
                    size_t nh = newton_hist.size();
                    bool stagn = nh > 1 && newton_hist[nh - 1] > st::HALVE * newton_hist[nh - 2];
                    if (stagn) {
                        nfloor[g] = std::max(newton_hist[nh - 1], newton_hist[nh - 2]);
                    } else if (c_.newton_floor_memo && newt == 0 && nfloor.count(g) && newton_hist[nh - 1] <= 2 * nfloor[g]) {
                        memo_used = true;
                        break;
                    }
                }
                if (st::newton_continue(newton_hist, Smax, newt, c_.newton, isint, c_.defect)) {
                    Xk = Xg;
                    continue;
                }
                break;
            }
            steps_[std::nearbyint(g * 1e9) / 1e9] = stc;
            if (!offint_done) {
                J_.close(']');
                C = with_grade(k_.c, C, Xg, g, P_.kofs, S_.e1, c_.R);
                k_.c.trace_tag = "pass " + std::to_string(ps) + " g " + std::to_string(g) + " rows after";
                k_.run(C, nullptr, c_.top);
                Vec rb = galb(g);
                double rres = 0.0;
                for (int m = 0; m < K; m++)
                    for (int j = 0; j < nn; j++)
                        if (act[j]) rres = std::max(rres, std::abs(rb[size_t(m) * nn + j]));
                if (isint) cache[g] = {Ms, Mc};
                J_.kv("rows_g", Json::num(rres));
                J_.kv("newton_hist", Json::nums(newton_hist));
                if (c_.newton_floor_memo) J_.kv("newton_floor_memo", memo_used ? "true" : "false");
                J_.kv("cent_after", cent());
                J_.kv("pending_after", Json::nums(Q.after()));
            }
            J_.close('}');
        }
        J_.close(']');
        // the pass's change of the series (n2a_nl: max over C of |v - Cprev.get(j, {}).get(t, 0 v)|)
        double dch = 0.0;
        for (auto& en : C) {
            const Entry* pv = nullptr;
            for (auto& q : Cprev)
                if (q.j == en.j && keq(q.t, en.t)) { pv = &q; break; }
            for (size_t i = 0; i < en.m.size(); i++) dch = std::max(dch, std::abs(en.m[i] - (pv ? pv->m[i] : cd(0.0, 0.0))));
        }
        J_.kv("order", Json::nums(order));
        order_last_ = order;
        J_.kv("change", Json::num(dch));
        J_.kv("cent_end", cent());
        J_.close('}');
        if (ps && dch < c_.pass_tol) { ps++; break; }
        if (c_.pass_stop > 0) {
            // research round 1049 (n2a_nl --pass-stop): converged when the change relative to the series' scale (its largest entry) stays below
            // pass_stop for two consecutive passes
            double scl = 0.0;
            for (auto& en : C)
                for (auto& v : en.m) scl = std::max(scl, std::abs(v));
            rel_hist.push_back(dch / std::max(scl > 0 ? scl : 1.0, 1e-300));
            if (ps && rel_hist.size() >= 2 && rel_hist.back() < c_.pass_stop && rel_hist[rel_hist.size() - 2] < c_.pass_stop) { ps++; break; }
        }
    }
    J_.close(']');
    J_.kv("families", families_json);
    C_final_ = C;
    fams_ = fams;
    J_.kv("passes_run", std::to_string(std::min(ps, c_.passes)));
    // the final series and the gauge families' Lie flow of it
    {
        std::vector<int> col, isi;
        std::vector<double> key;
        Vec mom;
        for (auto& en : C) {
            col.push_back(en.j); key.push_back(en.t.v); isi.push_back(en.t.isint);
            for (int kk = 0; kk < K; kk++) mom.push_back(en.m[kk]);
        }
        npy_write(out_ + "/series_col.npy", col.data(), "<i4", {long(col.size())}, 4);
        npy_write(out_ + "/series_key.npy", key.data(), "<f8", {long(key.size())}, 8);
        npy_write(out_ + "/series_isint.npy", isi.data(), "<i4", {long(isi.size())}, 4);
        npy_write(out_ + "/series_mom.npy", mom.data(), "<c16", {long(col.size()), K}, 16);
    }
    if (!c_.gauge.empty()) {
        Ser F[6];
        k_.fields(C, c_.top, F);
        fm::lie_flow(k_.c, F, c_.gauge, c_.J, S_.TH, c_.gauge_steps);
        static const char* FN[6] = {"psi2", "Phb", "qf", "br", "bt", "Qb"};
        for (int f = 0; f < 6; f++) {
            std::vector<double> ks;
            std::vector<int> ii;
            Vec pc;
            for (auto& kv : F[f].d) {
                ks.push_back(kv.first.v); ii.push_back(kv.first.isint);
                pc.insert(pc.end(), kv.second.a.begin(), kv.second.a.end());
            }
            std::string b = out_ + "/fields_gauge_" + FN[f];
            npy_write(b + "_key.npy", ks.data(), "<f8", {long(ks.size())}, 8);
            npy_write(b + "_isint.npy", ii.data(), "<i4", {long(ii.size())}, 4);
            npy_write(b + "_pieces.npy", pc.data(), "<c16", {long(ks.size()), K, k_.c.NF}, 16);
            std::ofstream(b + "_meta.txt") << (F[f].has_e ? std::to_string(F[f].e) : "None") << " " << Json::num(F[f].top) << "\n";
        }
    }
    J_.kv("cent_final", cent());
    J_.close('}');
    std::ofstream(out_ + "/declog.json") << J_.s.str();
}


// n2a_nl fit_out's fvec (route a, nb = 1): [the square rows at the data grades (moments 0, 1; / rw_out); the l = 0 conditions at every centre
// (/ the condition rows' scale); the monitored square rows at the other centres (zeros where the grade is beyond the rows' validity); grade
// 0's rows (--g0rows: aligned -1, 0); stage B's matching keys (--match-data: the fields through gmax + 1 after the gauge families' Lie flow,
// summed at r_m -- values, d/dr -- on the engine grid, interpolated to the matching nodes, each key's projector row)]
inline std::vector<cd> Engine::fvec(Kernel& kc, const std::vector<Entry>& Cx, const std::vector<std::pair<int, double>>& gauge,
                                    const std::vector<double>& cents) {
    int nn = P_.nn, nc = P_.nc, K = kc.c.K, NF = kc.c.NF;
    if (!S_.eval_r.empty()) throw std::runtime_error("mk: --eval-r (the outputs at radii) is not ported");
    kc.run(Cx, nullptr, c_.top);
    std::vector<cd> v;
    auto gal = [&](double gg, bool soft) {
        std::vector<cd> G;
        try {
            G = project(kc.c, kc.out, gg, 1, P_.gfam, P_.dE, P_.W, false);
        } catch (const std::runtime_error&) {
            if (!soft) throw;
            G.assign(size_t(K) * nn, cd(0.0, 0.0));
        }
        for (int m = 0; m < 2; m++)
            for (int i = 0; i < nn; i++) v.push_back(G[size_t(m) * nn + i] / S_.rw_out[i]);
    };
    for (int gg = 1; gg <= c_.gdata; gg++) gal(gg, false);
    for (double gc : cents) {
        auto Sq = project(kc.c, kc.out, gc, 1, P_.cfam, P_.cdE, P_.Wc, true);
        for (int m = 0; m < 2; m++)
            for (int q = 0; q < nc; q++) v.push_back(Sq[size_t(m) * nc + q] / S_.rwc[q]);
    }
    for (size_t i = size_t(c_.gdata); i < cents.size(); i++) gal(cents[i], true);
    if (S_.g0rows)
        for (int gg = -1; gg <= 0; gg++) gal(gg, false);
    if (S_.has_md) {
        Ser F[6];
        kc.fields(Cx, c_.gmax + 1, F);
        if (!gauge.empty()) fm::lie_flow(kc.c, F, gauge, c_.J, S_.TH, c_.gauge_steps);
        double r = S_.md_rm, n = kc.c.n;
        // n3_throat_eval.ms_at: r^e sum_t x^t sum_m c_m (ln x)^m / m!, x = r^n (None for an empty series)
        auto ms_at = [&](const Ser& ms, std::vector<cd>& tot) {
            if (ms.d.empty()) return false;
            double x = std::pow(r, n), lx = std::log(x);
            std::vector<double> fac(K);
            double f = 1.0;
            for (int m = 0; m < K; m++) { if (m) f *= m; fac[m] = std::pow(lx, m) / f; }
            tot.assign(NF, cd(0.0, 0.0));
            for (auto& kv : ms.d) {
                double xt = std::pow(x, kv.first.v);
                for (int q = 0; q < NF; q++) {
                    cd s = 0.0;
                    for (int m = 0; m < K; m++) s += fac[m] * kv.second.a[size_t(m) * NF + q];
                    tot[q] += xt * s;
                }
            }
            double re = std::pow(r, double(ms.e));
            for (auto& z : tot) z *= re;
            return true;
        };
        std::vector<cd> val[6], dval[6];
        bool hv[6], hd[6];
        for (int f = 0; f < 6; f++) {
            hv[f] = ms_at(F[f], val[f]);
            hd[f] = ms_at(kc.c.drad(F[f]), dval[f]);
            if (hd[f])
                for (auto& z : dval[f]) z = (r * z) / r;               // fields_at returns r d/dr; stage B's C1 is d/dr (fv / r_m)
        }
        size_t nk = S_.md_key.size() / 3;
        int nt = S_.md_nthm;
        for (size_t q = 0; q < nk; q++) {
            int f = S_.md_key[3 * q], d = S_.md_key[3 * q + 1];
            const std::vector<cd>& fv = d ? dval[f] : val[f];
            if (!(d ? hd[f] : hv[f])) { v.push_back(0.0); continue; }
            double s = 0.0;
            for (int t = 0; t < nt; t++) {
                cd y = 0.0;
                for (int k = 0; k < NF; k++) y += fv[k] * S_.md_I[size_t(t) * NF + k];
                s += y.real() * S_.md_p[q * size_t(nt) + t];
            }
            v.push_back(s);
        }
    }
    return v;
}

// M3: the tangents (n2a_nl fit_out, --tangent): per direction the tangent series dC, carried through the forward's last-pass order with
// each centre's step state, tangent passes until the change is < tol relative (with --tan-no-stall no stall exit; else the absolute stall
// test: the change failing to halve over two consecutive passes, from pass 3).  Directions: the data parameters (--set-data; a unit moment 0
// at the column's data centre), the --tgt targets ('tgt NAME'), the lapse families in order of their centres, the gauge families (their
// tangent series is zero: their derivative enters through the outputs' Lie flow).  Each direction runs on its own copy of the kernel (its
// own registry, as the oracle's forked workers).  Output: OUT/tangent/<i>_{col,key,isint,mom}.npy, OUT/tangent.json.
inline void Engine::tangents(const TanOpt& T) {
    int nn = P_.nn, K = k_.c.K, NB = nn;
    const std::vector<int>& kofs = P_.kofs;
    std::vector<std::pair<std::string, tn::Seed>> dirs;
    auto collab_of = [&](const std::string& nm, int& j, double& t) {
        size_t at = nm.find("@t");
        std::string cl = nm.substr(0, at);
        t = std::stod(nm.substr(at + 2));
        j = -1;
        for (int q = 0; q < nn; q++)
            if (S_.collab[q] == cl) { j = q; break; }
        if (j < 0) throw std::runtime_error("mk: tangent direction " + nm + ": no such column");
    };
    if (!S_.set_data.empty()) {
        size_t a = 0;
        while (a < S_.set_data.size()) {
            size_t b = S_.set_data.find(',', a);
            if (b == std::string::npos) b = S_.set_data.size();
            std::string it = S_.set_data.substr(a, b - a);
            if (it.find('=') != std::string::npos) dirs.push_back({it.substr(0, it.find('=')), tn::Seed{"data", it.substr(0, it.find('='))}});
            a = b + 1;
        }
    } else {
        for (auto& en : S_.C0)
            if (en.t.isint) {
                std::string nm = S_.collab[en.j] + "@t" + std::to_string(int(en.t.v));
                dirs.push_back({nm, tn::Seed{"data", nm}});
            }
    }
    {
        size_t a = 0;
        const std::string& t = c_.tgt;
        while (a < t.size()) {
            size_t b = t.find(';', a);
            if (b == std::string::npos) b = t.size();
            std::string it = t.substr(a, b - a);
            size_t e = it.find('=');
            if (e != std::string::npos) {
                std::string nm = it.substr(0, e);
                nm.erase(0, nm.find_first_not_of(' '));
                nm.erase(nm.find_last_not_of(' ') + 1);
                dirs.push_back({"tgt " + nm, tn::Seed{"tgt", nm}});
            }
            a = b + 1;
        }
    }
    {
        auto fs = fams_;
        std::stable_sort(fs.begin(), fs.end(), [](const std::pair<std::string, Key>& x, const std::pair<std::string, Key>& y) { return x.second.v < y.second.v; });
        for (auto& f : fs) dirs.push_back({f.first, tn::Seed{"lap", f.first}});
        auto gs = c_.gauge;
        std::sort(gs.begin(), gs.end());
        for (auto& gl : gs) dirs.push_back({"gauge l" + std::to_string(gl.first), tn::Seed{"gauge", "l" + std::to_string(gl.first)}});
    }
    if (T.which != "all") {
        std::vector<std::pair<std::string, tn::Seed>> keep;
        std::string w = ";" + T.which + ";";
        for (auto& d : dirs)
            if (w.find(";" + d.first + ";") != std::string::npos) keep.push_back(d);
        dirs = keep;
    }
    // the fit's centres (n2a_nl: the data grades, then the forward's last-pass order without them) and its outputs on the final series,
    // computed on the main kernel before the directions' copies are taken (the oracle computes them before forking its workers)
    std::vector<double> cents;
    for (int gg = 1; gg <= c_.gdata; gg++) cents.push_back(gg);
    for (double g : order_last_) {
        bool dat = false;
        for (int gg = 1; gg <= c_.gdata; gg++)
            if (std::abs(g - gg) < 1e-9) dat = true;
        if (!dat) cents.push_back(g);
    }
    std::vector<double> base;
    double base_imag = 0.0;
    bool outputs = !S_.rwc.empty();
    if (outputs) {
        for (auto& z : fvec(k_, C_final_, c_.gauge, cents)) { base.push_back(z.real()); base_imag = std::max(base_imag, std::abs(z.imag())); }
    }
    struct Res {
        std::vector<Entry> dC;
        std::vector<double> dv;
        double dv_imag = 0.0;
        std::vector<double> ph, step_max, step_last, step_last_abs;
        double secs = 0.0;
        std::string err;
    };
    std::vector<Res> res(dirs.size());
    auto dmax = [](const std::vector<Entry>& D) {
        double m = 0.0;
        for (auto& en : D)
            for (auto& z : en.m) m = std::max(m, std::abs(z));
        return m;
    };
    // the family centres' linear maps (n2a_nl fam_matrix), built once before the directions (the oracle: before its workers fork)
    std::map<double, tn::FamMap> famL;
    if (!dirs.empty() && (T.direct > 0 || (T.direct < 0 && k_.c.exact))) {      // (M4b: no directions, no family maps)
        for (double g : order_last_) {
            const tn::StepState& st = steps_.at(kf(g));
            if (st.isint || !st.has_label) continue;
            auto t0 = std::chrono::steady_clock::now();
            double top_g = T.top_local ? std::min(c_.top, std::ceil(g - 1e-9) + T.margin) : c_.top;
            std::vector<Entry> Cpt = with_grade(k_.c, C_final_, st.X, g, kofs, S_.e1, c_.R);
            const double es[4] = {T.h, -T.h, 2 * T.h, -2 * T.h};
            auto rows4 = [&](const std::vector<std::pair<int, int>>& cols) -> Vec {
                std::vector<Entry> Cq = Cpt;
                int Bq = 4 * int(cols.size());
                for (int j = 0; j < NB; j++) {
                    std::vector<std::pair<int, int>> mine;
                    for (size_t qi = 0; qi < cols.size(); qi++)
                        if (cols[qi].first == j) mine.emplace_back(int(qi), cols[qi].second);
                    if (mine.empty()) continue;
                    Key t = ckey(k_.c, g + kofs[j]);
                    int at = -1;
                    for (size_t e = 0; e < Cq.size(); e++)
                        if (Cq[e].j == j && keq(Cq[e].t, t)) { at = int(e); break; }
                    std::vector<cd> base = at >= 0 ? Cq[at].m : std::vector<cd>(K, cd(0.0, 0.0));
                    std::vector<cd> m(size_t(Bq) * K);
                    for (int b = 0; b < Bq; b++)
                        for (int k = 0; k < K; k++) m[size_t(b) * K + k] = base[k];
                    for (auto& qk : mine)
                        for (int b = 0; b < 4; b++) m[size_t(4 * qk.first + b) * K + qk.second] += cd(es[b], 0.0);
                    if (at >= 0) { Cq[at].m = std::move(m); Cq[at].nb = Bq; }
                    else Cq.push_back(Entry{j, t, Bq, std::move(m)});
                }
                k_.run(Cq, nullptr, top_g);
                return project(k_.c, k_.out, g, Bq, P_.gfam, P_.dE, P_.W, false);
            };
            tn::fam_matrix(st, K, NB, nn, S_.e1, c_.R, T.h, T.direct_chunk, rows4, famL[kf(g)]);
            std::fprintf(stderr, "mk: the family map at %.6f (%s) built in %.0f s\n", g, st.label.c_str(),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    auto one = [&](size_t i, Kernel& kc) {
        auto t0 = std::chrono::steady_clock::now();
        const tn::Seed& seed = dirs[i].second;
        Res& r = res[i];
        std::vector<Entry>& dC = r.dC;
        if (seed.kind == "data") {
            int j;
            double t;
            collab_of(seed.name, j, t);
            std::vector<cd> mv(K, cd(0.0, 0.0));
            mv[0] = 1.0;
            dC.push_back(Entry{j, Key{t, true}, 1, mv});
        }
        if (seed.kind != "gauge") {
            for (int tp = 0; tp < T.passes; tp++) {
                std::vector<Entry> prev = dC;
                for (double g : order_last_) {
                    const tn::StepState& st = steps_.at(kf(g));
                    double sc = dmax(dC);
                    double top_g = T.top_local ? std::min(c_.top, std::ceil(g - 1e-9) + T.margin) : c_.top;
                    std::vector<Entry> Cpt = with_grade(kc.c, C_final_, st.X, g, kofs, S_.e1, c_.R);
                    std::vector<Entry> Cs = strip_g(kc.c, C_final_, g, kofs);
                    auto drows = [&](const Vec* dX) -> Vec {
                        std::vector<Entry> dCf = dX ? with_grade(kc.c, dC, *dX, g, kofs, S_.e1, c_.R) : strip_g(kc.c, dC, g, kofs);
                        const std::vector<Entry>& Cb = dX ? Cpt : Cs;
                        Vec out(size_t(K) * NB, cd(0.0, 0.0));
                        double mx = dmax(dCf);
                        if (mx == 0.0) return out;
                        if (T.jvp) {
                            kc.run(Cb, &dCf, top_g);
                            out = project(kc.c, kc.dout, g, 1, P_.gfam, P_.dE, P_.W, false);
                        } else {
                            // _dfd_b: the four-point stencil [8 (f(h) - f(-h)) - (f(2h) - f(-2h))] / 12 h, its four evaluations batched (_addCb)
                            double h = T.h / mx;
                            const double es[4] = {h, -h, 2 * h, -2 * h};
                            std::vector<Entry> Cq = Cb;
                            for (auto& d : dCf) {
                                int at = -1;
                                for (size_t e = 0; e < Cq.size(); e++)
                                    if (Cq[e].j == d.j && keq(Cq[e].t, d.t)) { at = int(e); break; }
                                std::vector<cd> base = at >= 0 ? Cq[at].m : std::vector<cd>(K, cd(0.0, 0.0));
                                std::vector<cd> m(4 * size_t(K));
                                for (int b = 0; b < 4; b++)
                                    for (int k = 0; k < K; k++) m[size_t(b) * K + k] = base[k] + ct::mul(cd(es[b], 0.0), d.m[k]);
                                if (at >= 0) { Cq[at].m = std::move(m); Cq[at].nb = 4; }
                                else Cq.push_back(Entry{d.j, d.t, 4, std::move(m)});
                            }
                            kc.run(Cq, nullptr, top_g);
                            Vec v = project(kc.c, kc.out, g, 4, P_.gfam, P_.dE, P_.W, false);
                            size_t P = size_t(K) * NB;
                            for (size_t q = 0; q < P; q++)
                                out[q] = ct::div(ct::mul(cd(8.0, 0.0), v[q] - v[P + q]) - (v[2 * P + q] - v[3 * P + q]), cd(12 * h, 0.0));
                        }
                        for (int m = 0; m < K; m++)
                            for (int j = 0; j < NB; j++)
                                if (!st.act[j]) out[size_t(m) * NB + j] = 0.0;
                        if (const char* dg = std::getenv("MK_TAN_DROWS")) {   // diagnostic: the other mode's rows' derivative at centre g
                            if (std::abs(g - std::atof(dg)) < 1e-6 && !T.jvp && dX && std::getenv("MK_TAN_DUMP")) {
                                static int nd = 0;
                                std::string b = std::string(std::getenv("MK_TAN_DUMP")) + std::to_string(nd++);
                                auto dump = [&](const std::vector<Entry>& D, const std::string& nm) {
                                    std::vector<int> col, isi;
                                    std::vector<double> key;
                                    Vec mom;
                                    for (auto& en : D) {
                                        col.push_back(en.j); key.push_back(en.t.v); isi.push_back(en.t.isint);
                                        for (int k = 0; k < K; k++) mom.push_back(en.m[k]);
                                    }
                                    npy_write(b + nm + "_col.npy", col.data(), "<i4", {long(col.size())}, 4);
                                    npy_write(b + nm + "_key.npy", key.data(), "<f8", {long(key.size())}, 8);
                                    npy_write(b + nm + "_isint.npy", isi.data(), "<i4", {long(isi.size())}, 4);
                                    npy_write(b + nm + "_mom.npy", mom.data(), "<c16", {long(col.size()), K}, 16);
                                };
                                dump(Cb, "_Cb");
                                dump(dCf, "_dC");
                                npy_write(b + "_cent.npy", kc.c.cent.data(), "<f8", {long(kc.c.cent.size())}, 8);
                                std::ofstream(b + "_top.txt") << Json::num(top_g) << "\n";
                                Kernel kd = kc;
                                kd.jvp_check(Cb, dCf, top_g, 1e-7 / mx);
                            }
                            if (std::abs(g - std::atof(dg)) < 1e-6 && !T.jvp) {
                                kc.run(Cb, &dCf, top_g);
                                Vec jv = project(kc.c, kc.dout, g, 1, P_.gfam, P_.dE, P_.W, false);
                                for (int j = 0; j < NB; j++) {
                                    double a = 0.0, b = 0.0, d = 0.0;
                                    for (int m = 0; m < K; m++) {
                                        size_t q = size_t(m) * NB + j;
                                        if (!st.act[j]) continue;
                                        a = std::max(a, std::abs(out[q])); b = std::max(b, std::abs(jv[q])); d = std::max(d, std::abs(out[q] - jv[q]));
                                    }
                                    if (d > 1e-8 * std::max(a, 1e-300)) std::fprintf(stderr, "drows g %.6f %s col %s FD %.3e JVP %.3e diff %.3e\n", g, dX ? "dX" : "base", S_.collab[j].c_str(), a, b, d);
                                }
                            }
                        }
                        return out;
                    };
                    tn::StepOut so;
                    auto fmi = famL.find(kf(g));
                    tn::tan_step(st, seed, sc, T.ctol, T.iters, K, NB, nn, S_.e1, c_.R, drows, so, fmi == famL.end() ? nullptr : &fmi->second);
                    if (std::getenv("MK_TAN_TRACE")) {                  // diagnostic: per step |dX| (moments) and the update history
                        auto mo = ct::moments_of(so.dX, K, NB, S_.e1, c_.R);
                        double m = 0.0;
                        for (auto& z : mo) m = std::max(m, std::abs(z));
                        std::string hs;
                        for (double x : so.hh) hs += " " + Json::num(x);
                        std::fprintf(stderr, "tan %s pass %d g %.9f int %d rk %d |dX| %.6e sc %.3e hist%s\n", dirs[i].first.c_str(), tp, g, int(st.isint),
                                     st.rk, m, sc, hs.c_str());
                    }
                    r.step_max.push_back(so.hh.empty() ? 0.0 : *std::max_element(so.hh.begin(), so.hh.end()));
                    r.step_last.push_back(so.hh.empty() ? 0.0 : so.hh.back());
                    r.step_last_abs.push_back(so.hha.empty() ? 0.0 : so.hha.back());
                    dC = with_grade(kc.c, dC, so.dX, g, kofs, S_.e1, c_.R);
                }
                double ch = 0.0;
                for (auto& en : dC) {
                    const Entry* pv = nullptr;
                    for (auto& q : prev)
                        if (q.j == en.j && keq(q.t, en.t)) { pv = &q; break; }
                    for (size_t k = 0; k < en.m.size(); k++) ch = std::max(ch, std::abs(en.m[k] - (pv ? pv->m[k] : cd(0.0, 0.0))));
                }
                r.ph.push_back(ch / std::max(dmax(dC), 1e-300));
                size_t n = r.ph.size();
                bool stall = tp > 2 && r.ph[n - 1] > 0.5 * r.ph[n - 2] && r.ph[n - 2] > 0.5 * r.ph[n - 3];
                if (tp && (r.ph.back() < T.tol || (!T.no_stall && stall))) break;
                // the step histories of the last pass only (n2a_nl keeps every pass's; the last pass's are the converged ones)
                if (tp + 1 < T.passes) { r.step_max.clear(); r.step_last.clear(); r.step_last_abs.clear(); }
            }
        }
        // the outputs' derivative: the four-point stencil of fvec in the direction (dC, d gauge), h = --tan-h / the direction's largest
        // coefficient (n2a_nl's one batched call; here four evaluations)
        if (outputs) {
            std::map<int, double> gc;
            if (seed.kind == "gauge") gc[std::stoi(seed.name.substr(1))] = 1.0;
            double mx = dmax(dC);
            for (auto& q : gc) mx = std::max(mx, std::abs(q.second));
            double h = T.h / std::max(mx, 1e-300);
            std::map<int, double> gall;
            for (auto& q : c_.gauge) gall[q.first] = q.second;
            for (auto& q : gc) gall.emplace(q.first, 0.0);
            auto f = [&](double e) {
                std::vector<Entry> Cx = C_final_;
                for (auto& d : dC) {
                    int at = -1;
                    for (size_t q = 0; q < Cx.size(); q++)
                        if (Cx[q].j == d.j && keq(Cx[q].t, d.t)) { at = int(q); break; }
                    std::vector<cd> m(K);
                    for (int k = 0; k < K; k++) m[k] = (at >= 0 ? Cx[at].m[k] : cd(0.0, 0.0)) + ct::mul(cd(e, 0.0), d.m[k]);
                    if (at >= 0) Cx[at].m = m;
                    else Cx.push_back(Entry{d.j, d.t, 1, m});
                }
                std::vector<std::pair<int, double>> ga;
                for (auto& q : gall) ga.emplace_back(q.first, q.second + e * (gc.count(q.first) ? gc[q.first] : 0.0));
                return fvec(kc, Cx, ga, cents);
            };
            auto v1 = f(h), v2 = f(-h), v3 = f(2 * h), v4 = f(-2 * h);
            for (size_t q = 0; q < v1.size(); q++) {
                cd z = ct::div(ct::mul(cd(8.0, 0.0), v1[q] - v2[q]) - (v3[q] - v4[q]), cd(12 * h, 0.0));
                r.dv.push_back(z.real());
                r.dv_imag = std::max(r.dv_imag, std::abs(z.imag()));
            }
        }
        r.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    std::atomic<size_t> next(0);
    auto worker = [&]() {
        Kernel kc = k_;
        for (size_t i; (i = next++) < dirs.size();) {
            try {
                Kernel ki = kc;
                one(i, ki);
            } catch (std::exception& e) {
                res[i].err = e.what();
            }
        }
    };
    std::vector<std::thread> th;
    int nt = std::max(1, std::min<int>(T.threads, int(dirs.size())));
    for (int q = 0; q < nt; q++) th.emplace_back(worker);
    for (auto& t : th) t.join();
    mkdir((out_ + "/tangent").c_str(), 0755);
    Json J;
    J.open('{');
    J.kv("mode", Json::str(T.jvp ? "jvp" : "fd"));
    J.kv("tol", Json::num(T.tol));
    J.kv("passes_max", std::to_string(T.passes));
    J.kv("top_local", T.top_local ? "true" : "false");
    J.kv("no_stall", T.no_stall ? "true" : "false");
    J.kv("centres", Json::nums(cents));
    J.kv("base_imag", Json::num(base_imag));
    if (outputs) npy_write(out_ + "/fit_base.npy", base.data(), "<f8", {long(base.size())}, 8);
    if (outputs && !S_.condlab.empty()) {
        // fit.json: the outputs' labels and values as n2a_nl fit_out writes them (labels, base) -- n3b_linthroat reads the grade-0 columns'
        // runs from it (the hybrid route, research round 1028)
        auto fmt = [](const char* f, double x) { char b[64]; std::snprintf(b, sizeof b, f, x); return std::string(b); };
        std::vector<std::string> lab;
        for (int gg = 1; gg <= c_.gdata; gg++)
            for (int m = 0; m < 2; m++)
                for (int q = 0; q < P_.nn; q++) lab.push_back("eq row g" + std::to_string(gg) + " m" + std::to_string(m) + " " + S_.collab[q]);
        for (double gc : cents) {
            bool eq = std::abs(gc - std::nearbyint(gc)) < 1e-9 && gc >= 1 - 1e-9 && gc <= 2 + 1e-9;
            for (auto& f : fams_)
                if (std::abs(f.second.v - gc) < 1e-9) eq = true;
            for (int m = 0; m < 2; m++)
                for (int q = 0; q < P_.nc; q++) lab.push_back(std::string(eq ? "eq" : "mon") + " cond " + fmt("%.6f", gc) + " m" + std::to_string(m) + " " + S_.condlab[q]);
        }
        for (size_t i = size_t(c_.gdata); i < cents.size(); i++)
            for (int m = 0; m < 2; m++)
                for (int q = 0; q < P_.nn; q++) lab.push_back("monrow " + fmt("%.6f", cents[i]) + " m" + std::to_string(m) + " " + S_.collab[q]);
        if (S_.g0rows)
            for (int gg = -1; gg <= 0; gg++)
                for (int m = 0; m < 2; m++)
                    for (int q = 0; q < P_.nn; q++) lab.push_back("eq row g" + std::to_string(gg) + " m" + std::to_string(m) + " " + S_.collab[q]);
        for (size_t q = 0; q < S_.md_key.size() / 3; q++)
            lab.push_back("mk " + std::to_string(q) + " " + std::to_string(S_.md_key[3 * q]) + " " + std::to_string(S_.md_key[3 * q + 1]) + " " +
                          std::to_string(S_.md_key[3 * q + 2]));
        if (lab.size() != base.size()) throw std::runtime_error("mk: fit labels " + std::to_string(lab.size()) + " vs outputs " + std::to_string(base.size()));
        Json F;
        F.open('{');
        std::string ls = "[";
        for (size_t i = 0; i < lab.size(); i++) ls += (i ? "," : "") + Json::str(lab[i]);
        F.kv("labels", ls + "]");
        F.kv("base", Json::nums(base));
        F.kv("base_imag", Json::num(base_imag));
        F.kv("centres", Json::nums(cents));
        F.kv("native", "true");
        F.close('}');
        std::ofstream(out_ + "/fit.json") << F.s.str();
    }
    J.okey("directions");
    J.open('[');
    for (size_t i = 0; i < dirs.size(); i++) {
        const Res& r = res[i];
        std::vector<int> col, isi;
        std::vector<double> key;
        Vec mom;
        for (auto& en : r.dC) {
            col.push_back(en.j); key.push_back(en.t.v); isi.push_back(en.t.isint);
            for (int k = 0; k < K; k++) mom.push_back(en.m[k]);
        }
        std::string b = out_ + "/tangent/" + std::to_string(i);
        npy_write(b + "_col.npy", col.data(), "<i4", {long(col.size())}, 4);
        npy_write(b + "_key.npy", key.data(), "<f8", {long(key.size())}, 8);
        npy_write(b + "_isint.npy", isi.data(), "<i4", {long(isi.size())}, 4);
        npy_write(b + "_mom.npy", mom.data(), "<c16", {long(col.size()), K}, 16);
        J.sep();
        J.open('{');
        J.kv("name", Json::str(dirs[i].first));
        J.kv("index", std::to_string(i));
        J.kv("pass_change", Json::nums(r.ph));
        J.kv("step_max", Json::nums(r.step_max));
        J.kv("step_last", Json::nums(r.step_last));
        J.kv("step_last_abs", Json::nums(r.step_last_abs));
        J.kv("secs", Json::num(r.secs));
        J.kv("dv_imag", Json::num(r.dv_imag));
        if (outputs) npy_write(b + "_dv.npy", r.dv.data(), "<f8", {long(r.dv.size())}, 8);
        J.kv("error", r.err.empty() ? "null" : Json::str(r.err));
        J.close('}');
    }
    J.close(']');
    J.close('}');
    std::ofstream(out_ + "/tangent.json") << J.s.str();
}

}  // namespace en
}  // namespace mk

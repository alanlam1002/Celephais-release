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
#include "mk_cluster.hpp"
#include "mk_json.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <tuple>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
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
    // stage (ii) port, items 2-3 (n2a_nl N2A_CLUSTER, N2A_EPS_S / N2A_XMIN, --no-spur-orth): the clusters, the storage merge, the ii8 projection
    bool cluster = false, spur_orth = true;
    double eps_s = 0.0, xmin = 1e-5;
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
    k.c.gauge_reg = B.geti("gauge_reg", 0) != 0; k.c.reg_tol = B.getd("reg_tol", 1e-3);    // stage (ii) item 1 (n2a_nl N2A_GAUGE_REG / _REG_TOL)
    c.cluster = B.geti("cluster", 0) != 0; c.spur_orth = B.geti("spur_orth", 1) != 0;       // stage (ii) items 2-3
    c.eps_s = B.getd("eps_s", 0.0); c.xmin = B.getd("xmin", 1e-5);
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
    // G3 (i) (spinning_trumpet NOTES_M4_DESIGN.md section 19): the matching keys' own n-dependence at a FIXED series -- d mk / dn through the
    // gauge Lie flow (its generator's keys l / n move: a log-moment shift at the same key) and the evaluation at r_m (x_m = r_m^n)
    std::vector<double> mkey_dn(Kernel& kc, const std::vector<Entry>& Cx, const std::vector<std::pair<int, double>>& gauge);
    // after run(): the exact d mk / dn on the final series and its Richardson check (n moved by +-h, +-2h, the series frozen) -> OUT/mk_dn.json
    void mk_dn_check(double h);

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
        if (const auto* pl = Ctx::kv_cget(k_.c.clus, g))         // stage (ii): a cluster key -- its storage poles' own columns too
            if (!k_.c.noclus) return cl::strip(k_.c, C, g, *pl, P_.kofs);
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
    Vec moments(const Vec& X) { return ct::moments_of(X, k_.c.K, P_.nn, S_.e1, Rcur_); }
    dt::Eval eval(const std::vector<Entry>& base, double g) {
        dt::Eval ev;
        Kernel* k = &k_;
        const Proj* P = &P_;
        std::vector<int> xr = xrows_;
        Vec e1 = S_.e1;
        double R = Rcur_, top = c_.top;
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
        double R = Rcur_, top = c_.top;
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
    double Rcur_ = 0.0;                                             // stage (ii): the current step's contour radius (a cluster's larger)
    // M3: the forward's last pass -- each centre's step state, the centre order, the final series, the families
    std::map<double, tn::StepState> steps_;
    std::vector<double> order_last_;
    std::vector<Entry> C_final_;
    std::vector<std::pair<std::string, Key>> fams_;
    std::vector<std::pair<std::string, Key>> gfams_;                // stage (ii) item 1: the registered gauge roots ('gauge lL')
    // stage (ii) item 4: the spurious members (storage member, label, cluster key), sorted -- their l = 0 conditions are fit equations
    std::vector<std::tuple<double, std::string, double>> spur_r_;
    std::map<std::string, std::array<double, 3>> spur_;
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
    std::vector<std::pair<std::string, Key>> gfams;                 // fams_gauge (stage (ii) item 1, k_.c.gauge_reg): datum pinned to 0
    // the families' towers (n2a_nl towers(): the lapse families' -- exactly registered with the gauge roots -- then the gauge roots')
    auto towers_all = [&]() {
        std::vector<Key> fk, gk_;
        for (auto& f : fams) fk.push_back(f.second);
        for (auto& f : gfams) gk_.push_back(f.second);
        auto t = fm::towers(k_.c, fk, c_.gmax, k_.c.gauge_reg);
        auto t2 = fm::towers(k_.c, gk_, c_.gmax, true);
        t.insert(t.end(), t2.begin(), t2.end());
        return t;
    };
    // ---- stage (ii) port, items 2-3 (n2a_nl N2A_CLUSTER=1: spec sections 23-32) ----
    const bool CL = c_.cluster && k_.c.exact && k_.c.Rm > 0;
    const double R0 = c_.R;
    const double EPS_S = c_.eps_s > 0 ? c_.eps_s
        : std::min(k_.c.Rm > 0 ? k_.c.Rm : 1.0, std::pow(1e-12 * k_.c.fact[K - 1] * K, 1.0 / K) / std::abs(std::log(c_.xmin)));
    std::map<std::string, std::array<double, 3>> SPUR;            // 'spurious S' -> (root, storage member, cluster key)
    std::set<double> SPUR_SINGLE, ROUTED;                           // (kf keys)
    auto rkey = [&](double x) { double xi = std::nearbyint(x); return std::abs(x - xi) < 1e-9 ? Key{xi, true} : k_.c.xkey(x); };
    auto is_root = [&](double x) {                                  // an integer, a family (lapse / gauge) or its tower, a spurious member
        if (std::abs(x - std::nearbyint(x)) < 1e-9) return true;
        for (auto* fv : {&fams, &gfams})
            for (auto& f : *fv) {
                double k = std::nearbyint(x - f.second.v);
                if (k >= 0 && std::abs(x - f.second.v - k) < 1e-9) return true;
            }
        for (auto& sp : SPUR)
            if (std::abs(x - sp.second[0]) < 1e-9 || std::abs(x - sp.second[1]) < 1e-9) return true;
        return false;
    };
    auto reg_roots = [&]() {                                        // the registered roots in (gdata, gmax]: integers, families
        std::set<double> o;
        for (int x = c_.gdata + 1; x <= int(c_.gmax); x++) o.insert(double(x));
        for (auto* fv : {&fams, &gfams})
            for (auto& f : *fv)
                if (c_.gdata < f.second.v && f.second.v <= c_.gmax + 1e-9) o.insert(f.second.v);
        return std::vector<double>(o.begin(), o.end());
    };
    auto root_clusters = [&]() {                                    // the roots grouped transitively by 2 R_m (groups of two or more)
        std::set<double> rs;
        for (int x = c_.gdata + 1; x <= int(c_.gmax); x++) rs.insert(double(x));
        for (double c0 : k_.c.cent)
            if (c_.gdata < c0 && c0 <= c_.gmax + 1e-9 && is_root(c0) && !SPUR_SINGLE.count(kf(c0))) rs.insert(c0);
        std::vector<std::vector<double>> out;
        std::vector<double> grp;
        for (double r : rs) {
            if (!grp.empty() && r - grp.back() >= 2 * k_.c.Rm) {
                if (grp.size() > 1) out.push_back(grp);
                grp.clear();
            }
            grp.push_back(r);
        }
        if (grp.size() > 1) out.push_back(grp);
        return out;
    };
    auto spur_label = [&](double x) {
        int bL = -1;
        double bd = 1e300;
        for (auto& f : fams) {
            int L = std::stoi(f.first.substr(f.first.rfind('l') + 1));
            double dd = f.second.v - x;
            if (L >= 4 && 0.01 < dd && dd < 0.15 && dd < bd) { bd = dd; bL = L; }
        }
        if (bL >= 0) return "spurious i" + std::to_string(bL);
        bd = 1e300;
        for (int L = 2; L < 40; L += 2) {
            double dd = 1 + L / k_.c.n - x;
            if (0 < dd && dd < 0.3 && dd < bd) { bd = dd; bL = L; }
        }
        if (bL >= 0) return "spurious ii" + std::to_string(bL);
        char b[64];
        std::snprintf(b, sizeof b, "spurious %.4f", x);
        return std::string(b);
    };
    // ---- item 5 (research round 1048, spec section 31): identities by continuation, the frozen decisions ----
    // N2A_IDENT_PREV=FILE[,FILE]: the previous J steps' decision files -- their roots predicted at this J (one: its positions; two or more:
    // linear in J^2 through the last two) and matched within a quarter of the smallest spacing to the other predicted roots
    std::vector<std::pair<std::string, double>> PRED;
    {
        std::vector<js::Val> idp;
        if (const char* e = std::getenv("N2A_IDENT_PREV")) {
            std::string sv(e);
            size_t a0 = 0;
            while (a0 < sv.size()) {
                size_t b0 = sv.find(',', a0);
                if (b0 == std::string::npos) b0 = sv.size();
                if (b0 > a0) idp.push_back(js::load(sv.substr(a0, b0 - a0)));
                a0 = b0 + 1;
            }
        }
        std::stable_sort(idp.begin(), idp.end(), [](const js::Val& x, const js::Val& y) { return x.get("J")->n < y.get("J")->n; });
        if (!idp.empty()) {
            const js::Val& b_ = *idp.back().get("roots");
            bool lin = idp.size() > 1 && std::abs(idp.back().get("J")->n - idp[idp.size() - 2].get("J")->n) > 1e-12;
            double w_ = 0.0;
            const js::Val* a_ = nullptr;
            if (lin) {
                a_ = idp[idp.size() - 2].get("roots");
                double Ja = idp[idp.size() - 2].get("J")->n, Jb = idp.back().get("J")->n;
                w_ = (c_.J * c_.J - Ja * Ja) / (Jb * Jb - Ja * Ja);
            }
            for (auto& kv : b_.o) {
                double v = kv.second.n;
                if (lin && a_->has(kv.first)) v = (1 - w_) * a_->get(kv.first)->n + w_ * v;
                PRED.emplace_back(kv.first, v);
            }
            for (int i = 1; i <= int(c_.gmax) + 2; i++) {
                std::string k0 = "int " + std::to_string(i);
                bool have = false;
                for (auto& p0 : PRED)
                    if (p0.first == k0) { p0.second = i; have = true; }
                if (!have) PRED.emplace_back(k0, double(i));
            }
            std::printf("  [identity] the roots continued from %d decision file(s)\n", int(idp.size()));
        }
    }
    struct IdRec { double pos; std::string rule; bool has_margin; double margin; };
    std::vector<std::pair<std::string, IdRec>> IDLOG;               // label -> the last naming (insertion-ordered)
    auto idlog_set = [&](const std::string& k0, const IdRec& r) {
        for (auto& e : IDLOG)
            if (e.first == k0) { e.second = r; return; }
        IDLOG.emplace_back(k0, r);
    };
    auto root_label = [&](double er) {                              // spec section 31: a root of a step's operator named
        if (!PRED.empty()) {
            size_t ib = 0;
            for (size_t i = 1; i < PRED.size(); i++)
                if (std::abs(PRED[i].second - er) < std::abs(PRED[ib].second - er)) ib = i;
            double sp = 1e300;
            for (size_t i = 0; i < PRED.size(); i++)
                if (i != ib) sp = std::min(sp, std::abs(PRED[ib].second - PRED[i].second));
            if (std::abs(er - PRED[ib].second) < sp / 4) {
                idlog_set(PRED[ib].first, IdRec{er, "continued", true, sp / 4 - std::abs(er - PRED[ib].second)});
                return PRED[ib].first;
            }
        }
        std::string rl = PRED.empty() ? "first" : "new";
        std::string bk;
        double bp = 0.0, bd = 1e300;
        for (int i = 1; i <= int(c_.gmax) + 2; i++)
            if (std::abs(double(i) - er) < bd) { bd = std::abs(double(i) - er); bk = "int " + std::to_string(i); bp = i; }
        for (auto* fv : {&fams, &gfams})
            for (auto& f : *fv)
                if (std::abs(f.second.v - er) < bd) { bd = std::abs(f.second.v - er); bk = f.first; bp = f.second.v; }
        if (bd < R0 / 4) {
            idlog_set(bk, IdRec{er, rl, true, R0 / 4 - std::abs(er - bp)});
            return bk;
        }
        bk = spur_label(er);
        idlog_set(bk, IdRec{er, rl, false, 0.0});
        return bk;
    };
    // N2A_FREEZE=FILE: absent -- decided at this point (the J step's first) and written; present -- replayed by identity
    std::string FZP = std::getenv("N2A_FREEZE") ? std::getenv("N2A_FREEZE") : "";
    const double ID_TOL = std::getenv("N2A_ID_TOL") ? std::atof(std::getenv("N2A_ID_TOL")) : 1e-4;
    // the storage rule (n2a_nl's): 1061 (default) -- the absolute threshold 1e-15 and, in replay, (c) with the root exemption and the eps_s
    // merge, and a product new within the J step routed to the nearest registered root (never stepped); pre1059; 1059 / 1060 (withdrawn)
    const std::string STOR_RULE = std::getenv("N2A_STOR_RULE") ? std::getenv("N2A_STOR_RULE") : "1061";
    if (STOR_RULE != "pre1059" && STOR_RULE != "1059" && STOR_RULE != "1060" && STOR_RULE != "1061")
        throw std::runtime_error("N2A_STOR_RULE: 1061 (default), pre1059, 1059 or 1060");
    const bool STOR1059 = STOR_RULE == "1059" || STOR_RULE == "1060" || STOR_RULE == "1061";   // (c) in replay
    // research round 1062 (option (a)): every labelled cluster root's null component projected to its datum (n2a_nl N2A_ROOT_PROJ: all, the
    // default; spur -- the spurious roots only, rounds 1049-1061)
    const bool ROOT_PROJ_ALL = !(std::getenv("N2A_ROOT_PROJ") && std::string(std::getenv("N2A_ROOT_PROJ")) == "spur");
    const bool SRC_STRUCT = !(std::getenv("N2A_SRC_STRUCT") && std::string(std::getenv("N2A_SRC_STRUCT")) == "0");   // rule (a) off: 0
    const double SRC_REL = std::getenv("N2A_SRC_REL") ? std::atof(std::getenv("N2A_SRC_REL")) : 1e-12;   // research round 1059
    js::Val FZv;
    bool FZ = false;
    if (CL && !FZP.empty() && std::ifstream(FZP)) { FZv = js::load(FZP); FZ = true; }
    struct FzLog { std::string site, frozen, own, margin; int ps; };
    std::vector<FzLog> FZ_LOG;
    int PS_ = 0;
    auto fz_note = [&](const std::string& site, const std::string& frozen, const std::string& own, const std::string& margin) {
        FZ_LOG.push_back(FzLog{site, frozen, own, margin, PS_});
        if (frozen != own) std::printf("    [freeze] %s: frozen %s, the point's own %s (margin %s)\n", site.c_str(), frozen.c_str(), own.c_str(), margin.c_str());
    };
    auto fmtm = [](double m) { char b[48]; std::snprintf(b, sizeof b, "%.6g", m); return std::string(b); };
    // research round 1065 (spec section 34): an operator root is REAL when |imag| < 1e-6 x its distance to the nearest other root (a complex
    // pair fails by construction, a simple root's round-off passes); in replay a root the decisions file names keeps its realness, its
    // imaginary part logged (N2A_REAL_RULE=pre1065: |imag| < 1e-9)
    const bool REAL_PRE = std::getenv("N2A_REAL_RULE") && std::string(std::getenv("N2A_REAL_RULE")) == "pre1065";
    auto root_real = [&](size_t i, const std::vector<cd>& all) {
        cd e = all[i];
        if (REAL_PRE) return std::abs(e.imag()) < 1e-9;
        double d = 1e300;
        for (size_t j = 0; j < all.size(); j++)
            if (j != i) d = std::min(d, std::abs(e - all[j]));
        bool own = std::abs(e.imag()) < 1e-6 * d;
        if (FZ && FZv.get("roots") && !FZv.get("roots")->o.empty()) {
            std::vector<std::pair<std::string, double>> fr;
            for (auto& kv : FZv.get("roots")->o) fr.emplace_back(kv.first, kv.second.n);
            std::stable_sort(fr.begin(), fr.end(), [&](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
                return std::abs(a.second - e.real()) < std::abs(b.second - e.real()); });
            double sp = 1e300;
            for (size_t j = 1; j < fr.size(); j++) sp = std::min(sp, std::abs(fr[0].second - fr[j].second));
            if (std::abs(e.real() - fr[0].second) < sp / 4) {
                char b[160], m[48];
                std::snprintf(b, sizeof b, "root realness %s (%.6f)", fr[0].first.c_str(), e.real());
                std::snprintf(m, sizeof m, "%.1e i", e.imag());
                fz_note(b, "real", own ? "real" : "complex", m);
                return true;
            }
        }
        return own;
    };
    // this point's decisions (the last pass's; the routings cumulative)
    std::set<std::string> FZD_steps;
    std::vector<std::pair<std::string, std::string>> FZD_routed;
    struct FzClus { std::vector<std::string> members; double key; };
    std::vector<std::pair<std::string, FzClus>> FZD_clusters;
    struct FzSp { std::string status; double root; std::string storage; bool has_storage; };
    std::vector<std::pair<std::string, std::vector<std::pair<std::string, FzSp>>>> FZD_spur;
    struct FzSt { std::vector<std::string> poles, merged; };
    std::vector<std::pair<std::string, FzSt>> FZD_storage;
    std::map<double, std::string> STEPLAB;                          // kf(a stage (ii) step's key) -> the step's identity
    auto is_spur_ = [](const std::string& l) { return l.rfind("spurious", 0) == 0; };
    auto bases = [&]() {                                            // the constituents' current positions (pole_label's)
        std::vector<std::pair<std::string, double>> b;
        auto put = [&](const std::string& k0, double v) {
            for (auto& e : b)
                if (e.first == k0) { e.second = v; return; }
            b.emplace_back(k0, v);
        };
        for (auto* fv : {&fams, &gfams})
            for (auto& f : *fv) put(f.first, f.second.v);
        if (FZ)
            for (auto& kv : FZv.get("roots")->o)
                if (is_spur_(kv.first)) put(kv.first, kv.second.n);
        for (auto& sp : SPUR) put(sp.first, sp.second[0]);
        std::sort(b.begin(), b.end());                              // (Python: sorted(b_) -- by label)
        return b;
    };
    auto pole_label = [&](double c0) -> std::string {               // a registered centre's identity
        char buf[96];
        if (std::abs(c0 - std::nearbyint(c0)) < 1e-9) { std::snprintf(buf, sizeof buf, "int %d", int(std::nearbyint(c0))); return buf; }
        auto b = bases();
        for (auto& kv : b) {
            double kk = std::nearbyint(c0 - kv.second);
            if (std::abs(c0 - kv.second - kk) < 1e-9) {
                if (kk == 0) return kv.first;
                std::snprintf(buf, sizeof buf, "%s %+d", kv.first.c_str(), int(kk));
                return buf;
            }
        }
        int nb_ = int(b.size());
        for (double tol : {1e-7, ID_TOL})
            for (int m = 1; m <= 3; m++) {
                bool have = false;
                double bdv = 0.0;
                std::vector<int> bc;
                int bsh = 0;
                std::vector<int> idx(m, 0);
                while (true) {                                      // combinations with replacement, lexicographic (itertools' order)
                    double sm = 0.0;
                    for (int q : idx) sm += b[q].second;
                    double d = c0 - sm, dr = std::abs(d - std::nearbyint(d));
                    if (dr < tol && (!have || dr < bdv)) { have = true; bdv = dr; bc = idx; bsh = int(std::nearbyint(d)); }
                    int k = m - 1;
                    while (k >= 0 && idx[k] == nb_ - 1) k--;
                    if (k < 0) break;
                    idx[k]++;
                    for (int q = k + 1; q < m; q++) idx[q] = idx[k];
                }
                if (have) {
                    std::string t;
                    std::vector<int> uq = bc;
                    uq.erase(std::unique(uq.begin(), uq.end()), uq.end());
                    for (size_t q = 0; q < uq.size(); q++) {
                        int cnt = int(std::count(bc.begin(), bc.end(), uq[q]));
                        t += (q ? " + " : "") + (cnt > 1 ? std::to_string(cnt) + " " + b[uq[q]].first : b[uq[q]].first);
                    }
                    if (bsh == 0) return t;
                    std::snprintf(buf, sizeof buf, "%s %+d", t.c_str(), bsh);
                    return buf;
                }
            }
        std::snprintf(buf, sizeof buf, "x%.9f", c0);
        return buf;
    };
    auto pole_pos = [&](std::string lb) {                           // an identity resolved to its current position
        if (lb[0] == 'x') return std::stod(lb.substr(1));
        if (lb.rfind("int ", 0) == 0) return double(std::stoi(lb.substr(4)));
        auto b = bases();
        int sh = 0;
        size_t sp = lb.rfind(' ');
        if (sp != std::string::npos && (lb[sp + 1] == '+' || lb[sp + 1] == '-') && lb.size() > sp + 2 && std::isdigit(lb[sp + 2])) {
            sh = std::stoi(lb.substr(sp + 1));
            lb = lb.substr(0, sp);
        }
        double s0 = sh;
        size_t a0 = 0;
        while (a0 <= lb.size()) {
            size_t b0 = lb.find(" + ", a0);
            std::string t = lb.substr(a0, b0 == std::string::npos ? std::string::npos : b0 - a0);
            int m = 1;
            size_t s1 = t.find(' ');
            if (s1 != std::string::npos && std::all_of(t.begin(), t.begin() + long(s1), ::isdigit) && s1 > 0) { m = std::stoi(t.substr(0, s1)); t = t.substr(s1 + 1); }
            double v = std::nan("");
            for (auto& kv : b)
                if (kv.first == t) v = kv.second;
            if (std::isnan(v)) throw std::runtime_error("mk: pole_pos: unknown constituent '" + t + "'");
            s0 += m * v;
            if (b0 == std::string::npos) break;
            a0 = b0 + 3;
        }
        return s0;
    };
    auto fz_pending = [&]() {                                       // the frozen product steps and routings, visited at every point
        std::vector<Key> o2;
        if (!FZ || fams.empty()) return o2;
        std::vector<std::string> ls;
        for (auto& v : FZv.get("product_steps")->a) ls.push_back(v.s);
        for (auto& kv : FZv.get("routed")->o) ls.push_back(kv.first);
        for (auto& lb : ls) {
            double x = pole_pos(lb);
            if (c_.gdata < x && x <= c_.gmax + 1e-9) o2.push_back(k_.c.reg_exact(x));
        }
        for (auto& kv : FZv.get("routed")->o) ROUTED.insert(kf(pole_pos(kv.second.s)));
        return o2;
    };
    auto is_spur = [](const std::string& l) { return l.rfind("spurious", 0) == 0; };
    // spec section 28: the cluster's storage poles -- its member roots and every source exponent inside the contour (the square rows' pieces
    // of the stripped series at aligned |s - key| < Rr), each at its exact exponent; a source within eps_s of a pole merged into it
    struct Stor { int roots = 0, sources = 0, merged = 0, moments = 0; std::vector<double> poles; double cond = 0, svmin = 0, radius = 0; };
    std::map<double, Stor> STOR;
    auto set_storage = [&](const Ser* rr, double g, double Rr) {
        auto* cr = Ctx::kv_get(k_.c.clroots, g);
        std::vector<double> poles = *cr;
        // research round 1059 (the storage rule): every exponent of the stripped rows inside the contour a candidate with its largest piece;
        // (a) a source derived from a family (a lapse / gauge family or a spurious root, its towers and products, by identity) STRUCTURAL --
        // a source whatever its magnitude; (b) any other above 1e-12 x the step's largest piece (N2A_SRC_REL); (c) within a J step the pole
        // set frozen: a source not a frozen pole merged onto the nearest frozen pole, never promoted.  (Was: |piece| > 1e-15 absolute.)
        std::map<double, double> cand;
        for (int i = 0; i < nn; i++) {
            int fam = P_.gfam[i];
            for (auto& kv : rr[fam].d) {
                double sc = kv.first.v - ROWOFF1[fam] - P_.dE[i];
                if (!(std::abs(sc - g) < Rr)) continue;
                double mx = 0.0;
                for (auto& z : kv.second.a) mx = std::max(mx, std::abs(z));
                double k0 = std::nearbyint(sc * 1e11) / 1e11;
                auto it = cand.find(k0);
                if (it == cand.end()) cand[k0] = mx;
                else it->second = std::max(it->second, mx);
            }
        }
        double big = 0.0;
        for (auto& kv : cand) big = std::max(big, kv.second);
        if (STOR_RULE == "1060") {                                       // round 1060 (b'): the stripped rows' largest piece over all grades
            big = 0.0;
            for (int i = 0; i < nn; i++)
                for (auto& kv : rr[P_.gfam[i]].d)
                    for (auto& z : kv.second.a) big = std::max(big, std::abs(z));
        }
        std::set<double> src;
        for (auto& kv : cand) {
            std::string lb = (STOR1059 || std::getenv("MK_SRC_LOG")) ? pole_label(kv.first) : std::string();
            bool st = STOR_RULE == "1059" && SRC_STRUCT && !(lb.rfind("int ", 0) == 0 || lb.rfind("x", 0) == 0);
            bool kept = (STOR_RULE == "pre1059" || STOR_RULE == "1061") ? kv.second > 1e-15 : (st || kv.second > SRC_REL * big);
            if (kept) src.insert(kv.first);
            if (std::getenv("MK_SRC_LOG"))                          // diagnostic: every candidate and its rule
                std::printf("    [src] step %.6f source %.6f (%s): |piece| %.3e of %.3e%s\n", g, kv.first, lb.c_str(), kv.second, big,
                            st ? " structural" : kept ? " kept" : "");
        }
        int nsrc = 0;
        std::vector<double> absb;
        // research round 1048: the frozen storage of this step (by identity) -- a source whose identity is a frozen pole is a pole, a frozen
        // merge is merged, a new source takes the rule; a frozen pole with no source here is kept
        std::string sl = STEPLAB.count(kf(g)) ? STEPLAB[kf(g)] : std::string();
        const js::Val* fs = (FZ && !sl.empty()) ? FZv.get("storage")->get(sl) : nullptr;
        size_t nmem = poles.size();
        if (std::getenv("MK_DEBUG_STOR")) {                         // diagnostic: the sources read
            std::printf("    [debug storage %.6f] R %.6f, sources:", g, Rr);
            for (double s0 : src) std::printf(" %.11f", s0);
            std::printf("\n");
        }
        for (double s0 : src) {
            double d = 1e300;
            for (double p0 : poles) d = std::min(d, std::abs(s0 - p0));
            if (d < 1e-9) continue;
            std::string own = d < EPS_S ? "merged" : "pole";
            if (fs) {
                std::string lb = pole_label(s0), fz;
                for (auto& x : fs->get("poles")->a) if (x.s == lb) fz = "pole";
                if (fz.empty()) for (auto& x : fs->get("merged")->a) if (x.s == lb) fz = "merged";
                // (a root of the frozen set -- its member decision still to come in this pass -- is not a source: the point's rule)
                bool rt = FZv.get("roots") && FZv.get("roots")->has(lb);
                fz_note("storage " + sl + ": source " + lb, !fz.empty() ? fz : !STOR1059 ? "new (the point's rule)" : rt ? "a root (the point's rule)"
                        : "new (merged onto the nearest frozen pole)", own, fmtm(d - EPS_S));
                own = !fz.empty() ? fz : (rt || !STOR1059) ? own : "merged";   // research round 1059 (opt-in): never promoted
                if ((STOR_RULE == "1060" || STOR_RULE == "1061") && d < EPS_S) own = "merged";      // round 1060: the eps_s merge for any source within eps_s of a pole
            }
            if (own == "merged") { absb.push_back(s0); continue; }
            poles.push_back(s0);
            nsrc++;
        }
        if (fs)
            for (auto& x : fs->get("poles")->a) {
                double xp = pole_pos(x.s);
                bool far = true;
                for (double p0 : poles) far = far && std::abs(xp - p0) >= EPS_S;
                for (double p0 : absb) far = far && std::abs(xp - p0) >= EPS_S;
                if (far) {
                    fz_note("storage " + sl + ": source " + x.s, "pole", "absent", "None");
                    poles.push_back(xp);
                    nsrc++;
                }
            }
        if (!sl.empty()) {
            FzSt st_;
            for (size_t q = nmem; q < poles.size(); q++) st_.poles.push_back(pole_label(poles[q]));
            for (double a0 : absb) st_.merged.push_back(pole_label(a0));
            bool have = false;
            for (auto& e : FZD_storage)
                if (e.first == sl) { e.second = st_; have = true; }
            if (!have) FZD_storage.emplace_back(sl, st_);
        }
        std::vector<double> nw;
        for (double p0 : poles) nw.push_back(rkey(p0).v);
        std::sort(nw.begin(), nw.end());
        Ctx::kv_set(k_.c.clus, g, nw);
        Ctx::kv_set(k_.c.clabs, g, absb);
        std::vector<double> dmv;
        for (double p0 : nw) dmv.push_back(p0 - g);
        int NM;
        auto A = cl::clus_map(dmv, K, N, Rr, NM);
        int LK = int(nw.size()) * K;
        Vec Ac(A.size());
        for (int q = 0; q < LK; q++) {
            double cn = 0.0;
            for (int n_ = 0; n_ < NM; n_++) cn += A[size_t(n_) * LK + q] * A[size_t(n_) * LK + q];
            cn = std::max(std::sqrt(cn), 1e-300);
            for (int n_ = 0; n_ < NM; n_++) Ac[size_t(n_) * LK + q] = A[size_t(n_) * LK + q] / cn;
        }
        std::vector<double> sv;
        la::svd(Ac, NM, LK, sv);
        Stor st;
        st.roots = int(cr->size()); st.sources = nsrc; st.merged = int(absb.size()); st.poles = nw; st.moments = NM;
        st.cond = sv.front() / std::max(sv.back(), 1e-300); st.svmin = sv.back(); st.radius = Rr;
        STOR[kf(g)] = st;
        std::printf("    [cluster %.6f] storage (spec section 28): %d roots + %d source exponents (%d merged within eps_s %.1e) = %d poles x K %d = %d "
                    "unknowns from %d contour moments; the scaled map's condition %.1e (sigma_min %.1e)\n", g, st.roots, nsrc, st.merged, EPS_S,
                    int(nw.size()), K, LK, NM, st.cond, sv.back());
        std::fflush(stdout);
    };
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
        Q.manual = CL;                                              // stage (ii): n2a_nl's seen set (marked when decided, not when popped)
        std::vector<Key> lat;
        for (int g = c_.gdata + 1; g <= c_.gmax; g++) lat.push_back(Key{double(g), true});
        Q.push(lat);
        Q.push(towers_all());
        if (CL) {
            Q.push(fz_pending());
            PS_ = ps;
            FZD_steps.clear(); FZD_clusters.clear(); FZD_spur.clear(); FZD_storage.clear();   // (the routings cumulative)
        }
        std::vector<double> order;
        Key gk;
        while (Q.pop(gk)) {
            double g = gk.v;
            bool isint = gk.isint;
            if (k_.c.gauge_reg && !isint) {                         // spec section 26: a tower moved with its family (no longer registered)
                bool reg_ = false;
                for (double c_ : k_.c.cent)
                    if (std::abs(c_ - g) <= 1e-12) { reg_ = true; break; }
                if (!reg_) continue;
            }
            // ---- stage (ii) (items 2-3): the step's selection (n2a_nl's pass loop under N2A_CLUSTER=1) ----
            bool iscl = false;
            Rcur_ = c_.R;
            std::vector<double> mem;                                // a cluster's member roots (absolute)
            std::vector<cd> own_roots;                              // the cluster operator's roots within R_0 / 4 of a member
            std::vector<cd> roots_all;                              // (round 1065: the realness test's other roots -- the step operator's all)
            if (CL) {
                if (SPUR_SINGLE.count(kf(g))) {
                    // spec section 29: a root that is a member of a single product exponent's step is stepped there (stored by it)
                    std::vector<double> ks;
                    for (auto& sp : SPUR)
                        if (std::abs(sp.second[0] - g) < 1e-9 || std::abs(sp.second[1] - g) < 1e-9) ks.push_back(sp.second[2]);
                    if (ks.empty() || Q.has(rkey(ks[0]).v)) continue;
                    gk = rkey(ks[0]); g = gk.v; isint = gk.isint;
                    Q.remove(g);
                }
                if (!is_root(g)) {
                    // spec section 28 / round 1047: a product exponent is never a member -- inside a cluster of roots not yet stepped this pass it
                    // is that cluster's storage (the cluster first, the exponent back on the queue); within R_0 of a registered root not yet
                    // stepped, that root's step's storage (routed); else its own step
                    auto rcs = root_clusters();
                    std::vector<std::vector<double>> rg;
                    for (auto& q : rcs) {
                        double mn = 0.0;
                        for (double x : q) mn += x;
                        mn /= q.size();
                        if (std::abs(g - mn) < 1.2 * (q.back() - q.front()) / 2 + R0) rg.push_back(q);
                    }
                    std::vector<double> rr0;
                    for (double r : reg_roots())
                        if (std::abs(r - g) < R0 && !Q.has(rkey(r).v)) rr0.push_back(r);
                    std::vector<double> stm;
                    if (!rg.empty())
                        for (double r : rg[0])
                            if (!Q.has(rkey(r).v)) stm.push_back(r);
                    std::string own = !stm.empty() ? "stored" : !rr0.empty() ? "routed" : "stepped", dec = own;
                    std::string lbg = pole_label(g);
                    double r0 = 0.0;
                    if (!rr0.empty()) {
                        r0 = rr0[0];
                        for (double r : rr0)
                            if (std::abs(r - g) < std::abs(r0 - g)) r0 = r;
                    }
                    if (FZ) {
                        // research round 1048: the frozen decision for this product (by identity)
                        std::vector<std::string> slz;
                        for (auto& kv : FZv.get("storage")->o) {
                            bool in = false;
                            for (auto& x : kv.second.get("poles")->a) in = in || x.s == lbg;
                            for (auto& x : kv.second.get("merged")->a) in = in || x.s == lbg;
                            if (in) slz.push_back(kv.first);
                        }
                        bool inst = false;
                        for (auto& x : FZv.get("product_steps")->a) inst = inst || x.s == lbg;
                        std::string fz = FZv.get("routed")->has(lbg) ? "routed" : inst ? "stepped" : !slz.empty() ? "stored" : "";
                        double mg = 1e300;
                        for (double r : reg_roots()) mg = std::min(mg, std::abs(r - g));
                        fz_note("product " + lbg + " (" + fmtm(g) + ")", fz.empty() ? "new (the point's rule)" : fz, own, fmtm(mg - R0));
                        dec = fz.empty() ? own : fz;
                        if (fz.empty() && STOR_RULE == "1061" && own == "stepped" && !reg_roots().empty()) {
                            // research round 1061: (c) for the product steps -- new within the J step: routed to the nearest registered root
                            auto rr = reg_roots();
                            r0 = rr[0];
                            for (double r : rr)
                                if (std::abs(r - g) < std::abs(r0 - g)) r0 = r;
                            dec = "routed";
                            fz_note("product " + lbg + " (" + fmtm(g) + "): new, routed to " + pole_label(r0) + " (never stepped; round 1061)", "routed", own, "None");
                        }
                        if (fz == "routed") r0 = pole_pos(FZv.get("routed")->get(lbg)->s);
                        else if (fz == "stored") {
                            const js::Val* fc = FZv.get("clusters")->get(slz[0]);
                            std::vector<std::string> fm_;
                            if (fc) for (auto& x : fc->get("members")->a) fm_.push_back(x.s);
                            else fm_.push_back(slz[0]);
                            stm.clear();
                            for (auto& m_ : fm_)
                                if (!is_spur_(m_) && !Q.has(rkey(pole_pos(m_)).v)) stm.push_back(pole_pos(m_));
                        }
                    }
                    Key gp = gk;
                    if (dec == "stored") {
                        if (stm.empty()) { Q.mark(g); continue; }
                        Q.push({gp});
                        gk = rkey(*std::min_element(stm.begin(), stm.end()));
                        g = gk.v; isint = gk.isint;
                        Q.remove(g);
                    } else if (dec == "routed") {
                        if (Q.has(rkey(r0).v)) { Q.mark(g); continue; }
                        std::printf("    [g %.6f] a product exponent within reach of the registered root %.6f (%.4f < R_0 %.4f): stored at that root's step\n",
                                    gp.v, r0, std::abs(r0 - gp.v), R0);
                        bool have = false;
                        for (auto& e : FZD_routed)
                            if (e.first == lbg) { e.second = pole_label(r0); have = true; }
                        if (!have) FZD_routed.emplace_back(lbg, pole_label(r0));
                        Q.push({gp});
                        ROUTED.insert(kf(r0));
                        gk = rkey(r0); g = gk.v; isint = gk.isint;
                        Q.remove(g);
                    } else {
                        FZD_steps.insert(lbg);
                        STEPLAB[kf(g)] = lbg;
                    }
                }
                Q.mark(g);
                if (is_root(g)) {
                    // the roots closer than 2 R_m to g, transitively: one step at their mean (a re-formed cluster keeps its key)
                    std::set<double> cs;
                    for (int x = c_.gdata + 1; x <= int(c_.gmax); x++) cs.insert(double(x));
                    for (double c0 : k_.c.cent) {
                        bool iskey = false;
                        for (auto& e : k_.c.clroots)
                            if (std::abs(c0 - e.first) < 1e-9) iskey = true;
                        if (c_.gdata < c0 && c0 <= c_.gmax + 1e-9 && is_root(c0) && !SPUR_SINGLE.count(kf(c0)) && !iskey) cs.insert(c0);
                    }
                    const std::vector<double>* kc = nullptr;
                    double kck = 0.0;
                    for (auto& e : k_.c.clroots) {
                        bool hit = std::abs(e.first - g) < 1e-9;
                        for (double m_ : e.second)
                            if (std::abs(m_ - g) < 1e-9) hit = true;
                        if (hit) { kc = &e.second; kck = e.first; break; }
                    }
                    mem = kc ? *kc : std::vector<double>{g};
                    bool grow = !kc;
                    while (grow) {
                        grow = false;
                        for (double c0 : cs) {
                            bool in = false, near = false;
                            for (double m_ : mem) {
                                if (std::abs(c0 - m_) <= 1e-9) in = true;
                                if (std::abs(c0 - m_) < 2 * k_.c.Rm) near = true;
                            }
                            if (!in && near) { mem.push_back(c0); grow = true; }
                        }
                    }
                    bool have_fkey = false;
                    double fkey = 0.0;
                    if (FZ && !kc) {
                        // research round 1048: the frozen cluster holding g's identity -- its members' identities at this point's positions, its key
                        // the frozen one; no growth; g alone if no frozen cluster holds it
                        std::string lbg = pole_label(g);
                        const js::Val* fc = nullptr;
                        for (auto& kv : FZv.get("clusters")->o)
                            for (auto& x : kv.second.get("members")->a)
                                if (!fc && x.s == lbg) fc = &kv.second;
                        std::vector<std::string> ownl;
                        for (double m_ : mem) ownl.push_back(pole_label(m_));
                        std::sort(ownl.begin(), ownl.end());
                        std::set<double> ms{g};
                        std::vector<std::string> frl;
                        if (fc) {
                            for (auto& x : fc->get("members")->a)
                                if (!is_spur_(x.s)) { ms.insert(pole_pos(x.s)); frl.push_back(x.s); }
                            fkey = fc->get("key")->n;
                            have_fkey = true;
                            if (fc->get("members")->a.size() == 1) ROUTED.insert(kf(g));
                        }
                        mem.assign(ms.begin(), ms.end());
                        std::sort(frl.begin(), frl.end());
                        auto join = [](const std::vector<std::string>& v) { std::string t; for (size_t q = 0; q < v.size(); q++) t += (q ? " | " : "") + v[q]; return t; };
                        double mg = 1e300;
                        for (double c0 : cs)
                            if (std::abs(c0 - g) > 1e-9) mg = std::min(mg, std::abs(c0 - g) - 2 * k_.c.Rm);
                        fz_note("cluster at " + lbg, fc ? join(frl) : lbg, join(ownl), fmtm(mg));
                    }
                    if (mem.size() > 1 || ROUTED.count(kf(g))) {
                        std::sort(mem.begin(), mem.end());
                        for (double m_ : mem) Q.mark(rkey(m_).v);
                        double mn = 0.0;
                        for (double m_ : mem) mn += m_;
                        Key gck = kc ? Key{kck, std::abs(kck - std::nearbyint(kck)) < 1e-9}
                                     : (have_fkey && mem.size() > 1) ? k_.c.xkey(fkey) : k_.c.xkey(mn / mem.size());
                        {
                            std::vector<std::string> mlab;
                            for (double m_ : mem) mlab.push_back(pole_label(m_));
                            if (!STEPLAB.count(kf(gck.v))) {
                                std::string t = "cluster";
                                bool first = true;
                                for (auto& l : mlab)
                                    if (!is_spur_(l)) { t += (first ? " " : " | ") + l; first = false; }
                                STEPLAB[kf(gck.v)] = mem.size() == 1 ? mlab[0] : t;
                            }
                            bool have = false;
                            for (auto& e : FZD_clusters)
                                if (e.first == STEPLAB[kf(gck.v)]) { e.second = FzClus{mlab, gck.v}; have = true; }
                            if (!have) FZD_clusters.emplace_back(STEPLAB[kf(gck.v)], FzClus{mlab, gck.v});
                        }
                        std::vector<double> mr;
                        for (double m_ : mem) mr.push_back(std::abs(m_ - std::nearbyint(m_)) < 1e-9 ? std::nearbyint(m_) : m_);
                        Ctx::kv_set(k_.c.clroots, gck.v, mr);
                        Ctx::kv_set(k_.c.clus, gck.v, mr);
                        bool have = false;
                        for (double c0 : k_.c.cent)
                            if (std::abs(c0 - gck.v) <= 1e-9) have = true;
                        if (!have) { k_.c.cent.push_back(gck.v); std::sort(k_.c.cent.begin(), k_.c.cent.end()); }
                        gk = gck; g = gk.v; isint = gk.isint;
                        iscl = true;
                        Rcur_ = std::max(R0, 1.2 * (mem.back() - mem.front()) / 2 + R0);
                        std::printf("    [cluster %.6f] members", g);
                        for (double m_ : mem) std::printf(" %.6f", m_);
                        std::printf(" (spread %.4f < 2 R_m %.4f): one step, contour radius %.4f\n", mem.back() - mem.front(), 2 * k_.c.Rm, Rcur_);
                        std::fflush(stdout);
                    }
                }
            }
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
            Ser rows_pre[6];                                        // stage (ii): the stripped series' rows (the storage's sources, re-read)
            if (CL)
                for (int f = 0; f < 6; f++) rows_pre[f] = k_.out[f];
            if (iscl) set_storage(rows_pre, g, Rcur_);              // spec section 28: the storage at the contour's largest radius
            Vec S = CL ? project(k_.c, rows_pre, g, 1, P_.gfam, P_.dE, P_.W, false) : galb(g);
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
                if (iscl) {                                         // stage (ii): the operator from perturbations at the key, read plainly
                    opname = "engine-integer";
                    if (from_cache) { Ms = cache[g].first; Mc = cache[g].second; }
                    else {
                        std::vector<Entry> Cs2 = strip(C, g);       // (the storage poles' own columns too, as n2a_nl's strip now)
                        k_.c.noclus = true;
                        meff(k_, P_, Cs2, g, c_.top, c_.meff_mode, c_.meff_h, c_.meff_chunk, Ms, Mc);
                        k_.c.noclus = false;
                        cache[g] = {Ms, Mc};
                    }
                    have_mref = true; mref_g = g; mref_s = Ms; mref_c = Mc;
                } else if (isint || !have_mref) {
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
                if (CL && newt == 0 && (iscl || (!isint && !is_root(g)))) {
                    std::vector<int> iax;
                    {
                        auto ac = ct::active(g, P_.kofs);
                        for (int j = 0; j < NB; j++)
                            if (ac[j]) iax.push_back(j);
                    }
                    Vec ev = cl::pencil_roots(Ms, NB, iax, S_.rw, S_.cw);
                    for (auto& e : ev) e += g;
                    if (iscl) {
                        // stage (ii): the cluster's contour encloses its members' roots and no other root of its operator; spec sections 27 / 31:
                        // every root inside (1.2 x half the spread + R_0) that is no integer's or family's (the spurious third family) is excluded by
                        // the radius where a margin allows, else a member (its amplitude an N3b parameter)
                        double half = (mem.back() - mem.front()) / 2;
                        if (k_.c.gauge_reg) {
                            for (auto it = SPUR.begin(); it != SPUR.end();)
                                it = std::abs(it->second[2] - g) < 1e-9 ? SPUR.erase(it) : std::next(it);
                            // research round 1048: each root named by its identity; with the freeze, each spurious root's decision at this step
                            // (member / excluded, its storage pole) is the frozen one, a frozen member absent here kept at its frozen position
                            std::string stl = STEPLAB.count(kf(g)) ? STEPLAB[kf(g)] : std::string();
                            const js::Val* fsp = nullptr;
                            js::Val empty_;
                            empty_.t = js::Val::OBJ;
                            if (FZ) { fsp = FZv.get("spur")->get(stl); if (!fsp) fsp = &empty_; }
                            std::vector<std::pair<std::string, FzSp>> rec;
                            auto spur_member = [&](double er, const std::string& lb) {
                                double mm = mem[0];
                                for (double m_ : mem)
                                    if (std::abs(m_ - er) < std::abs(mm - er)) mm = m_;
                                std::string fst;
                                bool has_fst = false;
                                if (fsp && fsp->get(lb) && fsp->get(lb)->get("storage")) { fst = fsp->get(lb)->get("storage")->s; has_fst = true; }
                                double stm_;
                                if (std::abs(mm - er) < k_.c.reg_tol || (has_fst && fst != lb)) {
                                    if (std::abs(mm - er) < k_.c.reg_tol || !has_fst) stm_ = mm;
                                    else {
                                        double fp = pole_pos(fst);
                                        stm_ = mem[0];
                                        for (double m_ : mem)
                                            if (std::abs(m_ - fp) < std::abs(stm_ - fp)) stm_ = m_;
                                    }
                                } else {
                                    stm_ = k_.c.reg_exact(er).v;
                                    Key xk = k_.c.xkey(stm_);
                                    auto* cr = Ctx::kv_get(k_.c.clroots, g);
                                    cr->push_back(xk.v);
                                    std::sort(cr->begin(), cr->end());
                                    auto* cp = Ctx::kv_get(k_.c.clus, g);
                                    bool have = false;
                                    for (double m_ : *cp)
                                        if (std::abs(m_ - stm_) <= 1e-9) have = true;
                                    if (!have) { cp->push_back(xk.v); std::sort(cp->begin(), cp->end()); }
                                    mem.push_back(stm_);
                                    std::sort(mem.begin(), mem.end());
                                }
                                SPUR[lb] = {er, stm_, g};
                                bool own_pole = std::abs(stm_ - er) < 1e-9 && (!has_fst || fst == lb);
                                rec.emplace_back(lb, FzSp{"member", er, own_pole ? lb : pole_label(stm_), true});
                                std::printf("    [cluster %.6f] spurious root %.6f (the third family) too close to exclude: member '%s' (amplitude an N3b parameter, "
                                            "datum %.3e), stored at %.6f\n", g, er, lb.c_str(), offl_set.count(lb) ? offl_set[lb] : 0.0, stm_);
                            };
                            std::vector<cd> evs = ev;
                            std::sort(evs.begin(), evs.end(), [](const cd& a, const cd& b) { return a.real() < b.real(); });
                            std::set<std::string> done;
                            for (size_t ie = 0; ie < evs.size(); ie++) {
                                cd e = evs[ie];
                                double er = e.real();
                                if (std::abs(er - g) > 1.2 * half + R0 || !root_real(ie, evs)) continue;
                                std::string lb = root_label(er);
                                if (!is_spur(lb)) continue;
                                if (done.count(lb)) std::printf("    [cluster %.6f] WARNING: two roots named %s (the second %.6f)\n", g, lb.c_str(), er);
                                done.insert(lb);
                                double dmn = 1e300;
                                for (double m_ : mem) dmn = std::min(dmn, std::abs(er - m_));
                                std::string own = (std::abs(er - g) - R0 / 4 >= 1.1 * half && dmn >= R0 / 4) ? "excluded" : "member", dec = own;
                                if (fsp) {
                                    std::string fz = fsp->get(lb) ? fsp->get(lb)->get("status")->s : "";
                                    fz_note("spurious " + lb + " (" + fmtm(er) + ") at step " + stl, fz.empty() ? "new (the point's rule)" : fz, own,
                                            fmtm(std::min(std::abs(er - g) - R0 / 4 - 1.1 * half, dmn - R0 / 4)));
                                    if (!fz.empty()) dec = fz;
                                }
                                if (dec == "excluded") {
                                    rec.emplace_back(lb, FzSp{"excluded", er, "", false});
                                    std::printf("    [cluster %.6f] spurious root %.6f (%s, the third family): excluded by the radius\n", g, er, lb.c_str());
                                    continue;
                                }
                                spur_member(er, lb);
                            }
                            std::vector<std::pair<std::string, js::Val>> fso;
                            if (fsp) { fso = fsp->o; std::sort(fso.begin(), fso.end(), [](const std::pair<std::string, js::Val>& x, const std::pair<std::string, js::Val>& y) { return x.first < y.first; }); }
                            if (fsp)
                                for (auto& kv : fso)
                                    if (kv.second.get("status")->s == "member" && !done.count(kv.first)) {
                                        fz_note("spurious " + kv.first + " at step " + stl, "member (its frozen root " + fmtm(kv.second.get("root")->n) + ")", "absent", "None");
                                        spur_member(kv.second.get("root")->n, kv.first);
                                    }
                            bool have = false;
                            for (auto& e : FZD_spur)
                                if (e.first == stl) { e.second = rec; have = true; }
                            if (!have) FZD_spur.emplace_back(stl, rec);
                            half = (mem.back() - mem.front()) / 2;
                        }
                        std::vector<double> spr;
                        for (auto& sp : SPUR)
                            if (std::abs(sp.second[2] - g) < 1e-9) spr.push_back(sp.second[0]);
                        std::vector<cd> oth;
                        own_roots.clear();
                        roots_all = ev;
                        for (auto& e : ev) {
                            double dmn = 1e300;
                            for (double m_ : mem) dmn = std::min(dmn, std::abs(e - m_));
                            for (double m_ : spr) dmn = std::min(dmn, std::abs(e - m_));
                            (dmn < R0 / 4 ? own_roots : oth).push_back(e);
                        }
                        std::stable_sort(oth.begin(), oth.end(), [&](const cd& a, const cd& b) { return std::abs(a - g) < std::abs(b - g); });
                        double Rc = 1.2 * half + R0;
                        if (!oth.empty() && std::abs(oth[0] - g) - R0 / 4 < Rc) Rc = std::max(std::abs(oth[0] - g) - R0 / 4, 1.1 * half);
                        Rcur_ = Rc;
                        set_storage(rows_pre, g, Rcur_);
                        S = project(k_.c, rows_pre, g, 1, P_.gfam, P_.dE, P_.W, false);
                        Smax = maxabs(S);
                        for (double p0 : *Ctx::kv_get(k_.c.clus, g)) Q.mark(rkey(p0).v);
                        if (auto* ab = Ctx::kv_get(k_.c.clabs, g))
                            for (double p0 : *ab) Q.mark(rkey(p0).v);
                        std::printf("    [cluster %.6f] its operator's roots: members'", g);
                        for (auto& e : own_roots) std::printf(" %.6f", e.real());
                        if (oth.empty()) std::printf("; nearest other - -> contour radius %.4f\n", Rcur_);
                        else std::printf("; nearest other %.6f (distance %.4f) -> contour radius %.4f\n", oth[0].real(), std::abs(oth[0] - g), Rcur_);
                        std::fflush(stdout);
                    } else {
                        // spec section 29: a SINGLE product exponent's step that reaches a root of its operator -- excluded by the radius when a
                        // margin allows (|root - g| - R_0 / 4 >= eps_s), else a member, the step a cluster of that root with g's exponent a pole
                        std::vector<double> evr;
                        for (size_t ie = 0; ie < ev.size(); ie++)
                            if (std::abs(ev[ie] - g) < Rcur_ && root_real(ie, ev)) evr.push_back(ev[ie].real());
                        std::stable_sort(evr.begin(), evr.end(), [&](double a, double b) { return std::abs(a - g) < std::abs(b - g); });
                        std::vector<double> memb;
                        std::map<double, std::string> mlab;
                        std::string stl = STEPLAB.count(kf(g)) ? STEPLAB[kf(g)] : std::string();
                        const js::Val* fsp = nullptr;
                        js::Val empty_;
                        empty_.t = js::Val::OBJ;
                        if (FZ) { fsp = FZv.get("spur")->get(stl); if (!fsp) fsp = &empty_; }
                        std::vector<std::pair<std::string, FzSp>> rec;
                        for (double er : evr) {
                            std::string lb = root_label(er);
                            if (!is_spur(lb)) {
                                Rcur_ = std::min(Rcur_, std::max(std::abs(er - g) - R0 / 4, EPS_S));
                                std::printf("    [g %.6f] a root of its operator %.6f is the registered root %s: not spurious; excluded by the radius (%.4f)\n",
                                            g, er, lb.c_str(), Rcur_);
                                continue;
                            }
                            std::string own = std::abs(er - g) - R0 / 4 >= EPS_S ? "excluded" : "member", dec = own;
                            if (fsp) {
                                std::string fz = fsp->get(lb) ? fsp->get(lb)->get("status")->s : "";
                                fz_note("spurious " + lb + " (" + fmtm(er) + ") at step " + stl, fz.empty() ? "new (the point's rule)" : fz, own,
                                        fmtm(std::abs(er - g) - R0 / 4 - EPS_S));
                                if (!fz.empty()) dec = fz;
                            }
                            if (dec == "excluded") {
                                Rcur_ = std::min(Rcur_, std::max(std::abs(er - g) - R0 / 4, EPS_S));
                                rec.emplace_back(lb, FzSp{"excluded", er, "", false});
                                std::printf("    [g %.6f] a root of its operator %.6f (%s) inside the contour: excluded by the radius (%.4f)\n", g, er, lb.c_str(), Rcur_);
                            } else {
                                memb.push_back(er);
                                mlab[er] = lb;
                            }
                        }
                        if (fsp) {
                            std::vector<std::pair<std::string, js::Val>> fso = fsp->o;
                            std::sort(fso.begin(), fso.end(), [](const std::pair<std::string, js::Val>& x, const std::pair<std::string, js::Val>& y) { return x.first < y.first; });
                            for (auto& kv : fso) {
                                bool in = false;
                                for (auto& ml : mlab) in = in || ml.second == kv.first;
                                if (kv.second.get("status")->s == "member" && !in) {
                                    fz_note("spurious " + kv.first + " at step " + stl, "member (its frozen root " + fmtm(kv.second.get("root")->n) + ")", "absent", "None");
                                    memb.push_back(kv.second.get("root")->n);
                                    mlab[kv.second.get("root")->n] = kv.first;
                                }
                            }
                        }
                        if (!memb.empty()) {
                            for (auto it = SPUR.begin(); it != SPUR.end();)
                                it = std::abs(it->second[2] - g) < 1e-9 ? SPUR.erase(it) : std::next(it);
                            std::set<double> sts;
                            for (double er : memb) {
                                bool at_g = std::abs(er - g) < EPS_S;
                                if (fsp && fsp->get(mlab[er]) && fsp->get(mlab[er])->get("storage")) at_g = fsp->get(mlab[er])->get("storage")->s == "step";
                                rec.emplace_back(mlab[er], FzSp{"member", er, at_g ? "step" : mlab[er], true});
                                double stm_ = at_g ? g : k_.c.reg_exact(er).v;
                                SPUR[mlab[er]] = {er, stm_, g};
                                SPUR_SINGLE.insert(kf(er));
                                sts.insert(rkey(stm_).v);
                                std::printf("    [g %.6f] a root of its operator %.6f (%s) too close to exclude (%.1e): a member (amplitude an N3b parameter, "
                                            "datum %.3e), stored at %.6f\n", g, er, mlab[er].c_str(), std::abs(er - g), offl_set.count(mlab[er]) ? offl_set[mlab[er]] : 0.0, stm_);
                            }
                            std::vector<double> mr(sts.begin(), sts.end());
                            Ctx::kv_set(k_.c.clroots, g, mr);

                            Ctx::kv_set(k_.c.clus, g, mr);
                            iscl = true;
                            mem = mr;
                            own_roots.clear();
                            for (double er : memb) own_roots.push_back(cd(er, 0.0));
                            roots_all = own_roots;
                            set_storage(rows_pre, g, Rcur_);
                            S = project(k_.c, rows_pre, g, 1, P_.gfam, P_.dE, P_.W, false);
                            Smax = maxabs(S);
                            for (double p0 : *Ctx::kv_get(k_.c.clus, g)) Q.mark(rkey(p0).v);
                            if (auto* ab = Ctx::kv_get(k_.c.clabs, g))
                                for (double p0 : *ab) Q.mark(rkey(p0).v);
                        }
                        {
                            bool have = false;
                            for (auto& e : FZD_spur)
                                if (e.first == stl) { e.second = rec; have = true; }
                            if (!have) FZD_spur.emplace_back(stl, rec);
                        }
                    }
                }
                k_.c.trace_tag = "pass " + std::to_string(ps) + " g " + std::to_string(g) + " contour/sources";
                // the contour step
                bool clk_;
                auto dmk_ = cl::clus_mem(k_.c, g, clk_);              // stage (ii): a cluster key -- the sources' poles at its storage
                Vec Sn = newt ? st::newton_sources(Xk, Ms, K, NB, S_.e1, Rcur_, rowsfn(Cs, g))
                              : (clk_ ? cl::nodes_of(S, dmk_, K, NB, S_.e1, Rcur_) : ct::nodes_of(S, K, NB, S_.e1, Rcur_));
                act = ct::active(g, P_.kofs);
                xrows_.clear();
                for (int j = 0; j < NB; j++)
                    if (!act[j]) xrows_.push_back(j);
                double kf = std::nearbyint(g * 1e9) / 1e9;
                ct::Contour o;
                ct::contour(Ms, NB, S_.rw, S_.cw, S_.e1, Rcur_, act, Sn, rk_fix.count(kf) ? rk_fix[kf] : -1, o);
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
                        for (auto& l : L.gauge) gfams.emplace_back("gauge l" + std::to_string(l.first), l.second);
                        Q.push(towers_all());
                        if (CL) Q.push(fz_pending());
                        Json F;
                        F.open('{');
                        F.kv("block_roots_real", Json::nums(L.evr));
                        F.kv("eig_all", arr("eig_all", L.eig_all, {long(L.eig_all.size())}));
                        std::string la = "{";
                        for (size_t q = 0; q < L.lapse.size(); q++) la += (q ? "," : "") + Json::str("lapse l" + std::to_string(L.lapse[q].first)) + ":" + Json::num(L.lapse[q].second.v);
                        F.kv("lapse", la + "}");
                        std::string ga = "{";
                        for (size_t q = 0; q < L.gauge.size(); q++) ga += (q ? "," : "") + Json::str("gauge l" + std::to_string(L.gauge[q].first)) + ":" + Json::num(L.gauge[q].second.v);
                        F.kv("gauge", ga + "}");
                        std::string rg = "[";
                        for (size_t q = 0; q < L.registry.size(); q++) rg += (q ? "," : "") + std::string("[") + std::to_string(L.registry[q].first) + "," + Json::num(L.registry[q].second) + "]";
                        F.kv("registry", rg + "]");
                        F.close('}');
                        families_json = F.s.str();
                    }
                }
                if (!isint && iscl) {
                    // ---- stage (ii): the cluster's step (spec sections 24-32) -- n2a_nl's off-integer branch at a cluster key ----
                    bool okc;
                    std::vector<double> dm = cl::clus_mem(k_.c, g, okc);
                    double R = Rcur_;
                    auto tgdc = [&](const Vec& X) { return st::tg_dc(X, NB, S_.e1, R); };
                    auto tg2 = [&](const Vec& X) {                  // moments 0 and 1 of the node values (all columns): 2 NB
                        Vec o2(2 * NB, 0.0);
                        for (int k = 0; k < N; k++)
                            for (int j = 0; j < NB; j++) {
                                o2[j] += R / N * S_.e1[k] * X[size_t(k) * NB + j];
                                o2[NB + j] += R * R / N * S_.e1[k] * S_.e1[k] * X[size_t(k) * NB + j];
                            }
                        return o2;
                    };
                    // the labelled members: the lapse families first (by value), then the gauge roots, then the spurious members (by root)
                    std::vector<std::pair<std::string, double>> labl;
                    for (auto* fv : {&fams, &gfams})
                        for (auto& f : *fv) {
                            bool hit = std::abs(f.second.v - g) < 1e-12;
                            for (double m_ : mem)
                                if (std::abs(f.second.v - m_) < 1e-9) hit = true;
                            if (hit) labl.emplace_back(f.first, f.second.v);
                        }
                    std::stable_sort(labl.begin(), labl.end(), [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
                        bool ga = a.first.rfind("gauge", 0) == 0, gb = b.first.rfind("gauge", 0) == 0;
                        return ga != gb ? gb : a.second < b.second;
                    });
                    std::vector<std::pair<std::string, double>> spl;
                    for (auto& sp : SPUR)
                        if (std::abs(sp.second[2] - g) < 1e-9) spl.emplace_back(sp.first, sp.second[0]);
                    std::stable_sort(spl.begin(), spl.end(), [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) { return a.second < b.second; });
                    std::vector<std::string> lab_;
                    for (auto& l : labl) lab_.push_back(l.first);
                    for (auto& l : spl) lab_.push_back(l.first);
                    std::vector<Vec> Wq = o.Wq;
                    Vec Xg_ = o.X0;
                    struct Lab { std::string label; double root, svmin, svmax, pres; bool datum_none; double datum; };
                    std::vector<Lab> per;
                    std::vector<int> spq;
                    std::vector<cd> dat;
                    if (rk) {
                        if (!lab_.empty()) {
                            // each labelled member's datum on the residue combination nearest its null vector at its own root (the cluster's
                            // operator there); the registry's centre moved to the operator's root of that identity (its towers with it)
                            std::vector<double> own_r;
                            for (auto& e : own_roots) {
                                size_t ia = 0;
                                while (ia < roots_all.size() && roots_all[ia] != e) ia++;
                                bool re = ia < roots_all.size() ? root_real(ia, roots_all) : std::abs(e.imag()) < 1e-9;
                                if (re) own_r.push_back(e.real());
                                if (std::getenv("MK_ROOT_LOG"))             // (round 1064 diagnostic: the members' roots with their imaginary parts)
                                    std::printf("    [cluster %.6f] own root %.12f %+.3e i%s\n", g, e.real(), e.imag(), re ? "" : " (not real)");
                            }
                            Vec U_(size_t(NB) * rk);
                            for (int q = 0; q < rk; q++) {
                                auto t = tgdc(Wq[q]);
                                for (int j = 0; j < NB; j++) U_[size_t(j) * rk + q] = t[j];
                            }
                            std::vector<Vec> Acols;
                            int nl = std::min(int(lab_.size()), rk);
                            for (int li = 0; li < nl; li++) {
                                const std::string& lb = lab_[li];
                                bool sp = is_spur(lb);
                                auto* regv = lb.rfind("gauge", 0) == 0 ? &gfams : &fams;
                                double sf = 0.0;
                                Key* regk = nullptr;
                                if (sp) sf = SPUR[lb][0];
                                else
                                    for (auto& f : *regv)
                                        if (f.first == lb) { sf = f.second.v; regk = &f.second; }
                                if (!own_r.empty() && !sp) {
                                    std::vector<double> idr;
                                    for (double e : own_r)
                                        if (root_label(e) == lb) idr.push_back(e);
                                    double rt = sf;
                                    if (!idr.empty()) {
                                        rt = idr[0];
                                        for (double e : idr)
                                            if (std::abs(e - sf) < std::abs(rt - sf)) rt = e;
                                    }
                                    if (std::abs(rt - sf) > 1e-12 && !(mem.size() == 1 && ROUTED.count(Engine::kf(mem[0])))) {
                                        Key nk = k_.c.xkey(rt);
                                        std::printf("    [cluster %.6f] %s: the registered centre %.9f -> the operator's root %.9f (moved %.1e)\n", g, lb.c_str(), sf, nk.v, nk.v - sf);
                                        if (regk) *regk = nk;
                                        std::vector<std::pair<double, double>> mv{{sf, nk.v}};
                                        if (k_.c.gauge_reg)
                                            for (int kk = 1; kk <= int(c_.gmax); kk++)
                                                if (sf + kk <= c_.gmax + 1e-9) mv.emplace_back(sf + kk, nk.v + kk);
                                        for (auto& ow : mv) {
                                            double o_ = ow.first, w_ = ow.second;
                                            Key wk = k_.c.xkey(w_);
                                            std::vector<double> cn;
                                            for (double c0 : k_.c.cent)
                                                if (std::abs(c0 - o_) > 1e-12) cn.push_back(c0);
                                            bool have = false;
                                            for (double c0 : k_.c.cent)
                                                if (std::abs(c0 - w_) <= 1e-12) have = true;
                                            if (!have) cn.push_back(wk.v);
                                            std::sort(cn.begin(), cn.end());
                                            k_.c.cent = cn;
                                            for (auto* D_ : {&k_.c.clus, &k_.c.clroots})
                                                for (auto& e : *D_)
                                                    for (auto& m_ : e.second)
                                                        if (std::abs(m_ - std::nearbyint(m_)) > 1e-9 && std::abs(m_ - o_) < 1e-12) m_ = wk.v;
                                            if (std::abs(o_ - sf) > 1e-12) {
                                                int ndrop = 0;
                                                std::vector<Entry> Cn;
                                                for (auto& en : C) {
                                                    if (std::abs(en.t.v - (o_ + P_.kofs[en.j])) < 1e-12) { ndrop++; continue; }
                                                    Cn.push_back(en);
                                                }
                                                C = Cn;
                                                std::printf("    [cluster %.6f] %s + %d: the tower %.9f -> %.9f (%d stored entries dropped)\n", g, lb.c_str(),
                                                            int(std::nearbyint(o_ - sf)), o_, w_, ndrop);
                                                if (w_ > g + 1e-9) Q.push({wk});
                                            }
                                        }
                                        mem = *Ctx::kv_get(k_.c.clroots, g);
                                        sf = nk.v;
                                    }
                                }
                                double d = sf - g;
                                size_t nmm = size_t(NB) * NB;
                                int na_ = int(o.ia.size());
                                Vec Mf(size_t(na_) * na_);
                                for (int a = 0; a < na_; a++)
                                    for (int b = 0; b < na_; b++) {
                                        size_t ij = size_t(o.ia[a]) * NB + o.ia[b];
                                        Mf[size_t(a) * na_ + b] = (Ms[ij] + d * Ms[nmm + ij] + d * d * Ms[2 * nmm + ij]) / (S_.rw[o.ia[a]] * S_.cw[o.ia[b]]);
                                    }
                                std::vector<double> svf;
                                Vec Uf, Vf;
                                la::svd(Mf, na_, na_, svf, &Uf, &Vf, true);
                                Vec vf(NB, 0.0);
                                for (int b = 0; b < na_; b++) vf[o.ia[b]] = std::conj(Vf[size_t(na_ - 1) * na_ + b]) / S_.cw[o.ia[b]];
                                Vec a_ = la::lstsq(U_, NB, rk, vf, 1);
                                double rn = 0.0, vn = 0.0;
                                for (int j = 0; j < NB; j++) {
                                    cd t = -vf[j];
                                    for (int q = 0; q < rk; q++) t += U_[size_t(j) * rk + q] * a_[q];
                                    rn += std::norm(t);
                                    vn += std::norm(vf[j]);
                                }
                                Acols.push_back(Vec(a_.begin(), a_.begin() + rk));
                                Lab lr{lb, sf, svf.back(), svf.front(), std::sqrt(rn) / std::max(std::sqrt(vn), 1e-300), sp, 0.0};
                                lr.datum = sp ? (offl_set.count(lb) ? offl_set[lb] : 0.0) : lb.rfind("gauge", 0) == 0 ? 0.0 : (offl_set.count(lb) ? offl_set[lb] : 0.0);
                                lr.datum_none = false;                  // (the parameter route: a spurious member's datum is its N3b parameter)
                                per.push_back(lr);
                            }
                            std::vector<Vec> Wl;
                            for (int k = 0; k < nl; k++) {
                                Vec w(Wq[0].size(), 0.0);
                                for (int q = 0; q < rk; q++)
                                    for (size_t i = 0; i < w.size(); i++) w[i] += Acols[k][q] * Wq[q][i];
                                Wl.push_back(w);
                            }
                            if (rk > nl) {
                                Vec Am(size_t(rk) * (nl + rk), 0.0);
                                for (int q = 0; q < rk; q++) {
                                    for (int k = 0; k < nl; k++) Am[size_t(q) * (nl + rk) + k] = Acols[k][q];
                                    Am[size_t(q) * (nl + rk) + nl + q] = 1.0;
                                }
                                Vec Qa = cl::qr_q(Am, rk, nl + rk);           // rk x rk
                                for (int k = nl; k < rk; k++) {
                                    Vec w(Wq[0].size(), 0.0);
                                    for (int q = 0; q < rk; q++)
                                        for (size_t i = 0; i < w.size(); i++) w[i] += Qa[size_t(q) * rk + k] * Wq[q][i];
                                    Wl.push_back(w);
                                }
                            }
                            Wq = Wl;
                            for (auto& lr : per)
                                std::printf("    [cluster %.6f] residue rank %d; %s on its null vector (sigma_min %.1e of %.1e at the root %.6f; projection residual %.1e): datum %s\n",
                                            g, rk, lr.label.c_str(), lr.svmin, lr.svmax, lr.root, lr.pres, lr.label.rfind("gauge", 0) == 0 ? "pinned to 0 (gauge)" : "given");
                        }
                        for (auto& w : Wq) {                            // unit residue, the largest u_g residue real positive
                            auto t = tg2(w);
                            double nr = 0.0;
                            for (int j = 0; j < NB; j++) nr += std::norm(t[j]);
                            nr = std::sqrt(nr);
                            for (auto& z : w) z /= nr;
                            t = tg2(w);
                            int im = 0;
                            for (int j = 1; j < nn; j++)
                                if (std::abs(t[j]) > std::abs(t[im])) im = j;
                            cd ph = std::exp(cd(0.0, -std::arg(t[im])));
                            for (auto& z : w) z *= ph;
                        }
                        if (!lab_.empty())
                            for (size_t q = 0; q < Wq.size() && q < lab_.size(); q++) {
                                const std::string& lb = lab_[q];
                                dat.push_back(lb.rfind("gauge", 0) == 0 ? 0.0 : (offl_set.count(lb) ? offl_set[lb] : 0.0));
                                if (is_spur(lb)) spq.push_back(int(q));
                            }
                        for (size_t q = 0; q < dat.size(); q++)
                            for (size_t i = 0; i < Xg_.size(); i++) Xg_[i] += dat[q] * Wq[q][i];
                    }
                    // the contour solution's defect correction (the amplitudes held): the CHECK of the local parts below.  (The poles' offsets
                    // re-read: a labelled member moved to its operator's root moved its pole, as n2a_nl's clus_mem reads the registry here)
                    dm = cl::clus_mem(k_.c, g, okc);
                    std::vector<Entry> Cst = strip(C, g);
                    auto rowsat = [&](const Vec& X) {
                        k_.run(cl::with_grade(k_.c, Cst, X, g, dm, P_.kofs, S_.e1, R), nullptr, c_.top);
                        return galb(g);
                    };
                    Vec Ahold(size_t(NB) * std::max(rk, 1), 0.0);
                    for (int q = 0; q < rk; q++) {
                        auto t = tgdc(Wq[q]);
                        for (int j = 0; j < NB; j++) Ahold[size_t(j) * rk + q] = t[j];
                    }
                    std::vector<double> dres;
                    int ndef = std::max(c_.defect, 4);
                    for (int it = 0; it <= ndef; it++) {
                        Vec Rr = rowsat(Xg_);
                        dres.push_back(maxabs(Rr));
                        if (dres.back() < 1e-14 || it == ndef) break;
                        if (c_.robust && dres.size() > 1 && dres.back() > 0.5 * dres[dres.size() - 2]) break;
                        Vec nd = cl::nodes_of(Rr, dm, K, NB, S_.e1, R), dX(size_t(N) * NB);
                        for (int k = 0; k < N; k++)
                            for (int i = 0; i < NB; i++) {
                                cd sacc = 0.0;
                                for (int j = 0; j < NB; j++) sacc += o.Minv[k][size_t(i) * NB + j] * nd[size_t(k) * NB + j];
                                dX[size_t(k) * NB + i] = -sacc;
                            }
                        if (rk) {
                            auto t = tgdc(dX);
                            Vec a = la::lstsq(Ahold, NB, rk, t, 1);
                            for (int q = 0; q < rk; q++)
                                for (size_t i = 0; i < dX.size(); i++) dX[i] -= a[q] * Wq[q][i];
                        }
                        for (size_t i = 0; i < Xg_.size(); i++) Xg_[i] += dX[i];
                    }
                    // spec section 29: the storage from each pole's LOCAL principal part -- the particular part from the stripped series' source,
                    // the roots' data on their null directions, then the defect correction pole by pole (the full rows' stacked readout)
                    std::vector<std::pair<int, int>> bordl;
                    for (auto& bd : o.bord) {
                        int r_ = -1, c0 = -1;
                        for (size_t a = 0; a < o.ia.size(); a++) {
                            if (o.ia[a] == bd.first) r_ = int(a);
                            if (o.ia[a] == bd.second) c0 = int(a);
                        }
                        bordl.emplace_back(r_, c0);
                    }
                    std::map<int, cd> ampL;
                    for (auto& lr : per) {
                        int iL = 0;
                        for (size_t i = 1; i < dm.size(); i++)
                            if (std::abs(g + dm[i] - lr.root) < std::abs(g + dm[iL] - lr.root)) iL = int(i);
                        ampL[iL] = lr.label.rfind("gauge", 0) == 0 ? cd(0.0, 0.0) : cd(lr.datum, 0.0);
                    }
                    Vec mS(S.size());
                    for (size_t i = 0; i < S.size(); i++) mS[i] = -S[i];
                    cl::LocalInfo info;
                    Vec XL = cl::local_pp(Ms, NB, nn, K, o.ia, bordl, dm, mS, R, EPS_S, S_.rw, S_.cw, &ampL, &info);
                    std::vector<double> histL;
                    Vec RL;
                    for (int itL = 0; itL < ndef + 4; itL++) {
                        k_.run(cl::with_grade_mom(k_.c, Cst, XL, g, dm, P_.kofs), nullptr, c_.top);
                        RL = galb(g);
                        histL.push_back(maxabs(RL));
                        size_t h = histL.size();
                        if (histL.back() < 1e-14 || (h > 2 && histL[h - 1] > 0.5 * histL[h - 2] && histL[h - 2] > 0.5 * histL[h - 3])) break;
                        Vec mR(RL.size());
                        for (size_t i = 0; i < RL.size(); i++) mR[i] = -RL[i];
                        Vec dXL = cl::local_pp(Ms, NB, nn, K, o.ia, bordl, dm, mR, R, EPS_S, S_.rw, S_.cw, nullptr, nullptr);
                        for (size_t i = 0; i < XL.size(); i++) XL[i] += dXL[i];
                    }
                    Vec Fc = project(k_.c, k_.out, g, 1, P_.cfam, P_.cdE, P_.Wc, true);   // the conditions (not imposed), the last rows
                    std::set<int> projected;
                    if (!spq.empty() && c_.spur_orth) {
                        // research round 1049 (option (c)): the particular solution's component along a spurious member's null direction removed,
                        // the amplitude parameter the whole of it
                        for (int q : spq) {
                            const std::string& lb = lab_[q];
                            int i = 0;
                            for (size_t ii = 1; ii < dm.size(); ii++)
                                if (std::abs(g + dm[ii] - SPUR[lb][1]) < std::abs(g + dm[i] - SPUR[lb][1])) i = int(ii);
                            if (!info.vhat.count(i)) {
                                std::printf("    [cluster %.6f] %s: its storage pole %.6f is regular -- no null direction to project\n", g, lb.c_str(), g + dm[i]);
                                continue;
                            }
                            Vec v = info.vhat[i];
                            double vn = 0.0;
                            for (auto& z : v) vn += std::norm(z);
                            vn = std::sqrt(vn);
                            for (auto& z : v) z /= vn;
                            cd c0 = 0.0;
                            for (int j = 0; j < NB; j++) c0 += std::conj(v[j]) * XL[size_t(i * K) * NB + j];
                            cd a0 = ampL.count(i) ? ampL[i] : cd(0.0, 0.0);
                            cd dl0 = info.dl.count(i) ? info.dl[i] : cd(0.0, 0.0);
                            for (int m = 0; m < K; m++) {
                                cd f = (a0 - c0) * std::pow(dl0, m);
                                for (int j = 0; j < NB; j++) XL[size_t(i * K + m) * NB + j] += f * v[j];
                            }
                            std::printf("    [cluster %.6f] %s (pole %.6f): the particular solution's component along the null direction %.3e%+.3ei removed, the "
                                        "amplitude %.3e%+.3ei (the N3b parameter) the whole of it (round 1049)\n", g, lb.c_str(), g + dm[i], (c0 - a0).real(),
                                        (c0 - a0).imag(), a0.real(), a0.imag());
                            projected.insert(i);
                        }
                    }
                    if (ROOT_PROJ_ALL) {
                        // research round 1062 (option (a)): every labelled root's null component projected to its datum -- a gauge root to 0, a
                        // lapse root to its datum (its total amplitude); the spurious roots above
                        for (auto& lr : per) {
                            if (lr.label.rfind("spurious", 0) == 0) continue;
                            int i = 0;
                            for (size_t ii = 1; ii < dm.size(); ii++)
                                if (std::abs(g + dm[ii] - lr.root) < std::abs(g + dm[i] - lr.root)) i = int(ii);
                            if (!info.vhat.count(i) || projected.count(i)) continue;
                            Vec v = info.vhat[i];
                            double vn = 0.0;
                            for (auto& z : v) vn += std::norm(z);
                            vn = std::sqrt(vn);
                            for (auto& z : v) z /= vn;
                            cd c0 = 0.0;
                            for (int j = 0; j < NB; j++) c0 += std::conj(v[j]) * XL[size_t(i * K) * NB + j];
                            cd a0 = ampL.count(i) ? ampL[i] : cd(0.0, 0.0);
                            cd dl0 = info.dl.count(i) ? info.dl[i] : cd(0.0, 0.0);
                            for (int m = 0; m < K; m++) {
                                cd f = (a0 - c0) * std::pow(dl0, m);
                                for (int j = 0; j < NB; j++) XL[size_t(i * K + m) * NB + j] += f * v[j];
                            }
                            projected.insert(i);
                            std::printf("    [cluster %.6f] %s (pole %.6f): its null component %.3e%+.3ei projected to its datum %.3e%+.3ei (round 1062)\n", g,
                                        lr.label.c_str(), g + dm[i], c0.real(), c0.imag(), a0.real(), a0.imag());
                        }
                    }
                    // the CHECK: the contour solution's moments against the local parts'
                    int NMc;
                    auto Amc = cl::clus_map(dm, K, N, R, NMc);
                    double chk = 0.0, mumax = 0.0;
                    {
                        int LK = int(dm.size()) * K;
                        for (int n_ = 0; n_ < NMc; n_++)
                            for (int j = 0; j < NB; j++) {
                                cd mu = 0.0;
                                for (int k = 0; k < N; k++) mu += std::pow(R * S_.e1[k], n_ + 1) * Xg_[size_t(k) * NB + j];
                                mu = mu / double(N) / std::pow(R, n_);
                                cd ax = 0.0;
                                for (int q = 0; q < LK; q++) ax += Amc[size_t(n_) * LK + q] * XL[size_t(q) * NB + j] * std::pow(R, -(q % K));
                                chk = std::max(chk, std::abs(ax - mu));
                                mumax = std::max(mumax, std::abs(mu));
                            }
                    }
                    chk /= std::max(mumax, 1e-300);
                    std::printf("    [cluster %.6f] local principal parts (spec section 29): %d poles (%d roots), rows", g, int(dm.size()), int(info.singular.size()));
                    for (double h : histL) std::printf(" %.1e", h);
                    std::printf("; the CHECK |contour sum - contour solution| %.1e (relative)\n", chk);
                    std::fflush(stdout);
                    if (STOR.count(Engine::kf(g))) {
                        auto& st0 = STOR[Engine::kf(g)];
                        (void)st0;
                    }
                    C = cl::with_grade_mom(k_.c, C, XL, g, dm, P_.kofs);
                    stc.Wq = Wq;
                    stc.X = cl::nodes_of(XL, dm, K, NB, S_.e1, R);
                    stc.has_label = !lab_.empty();
                    stc.label = lab_.empty() ? std::string() : lab_[0];
                    J_.close('}');
                    J_.close(']');
                    J_.kv("offlattice", "true");
                    J_.kv("cluster", "{\"members\":" + Json::nums(mem) + ",\"radius\":" + Json::num(R) + ",\"poles\":" + Json::nums(*Ctx::kv_get(k_.c.clus, g)) +
                                         ",\"merged\":" + Json::nums(Ctx::kv_get(k_.c.clabs, g) ? *Ctx::kv_get(k_.c.clabs, g) : std::vector<double>{}) +
                                         ",\"check\":" + Json::num(chk) + ",\"singular\":" + Json::nums(info.singular) + "}");
                    std::string ls = "[";
                    for (size_t q = 0; q < lab_.size(); q++) ls += (q ? "," : "") + Json::str(lab_[q]);
                    J_.kv("labels", ls + "]");
                    J_.kv("defect", Json::nums(dres));
                    J_.kv("defect_local", Json::nums(histL));
                    J_.kv("conditions", arr("conditions", Fc, {long(Fc.size() / nc), nc}));
                    J_.kv("X", arr("X", XL, {long(XL.size() / NB), NB}));
                    J_.kv("cent_after", cent());
                    offint_done = true;
                    break;
                }
                if (!isint) {
                    // the off-integer centre: the family's datum, the defect loop, the conditions (not imposed)
                    std::string lab;
                    for (auto& f : fams)
                        if (keq(f.second, gk)) { lab = f.first; break; }
                    if (lab.empty())                                // stage (ii) item 1: a gauge root, its datum pinned to zero
                        for (auto& f : gfams)
                            if (keq(f.second, gk)) { lab = f.first; break; }
                    double datum = (!lab.empty() && lab.rfind("gauge", 0) != 0 && offl_set.count(lab)) ? offl_set[lab] : 0.0;
                    std::vector<Vec> Wn;
                    Vec Xs = o.X0;
                    bool hf = false;
                    cd amp(0.0, 0.0);
                    if (rk) dt::offl_start(o.X0, o.Wq, nn, NB, S_.e1, Rcur_, !lab.empty(), datum, c_.cond_fit, eval(Cs, g), Wn, Xs, hf, amp);
                    st::DefectOut d;
                    st::defect_off(Xs, o.Minv, Wn, K, NB, S_.e1, Rcur_, std::max(c_.defect, 4), c_.robust, rowsfn(Cs, g), d);
                    C = with_grade(k_.c, C, d.X, g, P_.kofs, S_.e1, Rcur_);
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
                    din.nn = nn; din.NB = NB; din.rk = rk; din.R = Rcur_; din.e1 = S_.e1; din.X0 = o.X0; din.Wq = o.Wq; din.g = g;
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
                st::defect_int(Xg, o.Minv, Wn, act, Smax, K, NB, S_.e1, Rcur_, c_.defect, c_.robust, dfix, rowsfn(Cs, g), d);
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
                C = with_grade(k_.c, C, Xg, g, P_.kofs, S_.e1, Rcur_);
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
    if (CL) {
        // research round 1048 (spec section 31): the stage (ii) decisions by identity -- with N2A_FREEZE absent, written here (the J step's first
        // point; also the next J step's N2A_IDENT_PREV: the roots' positions)
        auto S_ = [](const std::string& x) { return Json::str(x); };
        auto strs = [&](const std::vector<std::string>& v) { std::string r = "["; for (size_t q = 0; q < v.size(); q++) r += (q ? "," : "") + S_(v[q]); return r + "]"; };
        std::vector<std::pair<std::string, double>> roots;
        auto rput = [&](const std::string& k0, double v) {
            for (auto& e : roots)
                if (e.first == k0) { e.second = v; return; }
            roots.emplace_back(k0, v);
        };
        for (auto* fv : {&fams, &gfams})
            for (auto& f : *fv) rput(f.first, f.second.v);
        for (auto& e : IDLOG)
            if (is_spur_(e.first)) rput(e.first, e.second.pos);
        for (auto& sp : SPUR) rput(sp.first, sp.second[0]);
        std::string d = "{\"J\":" + Json::num(c_.J) + ",\"roots\":{";
        for (size_t q = 0; q < roots.size(); q++) d += (q ? "," : "") + S_(roots[q].first) + ":" + Json::num(roots[q].second);
        d += "},\"product_steps\":" + strs(std::vector<std::string>(FZD_steps.begin(), FZD_steps.end())) + ",\"routed\":{";
        for (size_t q = 0; q < FZD_routed.size(); q++) d += (q ? "," : "") + S_(FZD_routed[q].first) + ":" + S_(FZD_routed[q].second);
        d += "},\"clusters\":{";
        for (size_t q = 0; q < FZD_clusters.size(); q++)
            d += (q ? "," : "") + S_(FZD_clusters[q].first) + ":{\"members\":" + strs(FZD_clusters[q].second.members) + ",\"key\":" + Json::num(FZD_clusters[q].second.key) + "}";
        d += "},\"spur\":{";
        bool first = true;
        for (auto& e : FZD_spur) {
            if (e.first.empty()) continue;
            d += (first ? "" : ",") + S_(e.first) + ":{";
            first = false;
            for (size_t q = 0; q < e.second.size(); q++) {
                auto& r = e.second[q].second;
                d += (q ? "," : "") + S_(e.second[q].first) + ":{\"status\":" + S_(r.status) + ",\"root\":" + Json::num(r.root) +
                     (r.has_storage ? ",\"storage\":" + S_(r.storage) : std::string()) + "}";
            }
            d += "}";
        }
        d += "},\"storage\":{";
        for (size_t q = 0; q < FZD_storage.size(); q++)
            d += (q ? "," : "") + S_(FZD_storage[q].first) + ":{\"poles\":" + strs(FZD_storage[q].second.poles) + ",\"merged\":" + strs(FZD_storage[q].second.merged) + "}";
        d += "},\"eps_s\":" + Json::num(EPS_S) + ",\"R0\":" + Json::num(R0) + ",\"identities\":{";
        for (size_t q = 0; q < IDLOG.size(); q++)
            d += (q ? "," : "") + S_(IDLOG[q].first) + ":{\"position\":" + Json::num(IDLOG[q].second.pos) + ",\"rule\":" + S_(IDLOG[q].second.rule) +
                 ",\"margin\":" + (IDLOG[q].second.has_margin ? Json::num(IDLOG[q].second.margin) : std::string("null")) + "}";
        d += "}}";
        J_.kv("stage2_decisions", d);
        int ndiff = 0;
        for (auto& x : FZ_LOG)
            if (x.frozen != x.own && (x.ps == PS_ || x.site.rfind("cluster at", 0) == 0)) ndiff++;
        if (!FZP.empty() && !FZ) {
            std::string tmp = FZP + ".tmp" + std::to_string(getpid());
            std::ofstream(tmp) << d;
            std::rename(tmp.c_str(), FZP.c_str());
            std::printf("  [freeze] stage (ii) decisions written (by identity): %s\n", FZP.c_str());
        } else if (FZ)
            std::printf("  [freeze] stage (ii) decisions replayed from %s: %d decision sites, %d where the point's own differs\n", FZP.c_str(), int(FZ_LOG.size()), ndiff);
        J_.kv("freeze_differ", std::to_string(ndiff));
    }
    C_final_ = C;
    fams_ = fams;
    gfams_ = gfams;
    spur_ = SPUR;
    spur_r_.clear();
    for (auto& sp : SPUR) spur_r_.emplace_back(sp.second[1], sp.first, sp.second[2]);
    std::sort(spur_r_.begin(), spur_r_.end());
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
inline std::vector<double> Engine::mkey_dn(Kernel& kc, const std::vector<Entry>& Cx, const std::vector<std::pair<int, double>>& gauge) {
    Ctx& c = kc.c;
    int K = c.K, NF = c.NF;
    const double n = c.n;
    auto A2 = [&](const Ser& x, const Ser& y) { return x.empty() ? y : y.empty() ? x : c.add(x, y); };
    auto M2 = [&](const Ser& x, const Ser& y) { return (x.empty() || y.empty()) ? c.emptyser() : c.mul(x, y); };
    auto S2 = [&](const Ser& x, double f) { return x.empty() ? x : c.scale(x, cd(f, 0.0)); };
    auto DR = [&](const Ser& x) { return x.empty() ? x : c.drad(x); };
    auto DT = [&](const Ser& x) { return x.empty() ? x : c.dth(x); };
    auto drad_dn = [&](const Ser& a) { return a.empty() ? a : kc.drad_dn(a); };
    // d / dn drad(f) at fixed keys, f's own derivative df
    auto ddrad = [&](const Ser& f, const Ser& df) { return A2(DR(df), drad_dn(f)); };
    Ser F[6], dF[6];
    kc.fields(Cx, c_.gmax + 1, F);
    for (int i = 0; i < 6; i++) dF[i] = c.emptyser();
    if (!gauge.empty()) {
        // the generator and its n-derivative: a piece at key l / n with moment 0 a_0 -> d / dn: moment 1 = a_0 (-l / n^2) (x^key's key moving)
        fm::Xi xi = fm::lie_xi(c, gauge, S_.TH), dxi{c.emptyser(), c.emptyser(), c.emptyser()};
        {
            auto one = [&](double key, double cf, const std::vector<double>& f, int e, double dk) {
                Piece v = c.zeros(1);
                if (K > 1)
                    for (int q = 0; q < NF; q++) v.a[size_t(NF) + q] = ct::mul(cd(cf, 0.0), cd(f[q], 0.0)) * dk;
                std::vector<std::pair<Key, Piece>> d;
                d.emplace_back(Key{key, false}, std::move(v));
                return c.make(std::move(d), true, e, BIG);
            };
            for (auto& lc : gauge) {
                int l = lc.first;
                double key = l / n, dk = -double(l) / (n * n);
                std::vector<double> co(NF), si(NF);
                for (int q = 0; q < NF; q++) { co[q] = std::cos(l * S_.TH[q]); si[q] = std::sin(l * S_.TH[q]); }
                dxi.xr = A2(dxi.xr, one(key, lc.second, co, 1, dk));
                dxi.xr_r = A2(dxi.xr_r, one(key, lc.second, co, 0, dk));
                dxi.xt = A2(dxi.xt, one(key, lc.second, si, 0, dk));
            }
        }
        std::vector<double> cot(NF), s3(NF);
        for (int q = 0; q < NF; q++) {
            cot[q] = std::cos(S_.TH[q]) / std::sin(S_.TH[q]);
            s3[q] = (6 * c_.J) * std::pow(std::sin(S_.TH[q]), 3.0);
        }
        Ser cotS = fm::theta_ser(c, cot), thS = fm::theta_ser(c, s3);
        // lie_L and its n-derivative (the same sequence as fm::lie_L, each term's product rule)
        auto L2 = [&](const Ser G[6], const Ser dG[6], Ser out[6], Ser dout[6]) {
            auto dot = [&](const Ser& f, const Ser& df, Ser& v, Ser& dv) {
                if (f.empty() && df.empty()) { v = c.emptyser(); dv = c.emptyser(); return; }
                Ser dr = DR(f), dt = DT(f);
                v = A2(M2(xi.xr, dr), M2(xi.xt, dt));
                dv = A2(A2(M2(dxi.xr, dr), M2(xi.xr, ddrad(f, df))), A2(M2(dxi.xt, dt), M2(xi.xt, DT(df))));
            };
            Ser w = A2(xi.xr_r, M2(xi.xt, cotS)), dw = A2(dxi.xr_r, M2(dxi.xt, cotS));
            for (int i = 0; i < 2; i++) {
                Ser a, da;
                dot(G[i], dG[i], a, da);
                out[i] = A2(a, M2(G[i], w));
                dout[i] = A2(da, A2(M2(dG[i], w), M2(G[i], dw)));
            }
            {
                Ser a, da;
                dot(G[2], dG[2], a, da);
                out[2] = A2(A2(a, DR(xi.xr)), S2(w, -1.0));
                dout[2] = A2(A2(da, ddrad(xi.xr, dxi.xr)), S2(dw, -1.0));
            }
            for (int i = 3; i < 5; i++) {
                const Ser& x = i == 3 ? xi.xr : xi.xt;
                const Ser& dx = i == 3 ? dxi.xr : dxi.xt;
                Ser a, da;
                dot(G[i], dG[i], a, da);
                Ser xd = DR(x), dxd = ddrad(x, dx);
                Ser pq = A2(M2(G[3], xd), G[4].empty() ? c.emptyser() : M2(G[4], DT(x)));
                Ser dpq = A2(A2(M2(dG[3], xd), M2(G[3], dxd)), A2(M2(dG[4], DT(x)), M2(G[4], DT(dx))));
                out[i] = A2(a, S2(pq, -1.0));
                dout[i] = A2(da, S2(dpq, -1.0));
            }
            {
                Ser a, da;
                dot(G[5], dG[5], a, da);
                out[5] = A2(a, M2(xi.xt, thS));
                dout[5] = A2(da, M2(dxi.xt, thS));
            }
        };
        double h = 1.0 / c_.gauge_steps;
        auto addc = [&](const Ser A_[6], const Ser B_[6], double cf, Ser o[6]) { for (int i = 0; i < 6; i++) o[i] = A2(A_[i], S2(B_[i], cf)); };
        for (int st = 0; st < c_.gauge_steps; st++) {
            Ser k[4][6], dk[4][6], T[6], dT[6];
            L2(F, dF, k[0], dk[0]);
            addc(F, k[0], h / 2, T); addc(dF, dk[0], h / 2, dT);
            L2(T, dT, k[1], dk[1]);
            addc(F, k[1], h / 2, T); addc(dF, dk[1], h / 2, dT);
            L2(T, dT, k[2], dk[2]);
            addc(F, k[2], h, T); addc(dF, dk[2], h, dT);
            L2(T, dT, k[3], dk[3]);
            const double wgt[4] = {h / 6, h / 3, h / 3, h / 6};
            for (int i = 0; i < 6; i++) {
                Ser f = F[i], df = dF[i];
                for (int j = 0; j < 4; j++) { f = A2(f, S2(k[j][i], wgt[j])); df = A2(df, S2(dk[j][i], wgt[j])); }
                F[i] = f; dF[i] = df;
            }
        }
    }
    // the evaluation at r_m (fvec's ms_at) and its n-derivative: the series' part ms_at(dG), the explicit x = r^n part
    // d / dn [x^t (ln x)^m / m!] = ln r [t x^t (ln x)^m / m! + x^t (ln x)^(m-1) / (m-1)!]
    double r = S_.md_rm, x = std::pow(r, n), lx = std::log(x), lr = std::log(r);
    std::vector<double> fac(K);
    {
        double f = 1.0;
        for (int m = 0; m < K; m++) { if (m) f *= m; fac[m] = std::pow(lx, m) / f; }
    }
    auto ms = [&](const Ser& G, bool expl, std::vector<cd>& tot) {
        tot.assign(NF, cd(0.0, 0.0));
        if (G.empty()) return false;
        for (auto& kv : G.d) {
            double xt = std::pow(x, kv.first.v);
            for (int q = 0; q < NF; q++) {
                cd s = 0.0;
                for (int m = 0; m < K; m++) {
                    cd a = kv.second.a[size_t(m) * NF + q];
                    s += expl ? lr * (kv.first.v * fac[m] + (m ? fac[m - 1] : 0.0)) * a : fac[m] * a;
                }
                tot[q] += xt * s;
            }
        }
        double re = std::pow(r, double(G.e));
        for (auto& z : tot) z *= re;
        return true;
    };
    std::vector<cd> dval[6], ddval[6];
    for (int f = 0; f < 6; f++) {
        std::vector<cd> a, b;
        ms(dF[f], false, a);
        ms(F[f], true, b);
        dval[f].assign(NF, 0.0);
        for (int q = 0; q < NF; q++) dval[f][q] = a[q] + b[q];
        Ser G = DR(F[f]), dG = ddrad(F[f], dF[f]);
        ms(dG, false, a);
        ms(G, true, b);
        ddval[f].assign(NF, 0.0);
        for (int q = 0; q < NF; q++) ddval[f][q] = a[q] + b[q];
    }
    std::vector<double> out;
    size_t nk = S_.md_key.size() / 3;
    int nt = S_.md_nthm;
    for (size_t q = 0; q < nk; q++) {
        int f = S_.md_key[3 * q], d = S_.md_key[3 * q + 1];
        const std::vector<cd>& fv = d ? ddval[f] : dval[f];
        double s = 0.0;
        for (int t = 0; t < nt; t++) {
            cd y = 0.0;
            for (int k = 0; k < NF; k++) y += fv[k] * S_.md_I[size_t(t) * NF + k];
            s += y.real() * S_.md_p[q * size_t(nt) + t];
        }
        out.push_back(s);
    }
    return out;
}

inline void Engine::mk_dn_check(double h) {
    if (!S_.has_md) throw std::runtime_error("mk: mk-dn: the bundle has no matching keys");
    std::vector<double> cents;
    for (int gg = 1; gg <= c_.gdata; gg++) cents.push_back(gg);
    for (double g : order_last_) {
        bool dat = false;
        for (int gg = 1; gg <= c_.gdata; gg++)
            if (std::abs(g - gg) < 1e-9) dat = true;
        if (!dat) cents.push_back(g);
    }
    size_t nk = S_.md_key.size() / 3;
    auto mk_at = [&](double dn) {
        double n0 = k_.c.n;
        k_.c.n = n0 + dn;
        auto v = fvec(k_, C_final_, c_.gauge, cents);
        k_.c.n = n0;
        return std::vector<cd>(v.end() - long(nk), v.end());
    };
    auto ex = mkey_dn(k_, C_final_, c_.gauge);
    auto b0 = mk_at(0.0), p1 = mk_at(h), m1 = mk_at(-h), p2 = mk_at(2 * h), m2 = mk_at(-2 * h);
    std::ofstream f(out_ + "/mk_dn.json");
    f.precision(17);
    f << "{\"h\": " << h << ", \"n\": " << k_.c.n << ", \"keys\": [";
    for (size_t q = 0; q < nk; q++) {
        double D1 = (p1[q].real() - m1[q].real()) / (2 * h), D2 = (p2[q].real() - m2[q].real()) / (4 * h);
        f << (q ? ", " : "") << "[" << b0[q].real() << ", " << ex[q] << ", " << D1 << ", " << D2 << "]";
    }
    f << "]}\n";
}

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
    for (auto& sp : spur_r_) {                                      // spec section 27: the spurious members' l = 0 conditions (stored_member)
        double mem_ = std::get<0>(sp), key_ = std::get<2>(sp), ck;
        std::vector<double> cm;
        std::vector<cd> Sq;
        int off = 0;
        if (!kc.c.clus.empty() && kc.c.clus_at(key_, ck, cm)) {
            Sq = project(kc.c, kc.out, key_, 1, P_.cfam, P_.cdE, P_.Wc, true);
            for (size_t i = 1; i < cm.size(); i++)
                if (std::abs(cm[i] - mem_) < std::abs(cm[off] - mem_)) off = int(i);
        } else
            Sq = project(kc.c, kc.out, mem_, 1, P_.cfam, P_.cdE, P_.Wc, true);
        for (int m = 0; m < 2; m++)
            for (int q = 0; q < nc; q++) v.push_back(Sq[size_t(off * K + m) * nc + q] / S_.rwc[q]);
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
    Rcur_ = c_.R;
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
        for (auto& sp : spur_r_)
            for (int m = 0; m < 2; m++)
                for (int q = 0; q < P_.nc; q++)
                    lab.push_back("eq scond " + std::get<1>(sp) + " " + fmt("%.6f", std::get<0>(sp)) + " m" + std::to_string(m) + " " + S_.condlab[q]);
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

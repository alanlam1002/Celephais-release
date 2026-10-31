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

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <sys/stat.h>

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
    auto gs = [&](const std::string& key) { auto i = B.cfg.find(key); return i == B.cfg.end() ? std::string() : i->second; };
    c.R = B.getd("R", 0.015); c.N = B.geti("N", 96); c.gmax = B.geti("gmax", 8); c.gdata = B.geti("gdata", 1); c.top = B.getd("TOP", 10);
    c.passes = B.geti("passes", 1); c.pass_tol = B.getd("pass_tol", 1e-10); c.newton = B.geti("newton", 6); c.defect = B.geti("defect", 8);
    c.condtol = B.getd("condtol", 0.0); c.meff_chunk = B.geti("meff_chunk", 34); c.offl_lmax = B.geti("offl_lmax", 8); c.J = B.getd("J", 0.0);
    c.gauge_steps = B.geti("gauge_steps", 2); c.cond_fit = B.geti("cond_fit", 0); c.notgt = B.geti("notarget", 0);
    c.cond_full = B.geti("cond_full", 0); c.robust = B.geti("defect_stop_robust", 0); c.offl_auto = gs("offl") == "auto";
    c.meff_mode = gs("meff_mode") == "jvp" ? 1 : 0;
    c.tgt = gs("tgt");
    S.j_br0 = B.geti("j_br0", -1);
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

class Engine {
  public:
    Engine(Kernel& k, Proj& P, const Setup& S, const Cfg& c, const std::string& out) : k_(k), P_(P), S_(S), c_(c), out_(out) {
        mkdir(out.c_str(), 0755);
        mkdir((out + "/arrays").c_str(), 0755);
    }
    void run();

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
};

inline void Engine::run() {
    if (const char* t = std::getenv("MK_TRACE_CANON")) k_.c.trace_above = std::atof(t);
    int nn = P_.nn, K = k_.c.K, N = c_.N, NB = nn, nc = P_.nc;
    size_t nm = size_t(NB) * NB, nmc = size_t(nc) * NB;
    std::vector<Entry> C = S_.C0;
    std::map<double, std::pair<Vec, Vec>> cache;                    // meff_cache: the converged operator per integer centre
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
            std::vector<double> newton_hist;
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
                    din.nn = nn; din.NB = NB; din.rk = rk; din.R = c_.R; din.e1 = S_.e1; din.X0 = o.X0; din.Wq = o.Wq; din.g = g;
                    din.notgt = c_.notgt; din.cond_fit = c_.cond_fit; din.cond_full = c_.cond_full; din.condtol_fixed = c_.condtol; din.lc = S_.lc;
                    din.j_br0 = S_.j_br0;
                    din.warm = c_warm.count(kf) && int(c_warm[kf].size()) == rk;
                    if (din.warm) din.cwarm = c_warm[kf];
                    dt::data_rule(din, eval(C, g), dout);
                    c_warm[kf] = dout.c;
                    Xg = dout.Xg;
                    Wn = dout.Wq;
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
                        D.kv("alt", "{\"poly_deg\":" + std::to_string(dout.poly_deg) + ",\"poly\":" + Json::nums(dout.poly) + ",\"alt_tau\":" +
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
                newton_hist.push_back(d.rows_after);
                J_.kv("defect", Json::nums(d.hist));
                J_.kv("defect_drift", Json::num(d.drift));
                J_.kv("rows_after", Json::num(d.rows_after));
                J_.kv("X", arr("X", moments(Xg), {K, NB}));
                J_.kv("X_nodes", arr("X_nodes", Xg, {N, NB}));
                J_.close('}');
                if (st::newton_continue(newton_hist, Smax, newt, c_.newton, isint, c_.defect)) {
                    Xk = Xg;
                    continue;
                }
                break;
            }
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
        J_.kv("change", Json::num(dch));
        J_.kv("cent_end", cent());
        J_.close('}');
        if (ps && dch < c_.pass_tol) { ps++; break; }
    }
    J_.close(']');
    J_.kv("families", families_json);
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

}  // namespace en
}  // namespace mk

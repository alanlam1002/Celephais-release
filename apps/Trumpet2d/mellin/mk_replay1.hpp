// Mellin throat recursion (code round 463; research rounds 1074 / 1089 / 1092, the forward-mode replay, STAGE 1(c), part 3): ONE ENGINE STEP
// REPLAYED ON DUALS -- at an integer, non-cluster centre (production's --cond-fit data rule), the step the forward took, with the tangent of
// every quantity along one direction, given the tangent of the step's input (the stripped series):
//   the sources         S = the stripped series' rows at g: the kernel's exact linear response (Kernel::run with dC) -> dS, dSn
//   the operator        Ms = meff(base): its tangent by central differences of meff along the base's tangent (the build has no decision)
//   the contour         mk_contour_dual (dMinv, dX0, the residue directions' span)
//   the data rule       mk_data_dual (the amplitudes, the start; the data fix)
//   the defect          mk_steps_dual (the corrections' count the value's; the rows' tangent the kernel's linear response)
//   Newton              as many re-linearisations as the forward took (its decision), each with the operator and the sources at X_k
// The decisions are the forward's: the residue rank at each Newton iteration and their number (captured in Engine::RpCap).  The replayed
// value is checked against the forward's block (it must agree to round-off); the block's tangent is written for the gate against the
// frozen-decision finite differences (two forwards at p +- h dumping this step's input and block: MK_STEP_DUMP).
//   MK_STEP_DUMP=G:PASS:PREFIX     the step at centre G in pass PASS: PREFIX.Cs (the stripped series), PREFIX.X (the block's moments)
//   MK_STEP_REPLAY=G:PASS:DCS:OUT  the same step replayed with the input tangent DCS (entries, the dump format): OUT (the block, its tangent)
#pragma once
#include "mk_contour_dual.hpp"
#include "mk_data_dual.hpp"
#include "mk_steps_dual.hpp"

namespace mk {
namespace en {

// entries in text: per entry 'j t isint nb K' then nb x K complex pairs
inline void write_entries(const std::string& p, const std::vector<Entry>& C) {
    FILE* f = std::fopen(p.c_str(), "w");
    if (!f) throw std::runtime_error("mk: cannot write " + p);
    std::fprintf(f, "%zu\n", C.size());
    for (auto& e : C) {
        std::fprintf(f, "%d %.17g %d %d %zu\n", e.j, e.t.v, e.t.isint ? 1 : 0, e.nb, e.m.size());
        for (auto& z : e.m) std::fprintf(f, "%.17g %.17g\n", z.real(), z.imag());
    }
    std::fclose(f);
}
inline std::vector<Entry> read_entries(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "r");
    if (!f) throw std::runtime_error("mk: cannot read " + p);
    size_t n = 0;
    if (std::fscanf(f, "%zu", &n) != 1) throw std::runtime_error("mk: bad entries file " + p);
    std::vector<Entry> C(n);
    for (auto& e : C) {
        int isint = 0;
        size_t nm = 0;
        if (std::fscanf(f, "%d %lf %d %d %zu", &e.j, &e.t.v, &isint, &e.nb, &nm) != 5) throw std::runtime_error("mk: bad entry in " + p);
        e.t.isint = isint != 0;
        e.m.resize(nm);
        for (auto& z : e.m) {
            double a, b;
            if (std::fscanf(f, "%lf %lf", &a, &b) != 2) throw std::runtime_error("mk: bad moments in " + p);
            z = cd(a, b);
        }
    }
    std::fclose(f);
    return C;
}
// C + h dC (entries matched by column and centre; dC's other entries appended, scaled)
inline std::vector<Entry> axpy_entries(const std::vector<Entry>& C, double h, const std::vector<Entry>& dC) {
    std::vector<Entry> o = C;
    for (auto& d : dC) {
        bool hit = false;
        for (auto& e : o)
            if (e.j == d.j && keq(e.t, d.t)) {
                for (size_t i = 0; i < e.m.size() && i < d.m.size(); i++) e.m[i] += h * d.m[i];
                hit = true;
                break;
            }
        if (!hit) {
            Entry n = d;
            for (auto& z : n.m) z *= h;
            o.push_back(n);
        }
    }
    return o;
}
inline double maxabs_entries(const std::vector<Entry>& C) {
    double m = 0.0;
    for (auto& e : C)
        for (auto& z : e.m) m = std::max(m, std::abs(z));
    return m;
}

inline void Engine::replay_step(double g, const std::string& dcs_path, const std::string& outp) {
    const RpCap& r = rp_;
    if (!r.isint || r.iscl) throw std::runtime_error("mk replay: stage 1(c) replays an integer, non-cluster centre only");
    if (r.from_cache) throw std::runtime_error("mk replay: the step's operator came from the cache (an earlier pass's): replay pass 0");
    const int ND = 1, K = k_.c.K, nn = P_.nn, NB = nn, N = int(S_.e1.size());
    const double R = r.R, top = c_.top;
    const Vec& e1 = S_.e1;
    std::vector<Entry> dCs = read_entries(dcs_path);
    auto act = ct::active(g, P_.kofs);
    // the block's tangent entries (aligned-g columns) from node values dX
    auto block = [&](const Vec& dX) {
        auto Mo = ct::moments_of(dX, K, nn, e1, R);
        std::vector<Entry> o;
        for (int j = 0; j < nn; j++) {
            Key t = ckey(k_.c, g + P_.kofs[j]);
            if (!(t.v >= 1)) continue;
            std::vector<cd> m(K);
            for (int k = 0; k < K; k++) m[k] = Mo[size_t(k) * nn + j];
            o.push_back(Entry{j, t, 1, std::move(m)});
        }
        return o;
    };
    auto tan_at = [&](const Vec* dX) {                                     // the input's tangent, plus the block's when X is set
        std::vector<Entry> o = dCs;
        if (dX)
            for (auto& b : block(*dX)) o.push_back(b);
        return o;
    };
    auto rows_lin = [&](const std::vector<Entry>& Cx, const std::vector<Entry>& dCx, Vec& dR) {
        k_.run(Cx, &dCx, top);
        dR = project(k_.c, k_.dout, g, 1, P_.gfam, P_.dE, P_.W, false);
        return project(k_.c, k_.out, g, 1, P_.gfam, P_.dE, P_.W, false);
    };
    auto meff_at = [&](const std::vector<Entry>& Cx) {
        Vec Ms, Mc;
        meff(k_, P_, Cx, g, top, c_.meff_mode, c_.meff_h, c_.meff_chunk, Ms, Mc);
        return Ms;
    };
    sd::RowsT rowsT = [&](const Vec& X, const sd::Tv& dX, sd::Tv& dR) {
        Vec d;
        Vec v = rows_lin(with_grade(k_.c, r.Cs, X, g, P_.kofs, e1, R), tan_at(&dX[0]), d);
        dR.assign(1, d);
        return v;
    };
    Vec Xk, dXk;
    std::vector<double> report;
    for (int it = 0; it < r.nits; it++) {
        std::vector<Entry> Cb = it == 0 ? r.Cs : with_grade(k_.c, r.Cs, Xk, g, P_.kofs, e1, R);
        std::vector<Entry> dCb = tan_at(it == 0 ? nullptr : &dXk);
        // the operator and its tangent (central differences of meff)
        Vec Ms = meff_at(Cb);
        double hm = 1e-6 * std::max(1.0, maxabs_entries(Cb)) / std::max(maxabs_entries(dCb), 1e-300);
        Vec Mp = meff_at(axpy_entries(Cb, hm, dCb)), Mm = meff_at(axpy_entries(Cb, -hm, dCb)), dMs(Ms.size());
        for (size_t i = 0; i < Ms.size(); i++) dMs[i] = (Mp[i] - Mm[i]) / (2 * hm);
        // the sources at the nodes and their tangent
        Vec Sn, dSn(size_t(N) * NB);
        if (it == 0) {
            Vec dS, S = rows_lin(r.Cs, dCs, dS);
            Sn = ct::nodes_of(S, K, NB, e1, R);
            dSn = ct::nodes_of(dS, K, NB, e1, R);
        } else {
            Vec dr, rr = rows_lin(Cb, dCb, dr);
            Vec nd = ct::nodes_of(rr, K, NB, e1, R), dnd = ct::nodes_of(dr, K, NB, e1, R);
            Sn.assign(size_t(N) * NB, 0.0);
            size_t nm = size_t(NB) * NB;
            for (int k = 0; k < N; k++) {
                cd zz = R * e1[k], z2 = zz * zz;
                for (int i = 0; i < NB; i++) {
                    cd s = 0.0, ds = 0.0;
                    for (int j = 0; j < NB; j++) {
                        size_t a = size_t(i) * NB + j;
                        cd M = Ms[a] + zz * Ms[nm + a] + z2 * Ms[2 * nm + a], dM = dMs[a] + zz * dMs[nm + a] + z2 * dMs[2 * nm + a];
                        s += M * Xk[size_t(k) * NB + j];
                        ds += dM * Xk[size_t(k) * NB + j] + M * dXk[size_t(k) * NB + j];
                    }
                    Sn[size_t(k) * NB + i] = nd[size_t(k) * NB + i] - s;
                    dSn[size_t(k) * NB + i] = dnd[size_t(k) * NB + i] - ds;
                }
            }
        }
        // the contour (the forward's rank) and its tangent
        ct::Contour o;
        ct::contour(Ms, NB, S_.rw, S_.cw, e1, R, act, Sn, r.rks[it], o);
        cdl::ContourT tc;
        cdl::contour_t(Ms, {dMs}, NB, e1, R, Sn, {dSn}, o, ND, tc);
        // the data rule (cond-fit) and its tangent
        Vec Xs = o.X0, dXs = tc.dX0[0];
        std::vector<Vec> Wn;
        std::vector<std::vector<Vec>> dWn(ND);
        dt::DataIn din = r.din;
        dt::DataOut dout;
        const sd::FixT* fixp = nullptr;
        sd::FixT fix;
        if (o.rk) {
            if (!r.have_din) throw std::runtime_error("mk replay: residues without the forward's data rule");
            din.rk = o.rk; din.X0 = o.X0; din.Wq = o.Wq; din.R = R;
            dt::Eval ev;
            ev.cond = [](const Vec&) { return Vec(); };
            ev.gate = [](const Vec*, double) { return 0.0; };
            dt::data_rule(din, ev, dout);
            ddl::DataT td;
            ddl::data_rule_t(din, dout, tc.dX0, tc.dWq, {}, ND, td);
            Xs = dout.Xg;
            dXs = td.dXg[0];
            Wn = dout.Wq;
            dWn = td.dWn;
            if (c_.cond_fit) {
                fix = [din, dout, dWn](const Vec& X, const sd::Tv& dX, sd::Tv& dO) { return ddl::data_fix_t(X, dX, din, dout, dWn, {}, 1, dO); };
                fixp = &fix;
            }
        }
        sd::DefectT dd;
        sd::defect_int_t(Xs, {dXs}, o.Minv, tc.dMinv, Wn, dWn, act, r.Smax, K, NB, e1, R, c_.defect, c_.robust, fixp, rowsT, ND, dd);
        Xk = dd.X;
        dXk = dd.dX[0];
        report.push_back(double(o.rk));
        report.push_back(double(dd.corrections));
    }
    // the value against the forward's block; the block's tangent (moments)
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < Xk.size(); i++) { num += std::norm(Xk[i] - r.Xg[i]); den += std::norm(r.Xg[i]); }
    Vec X_m = ct::moments_of(Xk, K, nn, e1, R), dX_m = ct::moments_of(dXk, K, nn, e1, R);
    FILE* f = std::fopen(outp.c_str(), "w");
    std::fprintf(f, "# replay at g %.9f: %d Newton iterations (rank, corrections:", g, r.nits);
    for (size_t i = 0; i < report.size(); i += 2) std::fprintf(f, " %d/%d", int(report[i]), int(report[i + 1]));
    std::fprintf(f, "); the value against the forward's block %.3e (relative)\n%d %d\n", std::sqrt(num / std::max(den, 1e-300)), K, NB);
    for (int k = 0; k < K; k++)
        for (int j = 0; j < NB; j++) {
            size_t a = size_t(k) * NB + j;
            std::fprintf(f, "%.17g %.17g %.17g %.17g\n", X_m[a].real(), X_m[a].imag(), dX_m[a].real(), dX_m[a].imag());
        }
    std::fclose(f);
    std::printf("    [replay g %.6f] %d Newton iterations; the value against the forward's %.3e; -> %s\n", g, r.nits,
                std::sqrt(num / std::max(den, 1e-300)), outp.c_str());
}

inline void Engine::step_hook(double g) {
    if (const char* e = std::getenv("MK_STEP_DUMP")) {
        std::string s(e), pfx = s.substr(s.find(':', s.find(':') + 1) + 1);
        write_entries(pfx + ".Cs", rp_.Cs);
        Vec X_m = ct::moments_of(rp_.Xg, k_.c.K, P_.nn, S_.e1, rp_.R);
        FILE* f = std::fopen((pfx + ".X").c_str(), "w");
        std::fprintf(f, "%d %d\n", k_.c.K, P_.nn);
        for (auto& z : X_m) std::fprintf(f, "%.17g %.17g\n", z.real(), z.imag());
        std::fclose(f);
        std::printf("    [step dump g %.6f] %d Newton iterations, ranks", g, rp_.nits);
        for (int k : rp_.rks) std::printf(" %d", k);
        std::printf(" -> %s.Cs / .X\n", pfx.c_str());
    }
    if (const char* e = std::getenv("MK_STEP_REPLAY")) {
        std::string s(e);
        size_t a = s.find(':'), b = s.find(':', a + 1), c = s.find(':', b + 1);
        replay_step(g, s.substr(b + 1, c - b - 1), s.substr(c + 1));
    }
    if (tn_.on) tan_step(g);                                       // (research round 1093: the whole forward on duals, mk_replay2.hpp)
}

}  // namespace en
}  // namespace mk

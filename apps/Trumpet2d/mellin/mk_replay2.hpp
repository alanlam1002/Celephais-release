// Mellin throat recursion (code round 463; research round 1093, the forward-mode replay, STAGE 2, part 1): the WHOLE FORWARD on duals --
// every centre of every pass the forward took, the series' tangent carried from step to step, so the tangent of the final series along one
// direction is exact up to the operator's (meff's central differences: stage 2 part 3 replaces them).  Covered: integer centres (the
// --cond-fit data rule) and off-integer centres (the off-integer start, mk_data_dual offl_start_t, and the defect loop, defect_off_t), outside
// clusters (a cluster step throws: stage 2 part 2); the operator as the forward chose it -- built at the stripped series, from the cache (an
// earlier pass's, its tangent cached with it), or re-centred from the last integer centre's (its tangent re-centred, linear).
// The decisions are the forward's, captured per step (RpCap: the residue rank per Newton iteration, their number, the cache, the data rule's
// input, the family's label and datum); each step's replayed value is checked against the forward's block.
//   MK_TAN=data:NAME@tT | offl:FAMILY    the direction: a datum of the series (set_data's entry, moment 0) or an off-lattice family's datum
//   MK_TAN_OUT=FILE                      the final series (before the gauge flow) and its tangent (entries: value, tangent)
//   MK_SERIES_OUT=FILE                   the final series alone (the finite-difference gate's two forwards)
#pragma once

namespace mk {
namespace en {

// research round 1094 (spec section 48, stage 2 part 3): the EXACT tangent of the forward's own operator.  With meff's finite-difference mode
// (production: Ms[., ., col] = (R(C + h e_col) - R(C - h e_col)) / 2h, e_col the unit at the block column's moment 2) its derivative along dC
// is (R'(C + h e_col)[dC] - R'(C - h e_col)[dC]) / 2h -- the kernel's exact linear response at the two perturbed series, on meff's own batch;
// no second-order response is needed (the operator IS a difference).  Only dMs (the conditions' Mc is not differentiated).
inline void meff_dual(Kernel& kr, const Proj& P, const std::vector<Entry>& Cx, const std::vector<Entry>& dCx, double g, double top, double h, int chunk,
                      std::vector<cd>& dMs) {
    Ctx& c = kr.c;
    int K = c.K, nn = P.nn, NB = P.nb * nn;
    dMs.assign(size_t(3) * NB * NB, 0.0);
    std::vector<std::pair<int, Key>> colsb;
    for (int j = 0; j < nn; j++) colsb.emplace_back(j, ckey(c, g + P.kofs[j]));
    for (size_t c0 = 0; c0 < colsb.size(); c0 += size_t(chunk)) {
        size_t c1 = std::min(colsb.size(), c0 + size_t(chunk));
        int nq = int(c1 - c0), Bt = 2 * nq;
        std::vector<Entry> Cp = Cx;
        for (int q = 0; q < nq; q++) {
            int j = colsb[c0 + q].first;
            const Key& t = colsb[c0 + q].second;
            int at = -1;
            for (size_t e = 0; e < Cp.size(); e++)
                if (Cp[e].j == j && keq(Cp[e].t, t)) { at = int(e); break; }
            if (at < 0) { Cp.push_back(Entry{j, t, 1, std::vector<cd>(K, 0.0)}); at = int(Cp.size()) - 1; }
            Entry& en = Cp[at];
            if (en.nb == 1) {
                std::vector<cd> arr(size_t(Bt) * K);
                for (int b = 0; b < Bt; b++)
                    for (int k = 0; k < K; k++) arr[size_t(b) * K + k] = en.m[k];
                en.m = std::move(arr);
                en.nb = Bt;
            }
            en.m[size_t(2 * q) * K + 2] += h;
            en.m[size_t(2 * q + 1) * K + 2] -= h;
        }
        // the tangent broadcast to the batch (every batch member moves along the same dC)
        std::vector<Entry> dCb;
        for (auto& e : dCx) {
            if (e.nb != 1) throw std::runtime_error("meff_dual: a batched tangent");
            std::vector<cd> arr(size_t(Bt) * K);
            for (int b = 0; b < Bt; b++)
                for (int k = 0; k < K && k < int(e.m.size()); k++) arr[size_t(b) * K + k] = e.m[k];
            dCb.push_back(Entry{e.j, e.t, Bt, std::move(arr)});
        }
        kr.run(Cp, &dCb, top);
        auto dRg = project(c, kr.dout, g, Bt, P.gfam, P.dE, P.W, false);
        for (int q = 0; q < nq; q++) {
            int col = colsb[c0 + q].first;
            for (int i = 0; i < nn; i++)
                for (int m = 0; m < 3; m++) {
                    size_t ap = (size_t(2 * q) * K + (2 - m)) * nn + i, am = (size_t(2 * q + 1) * K + (2 - m)) * nn + i;
                    dMs[(size_t(m) * NB + i) * NB + col] = (dRg[ap] - dRg[am]) / (2 * h);
                }
        }
    }
}

inline void Engine::tan_init(std::vector<Entry>& C) {
    const char* e = std::getenv("MK_TAN");
    if (!e) return;
    std::string s(e);
    size_t c = s.find(':');
    if (c == std::string::npos) throw std::runtime_error("MK_TAN: data:NAME@tT or offl:FAMILY");
    tn_ = TanMode();
    tn_.on = true;
    tn_.kind = s.substr(0, c);
    tn_.name = s.substr(c + 1);
    if (tn_.kind == "data") {
        size_t a = tn_.name.find("@t");
        std::string col = tn_.name.substr(0, a);
        double t = std::atof(tn_.name.substr(a + 2).c_str());
        bool found = false;
        for (auto& en : C)
            if (S_.collab[en.j] == col && std::abs(en.t.v - t) < 1e-12) {
                std::vector<cd> m(en.m.size(), 0.0);
                m[0] = 1.0;
                tn_.dC.push_back(Entry{en.j, en.t, en.nb, std::move(m)});
                found = true;
            }
        if (!found) throw std::runtime_error("MK_TAN: no series entry " + tn_.name);
    } else if (tn_.kind != "offl")
        throw std::runtime_error("MK_TAN: data:NAME@tT or offl:FAMILY");
    std::printf("  [tangent] the whole forward on duals along %s:%s (research round 1093, stage 2)\n", tn_.kind.c_str(), tn_.name.c_str());
}

inline void Engine::tan_step(double g) {
    const RpCap& r = rp_;
    if (r.iscl) throw std::runtime_error("mk tangent: a cluster step (stage 2 part 2)");
    const int ND = 1, K = k_.c.K, nn = P_.nn, NB = nn, N = int(S_.e1.size());
    const size_t nm = size_t(NB) * NB;
    const double R = r.R, top = c_.top;
    const Vec& e1 = S_.e1;
    const double kf = std::nearbyint(g * 1e9) / 1e9;
    const std::vector<Entry> dCs = strip(tn_.dC, g);
    auto act = ct::active(g, P_.kofs);
    auto tan_at = [&](const Vec* dX, double Rw) { return dX ? with_grade(k_.c, dCs, *dX, g, P_.kofs, e1, Rw) : dCs; };
    // the value by the forward's own evaluation (bit for bit: a step amplifies round-off by up to ~1e10 at the rank-2 integer centre 2 in later
    // passes), the tangent by the kernel's exact linear response
    auto rows_lin = [&](const std::vector<Entry>& Cx, const std::vector<Entry>& dCx, Vec& dR) {
        k_.run(Cx, &dCx, top);
        dR = project(k_.c, k_.dout, g, 1, P_.gfam, P_.dE, P_.W, false);
        k_.run(Cx, nullptr, top);
        return project(k_.c, k_.out, g, 1, P_.gfam, P_.dE, P_.W, false);
    };
    auto meff_t = [&](const std::vector<Entry>& Cx, const std::vector<Entry>& dCx, Vec& dMs) {
        Vec Ms, Mc;
        meff(k_, P_, Cx, g, top, c_.meff_mode, c_.meff_h, c_.meff_chunk, Ms, Mc);
        double dn = maxabs_entries(dCx);
        dMs.assign(Ms.size(), 0.0);
        // research round 1094: the exact tangent of meff's finite-difference operator (MK_TAN_MEFF=fd: the round-1093 central difference)
        static const bool OP_FD = std::getenv("MK_TAN_MEFF") && std::string(std::getenv("MK_TAN_MEFF")) == "fd";
        if (dn > 0 && c_.meff_mode == 0 && !OP_FD) {
            meff_dual(k_, P_, Cx, dCx, g, top, c_.meff_h, c_.meff_chunk, dMs);
            dn = 0.0;
        }
        if (dn > 0) {
            static const double HM = std::getenv("MK_TAN_MEFF_H") ? std::atof(std::getenv("MK_TAN_MEFF_H")) : 1e-4;   // (the operator's FD step, relative: meff is itself a finite difference -- 1e-6 round-off-limited, 2.6e-5 on the PS2 gate; 1e-4 at the gate's own noise 4e-6; the exact tangent is stage 2 part 3)
            double hm = HM * std::max(1.0, maxabs_entries(Cx)) / dn;
            Vec Mp, Mm, Mc2;
            meff(k_, P_, axpy_entries(Cx, hm, dCx), g, top, c_.meff_mode, c_.meff_h, c_.meff_chunk, Mp, Mc2);
            meff(k_, P_, axpy_entries(Cx, -hm, dCx), g, top, c_.meff_mode, c_.meff_h, c_.meff_chunk, Mm, Mc2);
            for (size_t i = 0; i < Ms.size(); i++) dMs[i] = (Mp[i] - Mm[i]) / (2 * hm);
        }
        return Ms;
    };
    sd::RowsT rowsT = [&](const Vec& X, const sd::Tv& dX, sd::Tv& dR) {
        Vec d;
        Vec v = rows_lin(with_grade(k_.c, r.Cs, X, g, P_.kofs, e1, R), tan_at(&dX[0], R), d);
        dR.assign(1, d);
        return v;
    };
    Vec Xk, dXk, Ms_last, dMs_last, Xs_dbg, Xlast_own;
    const int nits = r.offint ? 1 : r.nits;
    for (int it = 0; it < nits; it++) {
        // the operator, as the forward chose it
        Vec Ms, dMs;
        const bool intpath = r.isint || !tn_.have_mref;
        if (intpath) {
            if (it == 0 && r.from_cache) {
                if (!tn_.cache.count(kf)) throw std::runtime_error("mk tangent: a cached operator without its tangent");
                Ms = tn_.cache[kf].first;
                dMs = tn_.cache[kf].second;
            } else if (it == 0) {
                Ms = meff_t(r.Cs, dCs, dMs);
                tn_.cache[kf] = {Ms, dMs};
            } else {
                Ms = meff_t(with_grade(k_.c, r.Cs, Xk, g, P_.kofs, e1, c_.R), tan_at(&dXk, c_.R), dMs);
            }
            tn_.have_mref = true;
            tn_.mref_g = g;
            tn_.mref_s = Ms;
            tn_.dmref_s = dMs;
        } else {
            recentre(tn_.mref_s, Ms, nm, g - tn_.mref_g);
            recentre(tn_.dmref_s, dMs, nm, g - tn_.mref_g);
        }
        Ms_last = Ms;
        dMs_last = dMs;
        // the sources at the nodes and their tangent
        Vec Sn, dSn(size_t(N) * NB);
        if (it == 0) {
            Vec dS, S = rows_lin(r.Cs, dCs, dS);
            Sn = ct::nodes_of(S, K, NB, e1, R);
            dSn = ct::nodes_of(dS, K, NB, e1, R);
        } else {
            Vec dr, rr = rows_lin(with_grade(k_.c, r.Cs, Xk, g, P_.kofs, e1, R), tan_at(&dXk, R), dr);
            Vec nd = ct::nodes_of(rr, K, NB, e1, R), dnd = ct::nodes_of(dr, K, NB, e1, R);
            const Vec Sn_fwd = st::newton_sources(Xk, Ms, K, NB, e1, R, rowsfn(r.Cs, g));   // (the value: the forward's own)
            Sn.assign(size_t(N) * NB, 0.0);
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
            Sn = Sn_fwd;
        }
        // the contour at the forward's rank, and its tangent
        ct::Contour o;
        ct::contour(Ms, NB, S_.rw, S_.cw, e1, R, act, Sn, r.rks[it], o);
        cdl::ContourT tc;
        cdl::contour_t(Ms, {dMs}, NB, e1, R, Sn, {dSn}, o, ND, tc);
        sd::DefectT dd;
        if (r.offint) {
            // the off-integer start (the family's datum on its first direction) and the defect loop
            std::vector<Vec> Wn;
            Vec Xs = o.X0;
            Vec dXs = tc.dX0[0];
            std::vector<sd::Tv> dWn(ND);
            if (o.rk) {
                bool hf = false;
                cd amp;
                const bool labelled = !r.lab.empty();
                dt::offl_start(o.X0, o.Wq, nn, NB, e1, R, labelled, r.datum, c_.cond_fit, eval(r.Cs, g), Wn, Xs, hf, amp);
                if (hf) throw std::runtime_error("mk tangent: an amplitude fixed by the conditions (not --cond-fit)");
                ddl::OfflT to;
                const double dd0 = (tn_.kind == "offl" && r.lab == tn_.name) ? 1.0 : 0.0;
                ddl::offl_start_t(o.X0, o.Wq, nn, NB, e1, R, labelled, r.datum, tc.dX0, tc.dWq, {dd0}, ND, Wn, to);
                dXs = to.dXg[0];
                dWn = to.dWq;
                if (dd0 != 0.0) tn_.seeded = true;
            }
            sd::defect_off_t(Xs, {dXs}, o.Minv, tc.dMinv, Wn, dWn, K, NB, e1, R, std::max(c_.defect, 4), c_.robust, rowsT, ND, dd);
        } else {
            // the data rule (--cond-fit) and the defect correction
            Vec Xs = o.X0, dXs = tc.dX0[0];
            std::vector<Vec> Wn;
            std::vector<std::vector<Vec>> dWn(ND);
            dt::DataIn din = r.din;
            dt::DataOut dout;
            const sd::FixT* fixp = nullptr;
            sd::FixT fix;
            if (o.rk) {
                if (!r.have_din) throw std::runtime_error("mk tangent: residues without the forward's data rule");
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
            Xs_dbg = Xs;
            sd::defect_int_t(Xs, {dXs}, o.Minv, tc.dMinv, Wn, dWn, act, r.Smax, K, NB, e1, R, c_.defect, c_.robust, fixp, rowsT, ND, dd);
        }
        Xk = dd.X;
        dXk = dd.dX[0];
        const Vec Xk_own = Xk;
        Xlast_own = Xk_own;
        // the value SNAPPED to the forward's own iterate (captured): the tangent is taken along the forward's trajectory, bit for bit, so
        // the replay's round-off (amplified up to ~1e7 by the rank-2 step at the integer centre 2) does not accumulate across steps
        if (!r.offint && it < int(r.Xits.size()) && r.Xits[it].size() == Xk.size()) Xk = r.Xits[it];
        else if (r.offint && r.Xg.size() == Xk.size()) Xk = r.Xg;
        if (!r.offint && it < int(r.Xits.size()) && std::getenv("MK_TAN_DEBUG")) {
            auto relv = [](const Vec& a, const Vec& b) { double n = 0, d = 0; for (size_t i = 0; i < a.size() && i < b.size(); i++) { n += std::norm(a[i] - b[i]); d += std::norm(b[i]); } return std::sqrt(n / std::max(d, 1e-300)); };
            std::printf("      [tangent debug g %.6f it %d] operator %.2e, start %.2e, result %.2e (rank %d, %s)\n", g, it,
                        it < int(r.Msits.size()) ? relv(Ms, r.Msits[it]) : -1.0, it < int(r.X0its.size()) ? relv(Xs_dbg, r.X0its[it]) : -1.0, relv(Xk_own, r.Xits[it]), r.rks[it],
                        r.isint || !tn_.have_mref ? (it == 0 && r.from_cache ? "cached" : "built") : "recentred");
        }
    }
    if (r.isint) tn_.cache[kf] = {Ms_last, dMs_last};             // (the forward's: an integer centre's cache holds its LAST operator)
    // the value against the forward's block; the series' tangent updated (the block's tangent in place of the step's columns)
    double num = 0.0, den = 0.0;                                     // (the last iteration's own value, before the snap)
    for (size_t i = 0; i < Xlast_own.size() && i < r.Xg.size(); i++) { num += std::norm(Xlast_own[i] - r.Xg[i]); den += std::norm(r.Xg[i]); }
    const double rel = std::sqrt(num / std::max(den, 1e-300));
    tn_.worst_val = std::max(tn_.worst_val, rel);
    tn_.steps++;
    tn_.dC = with_grade(k_.c, dCs, dXk, g, P_.kofs, e1, R);
    double dn = 0.0;
    for (auto& z : dXk) dn = std::max(dn, std::abs(z));
    std::printf("    [tangent g %.6f%s] %d iteration(s); the value against the forward's block %.2e; |d block| %.3e\n", g, r.offint ? " off" : "",
                nits, rel, dn);
    std::fflush(stdout);
}

inline void Engine::tan_finish(const std::vector<Entry>& C) {
    if (const char* p = std::getenv("MK_SERIES_OUT")) write_entries(p, C);
    if (!tn_.on) return;
    if (tn_.kind == "offl" && !tn_.seeded) throw std::runtime_error("MK_TAN: no step carried the family " + tn_.name);
    if (const char* p = std::getenv("MK_TAN_OUT")) {
        write_entries(std::string(p) + ".C", C);
        write_entries(std::string(p) + ".dC", tn_.dC);
    }
    std::printf("  [tangent] %d steps on duals; the worst replayed value against the forward's blocks %.2e\n", tn_.steps, tn_.worst_val);
}

}  // namespace en
}  // namespace mk

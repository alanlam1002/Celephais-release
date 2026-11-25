// Mellin throat recursion (code round 463; research round 1093, the forward-mode replay, STAGE 2, part 1): the WHOLE FORWARD on duals --
// every centre of every pass the forward took, the series' tangent carried from step to step, so the tangent of the final series along one
// direction is exact up to the operator's (meff's central differences: stage 2 part 3 replaces them).  Covered: integer centres (the
// --cond-fit data rule) and off-integer centres (the off-integer start, mk_data_dual offl_start_t, and the defect loop, defect_off_t), outside
// clusters (a cluster step throws: stage 2 part 2); the operator as the forward chose it -- built at the stripped series, from the cache (an
// earlier pass's, its tangent cached with it), or re-centred from the last integer centre's (its tangent re-centred, linear).
// The decisions are the forward's, captured per step (RpCap: the residue rank per Newton iteration, their number, the cache, the data rule's
// input, the family's label and datum); each step's replayed value is checked against the forward's block.
//   MK_TAN=data:NAME@tT | offl:FAMILY | cyl:KEY    the direction: a datum of the series (set_data's entry, moment 0), an off-lattice family's
//                                        datum, or a cylinder parameter (research round 1095: KEY uJ, P_shapeJ, qJ, QJ, bJ, btJ, c, n, K as
//                                        --opbuild-dtheta / --bundle-at perturb=; the kernel's run2 response d2 beside the series' d1)
// A cylinder direction moves the lapse families' centres (their Sturm-Liouville exponents on the cylinder).  The replay keeps every centre at
// its key and carries the motion as a moment shift there (research round 1095, spec section 49: d(c x^sigma) = dc x^sigma + c dsigma ln x
// x^sigma; for a tangent only the first-order term exists): the forward's step at a key is covariant under the shift (its contour and its
// residues are those of the same s-plane function about either centre), so the fixed-key replay yields the shifted tangent itself; the moving
// centre enters only the finite-difference gate, whose two forwards' series are shifted back to the base keys.
// CORRECTION (research round 1096, spec section 50): the moment shift alone is exact only where the coefficient does not depend on the centre; a
// family's descendant steps (sigma + 1, + 2, ..: -M(sigma)^-1 S, a root inside their contour) do.  Each off-integer step's centre is carried as a
// dual: MK_TAN_DSIG=c0:ds,... (a base centre and its motion; a step at n c0 + k, n = 1..3, moves by n ds): the operator, quadratic about its
// centre, dMs += dg (Ms1, 2 Ms2, 0); the rows projected at the moving key, dR -= dg k R_(k-1); the block entered at its fixed key, dX += dg k X_(k-1)
// (moments of ln^k x / k!).
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
                      std::vector<cd>& dMs, const Kernel::DCyl* dcy = nullptr) {
    Ctx& c = kr.c;
    int K = c.K, nn = P.nn, NB = P.nb * nn;
    if (K < 3) throw std::runtime_error("mk: K = " + std::to_string(K) + " log moments: the step's operator needs K >= 3 (round 1187)");
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
        if (dcy) {                                                   // research round 1095: the cylinder's response at each batch member
            kr.run2(Cp, dCb, *dcy, top);
            auto d2g = project(c, kr.d2out, g, Bt, P.gfam, P.dE, P.W, false);
            for (size_t i = 0; i < dRg.size() && i < d2g.size(); i++) dRg[i] += d2g[i];
        }
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
    } else if (tn_.kind == "cyl") {
        // research round 1095: the cylinder's derivative in KEY on the bundle's theta grid (mk_kernel.cpp --opbuild-dtheta's rule); no series seed
        const std::string& key = tn_.name;
        auto grid = [&](auto f) { std::vector<double> v(S_.TH.size()); for (size_t q = 0; q < v.size(); q++) v[q] = f(S_.TH[q]); return v; };
        Kernel::DCyl& d = tn_.dcy;
        if (key == "n") d.dn = 1.0;
        else if (key == "K") d.dK = 1.0;
        else if (key == "c") { d.b = grid([](double) { return 1.0; }); d.dbnorm = 1.0; }
        else {
            size_t dg = key.find_first_of("0123456789");
            if (dg == std::string::npos) throw std::runtime_error("MK_TAN: cyl:" + key);
            std::string fam = key.substr(0, dg);
            double j = std::stod(key.substr(dg));
            if (fam == "u") d.U = grid([&](double x) { return std::cos(2.0 * j * x); });
            else if (fam == "P_shape") d.P = grid([&](double x) { return std::cos(2.0 * j * x); });
            else if (fam == "q") d.q = grid([&](double x) { double s_ = std::sin(x); return s_ * s_ * std::cos(2.0 * j * x); });
            else if (fam == "Q") d.Qb = grid([&](double x) { double s_ = std::sin(x); return s_ * s_ * s_ * s_ * std::cos((2.0 * j + 1.0) * x); });
            else if (fam == "b") d.b = grid([&](double x) { return std::cos(2.0 * j * x); });
            else if (fam == "bt") d.bt = grid([&](double x) { return std::sin(2.0 * (j + 1.0) * x); });
            else throw std::runtime_error("MK_TAN: cyl:" + key);
        }
        tn_.cyl = true;
        tn_.seeded = true;
    } else if (tn_.kind != "offl")
        throw std::runtime_error("MK_TAN: data:NAME@tT, offl:FAMILY or cyl:KEY");
    if (const char* ds = std::getenv("MK_TAN_DSIG")) {              // research round 1096: the centres' motion along the direction
        std::string t(ds);
        size_t p0 = 0;
        while (p0 < t.size()) {
            size_t p1 = t.find(',', p0);
            if (p1 == std::string::npos) p1 = t.size();
            std::string it = t.substr(p0, p1 - p0);
            size_t c_ = it.find(':');
            if (c_ != std::string::npos) tn_.dsig.emplace_back(std::atof(it.substr(0, c_).c_str()), std::atof(it.substr(c_ + 1).c_str()));
            p0 = p1 + 1;
        }
        std::printf("  [tangent] %d moving centre(s) carried as duals (research round 1096)\n", int(tn_.dsig.size()));
    }
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
    // research round 1096: this step's centre motion dg (0 at integer centres and centres not moving)
    double dg = 0.0;
    if (!r.isint)
        for (auto& cd_ : tn_.dsig) {
            for (int n_ = 1; n_ <= 3 && dg == 0.0; n_++) {
                double k_r = g - n_ * cd_.first;
                if (k_r > -1e-7 && std::abs(k_r - std::nearbyint(k_r)) < 1e-7) dg = n_ * cd_.second;
            }
            if (dg != 0.0) break;
        }
    auto mshift = [&](const Vec& M) {                                // D: moment k <- k x moment k - 1 (layout [k][i])
        Vec D(M.size(), 0.0);
        for (int k = 1; k < K; k++)
            for (int i = 0; i < NB; i++) D[size_t(k) * NB + i] = double(k) * M[size_t(k - 1) * NB + i];
        return D;
    };
    auto nshift = [&](const Vec& Xn, double Rw) { return ct::nodes_of(mshift(ct::moments_of(Xn, K, NB, e1, Rw)), K, NB, e1, Rw); };
    auto fixd = [&](const Vec& X, const Vec& dX, double Rw) {        // the block's tangent at its FIXED key: dX + dg D(X)
        if (dg == 0.0) return dX;
        Vec o = dX, sh = nshift(X, Rw);
        for (size_t i = 0; i < o.size(); i++) o[i] += dg * sh[i];
        return o;
    };
    auto rows_dg = [&](const Vec& v, Vec& dR) {                       // the rows projected at the MOVING key: dR - dg D(R)
        if (dg == 0.0) return;
        Vec sh = mshift(v);
        for (size_t i = 0; i < dR.size() && i < sh.size(); i++) dR[i] -= dg * sh[i];
    };
    if (dg != 0.0) std::printf("    [tangent g %.6f] the centre moves: %+.6e (research round 1096)\n", g, dg);
    // the value by the forward's own evaluation (bit for bit: a step amplifies round-off by up to ~1e10 at the rank-2 integer centre 2 in later
    // passes), the tangent by the kernel's exact linear response
    auto rows_lin = [&](const std::vector<Entry>& Cx, const std::vector<Entry>& dCx, Vec& dR) {
        k_.run(Cx, &dCx, top);
        dR = project(k_.c, k_.dout, g, 1, P_.gfam, P_.dE, P_.W, false);
        if (tn_.cyl) {                                               // research round 1095: + the cylinder's response (run2's d2)
            k_.run2(Cx, dCx, tn_.dcy, top);
            Vec d2 = project(k_.c, k_.d2out, g, 1, P_.gfam, P_.dE, P_.W, false);
            for (size_t i = 0; i < dR.size() && i < d2.size(); i++) dR[i] += d2[i];
        }
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
        if ((dn > 0 || tn_.cyl) && c_.meff_mode == 0 && !OP_FD) {
            meff_dual(k_, P_, Cx, dCx, g, top, c_.meff_h, c_.meff_chunk, dMs, tn_.cyl ? &tn_.dcy : nullptr);
            dn = 0.0;
        } else if (tn_.cyl)
            throw std::runtime_error("mk tangent: a cylinder direction needs the exact operator tangent (meff mode 0, MK_TAN_MEFF unset)");
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
        const Vec dXf = fixd(X, dX[0], R);
        Vec v = rows_lin(with_grade(k_.c, r.Cs, X, g, P_.kofs, e1, R), tan_at(&dXf, R), d);
        rows_dg(v, d);
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
                const Vec dXf = fixd(Xk, dXk, c_.R);
                Ms = meff_t(with_grade(k_.c, r.Cs, Xk, g, P_.kofs, e1, c_.R), tan_at(&dXf, c_.R), dMs);
            }
            tn_.have_mref = true;
            tn_.mref_g = g;
            tn_.mref_s = Ms;
            tn_.dmref_s = dMs;
        } else {
            recentre(tn_.mref_s, Ms, nm, g - tn_.mref_g);
            recentre(tn_.dmref_s, dMs, nm, g - tn_.mref_g);
        }
        if (dg != 0.0)                                               // research round 1096: the operator about the moving centre (quadratic: exact)
            for (size_t a = 0; a < nm; a++) { dMs[a] += dg * Ms[nm + a]; dMs[nm + a] += dg * 2.0 * Ms[2 * nm + a]; }
        Ms_last = Ms;
        dMs_last = dMs;
        // the sources at the nodes and their tangent
        Vec Sn, dSn(size_t(N) * NB);
        if (it == 0) {
            Vec dS, S = rows_lin(r.Cs, dCs, dS);
            rows_dg(S, dS);
            Sn = ct::nodes_of(S, K, NB, e1, R);
            dSn = ct::nodes_of(dS, K, NB, e1, R);
        } else {
            const Vec dXf = fixd(Xk, dXk, R);
            Vec dr, rr = rows_lin(with_grade(k_.c, r.Cs, Xk, g, P_.kofs, e1, R), tan_at(&dXf, R), dr);
            rows_dg(rr, dr);
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
    tn_.dC = with_grade(k_.c, dCs, fixd(Xk, dXk, R), g, P_.kofs, e1, R);
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

// tangents stage 1 (research round 1203): the S2b pin in the replay.  The forward (mk_engine.hpp, MK_S2B, the integer centre whose contour holds
// sigma_2) removes from every column j (key g + kofs_j >= 1) the member p d^k a1_j, a1_j = m1_j / d, at the offset d the stored moments carry,
// d = Re(num / den), num = sum_j sum_{k=1}^{K-2} conj(m_k) m_(k+1), den = sum_j sum_{k=1}^{K-2} |m_k|^2 (MK_S2B_OFFSET=sigma: d = sigma_2 - g, fixed).
// Its exact linearisation along the replay's direction, applied to the series tangent tn_.dC before the forward edits C (C here pre-pin):
//   dd = Re(dnum / den - num dden / den^2)   (0 when d is fixed),   da1_j = dm1_j / d - m1_j dd / d^2,
//   dm_k <- dm_k - (d^k da1_j + k d^(k-1) dd a1_j)   (k = 0 .. K - 1, the forward's loop),
// and PH2's da1 kept (the flow's amplitude in the replay column's stencil).  Nothing decided here: which centre pins and which columns enter
// are the forward's tests, read from C.
inline void Engine::s2b_pin_tan(const std::vector<Entry>& C, double g, double d, bool d_fixed) {
    const int K = k_.c.K, nn = P_.nn;
    std::vector<int> qc(nn, -1), qd(nn, -1);
    for (int j = 0; j < nn; j++) {
        Key t = ckey(k_.c, g + P_.kofs[j]);
        if (!(t.v >= 1)) continue;
        for (size_t q = 0; q < C.size(); q++)
            if (C[q].j == j && keq(C[q].t, t) && C[q].nb == 1) { qc[j] = int(q); break; }
        if (qc[j] < 0) continue;
        for (size_t q = 0; q < tn_.dC.size(); q++)
            if (tn_.dC[q].j == j && keq(tn_.dC[q].t, t)) { qd[j] = int(q); break; }
        if (qd[j] < 0) {                                             // (no tangent there yet: a zero entry, filled below if a1's derivative is not zero)
            tn_.dC.push_back(Entry{j, t, 1, std::vector<cd>(K, 0.0)});
            qd[j] = int(tn_.dC.size()) - 1;
        }
        if (tn_.dC[qd[j]].nb != 1) throw std::runtime_error("s2b_pin_tan: a batched tangent");
        if (int(tn_.dC[qd[j]].m.size()) < K) tn_.dC[qd[j]].m.resize(K, 0.0);
    }
    double dd = 0.0;
    if (!d_fixed) {
        cd num = 0.0, dnum = 0.0;
        double den = 0.0, dden = 0.0;
        for (int j = 0; j < nn; j++) {
            if (qc[j] < 0) continue;
            const auto& m = C[qc[j]].m;
            const auto& dm = tn_.dC[qd[j]].m;
            for (int k = 1; k + 1 < K; k++) {
                num += std::conj(m[k]) * m[k + 1];
                dnum += std::conj(dm[k]) * m[k + 1] + std::conj(m[k]) * dm[k + 1];
                den += std::norm(m[k]);
                dden += 2.0 * (std::conj(m[k]) * dm[k]).real();
            }
        }
        if (den > 0) dd = (dnum / den - num * dden / (den * den)).real();
    }
    for (int j = 0; j < nn; j++) {
        if (qc[j] < 0) continue;
        const auto& m = C[qc[j]].m;
        auto& dm = tn_.dC[qd[j]].m;
        const cd a1 = m[1] / d;
        const cd da1 = dm[1] / d - m[1] * dd / (d * d);
        double pk = 1.0, pk1 = 0.0;                                  // d^k, k d^(k-1)
        for (int k = 0; k < K; k++) {
            dm[k] -= pk * da1 + pk1 * dd * a1;
            pk1 = (k + 1) * pk;
            pk *= d;
        }
        if (S_.collab[j] == "PH2") { tn_.da1 = da1; tn_.pinned = true; }
    }
    std::printf("    [tangent s2b g %.0f] the pin linearised: d %+.9f, dd %+.6e%s; PH2's da1 %+.6e\n", g, d, dd, d_fixed ? " (d fixed)" : "",
                tn_.da1.real());
}

}  // namespace en
}  // namespace mk

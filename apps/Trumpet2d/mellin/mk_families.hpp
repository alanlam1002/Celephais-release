// Mellin throat recursion, M2e (code round 463; research rounds 1018-1025): the FAMILIES and the REGISTRY.
//
// spinning_trumpet/scripts/n2a_nl.py:
//   announce     the centres the rows announce above g (--offl auto): every Galerkin row's family pieces in the rows of the series stripped
//                at g, aligned at key - ROWOFF - dE; a candidate in (g, gmax] with a piece above 1e-15 joins the pending queue (ckey)
//   lapse roots  at aligned 2 in pass 1: the bordered quadratic pencil linearised (scipy.linalg.eig on [[0, I], [-Q0, -Q1]], [[I, 0], [0, Q2]],
//                Q = Ms / rw / cw, the bordering's unit vectors in Q0), its real finite roots; per lapse l (4 .. --offl-lmax) the registry guess
//                1 + (-sqrt2 + sqrt(2 + l (l + 1))) / n refined to the nearest root (dropped beyond 0.1 or above gmax)
//   towers       the families' lattice towers sf + k <= gmax (centres whatever the amplitude)
//   queue        the pending centres: sorted(set(pending), key=float), the smallest unseen next
//   gauge flow   the gauge families as the exact Lie flow of the converged series (RK4 in the flow parameter, --gauge-steps)
// Header-only, compiled in the kernel's translation unit (it is series arithmetic: the kernel's operations, its flags).
#pragma once
#include <cstdlib>
#include <string>
#include "mk_contour.hpp"
#include "mk_operator.hpp"

#include <cmath>
#include <map>
#include <set>

namespace mk {
namespace fm {

constexpr double ANNOUNCE_MIN = 1e-15;   // np.abs(piece_).max() > 1e-15
constexpr double REAL_ROOT = 1e-6;       // abs(e_.imag) < 1e-6
constexpr double ROOT_FAR = 0.1;         // abs(sgm - guess) > 0.1: dropped
constexpr double GUESS_MARGIN = 0.2;     // guess > gmax + 0.2: skipped

// research round 1078 (spinning_trumpet analysis_slicing/SLICING_FAMILIES.md section 3): the lapse families' aligned exponents from the
// Sturm-Liouville problem on the solved cylinder, d/dx( g (1 - x^2) Y' ) + mu g Y = 0 with g = P^2 e^{-2u} (u, P the cylinder's cos 2j theta
// modes), in the even-Legendre basis (30 functions, 400 Gauss-Legendre nodes; n2a_nl.lapse_sl_sigma); sigma_l = 1 + (-n + sqrt(n^2 + mu_{l/2})) / n
inline std::map<int, double> lapse_sl_sigma(const std::vector<double>& u, const std::vector<double>& P, double n, int lmax, int nl = 30, int nq = 400) {
    std::vector<double> xq(nq), wq(nq);
    for (int i = 0; i < nq; i++) {                                 // Gauss-Legendre by Newton on P_nq
        double x = std::cos(M_PI * (i + 0.75) / (nq + 0.5)), dp = 0;
        for (int it = 0; it < 100; it++) {
            double p0 = 1, p1 = x;
            for (int k = 2; k <= nq; k++) { double p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
            dp = nq * (x * p1 - p0) / (x * x - 1);
            double dx = p1 / dp;
            x -= dx;
            if (std::abs(dx) < 1e-16) break;
        }
        xq[i] = x; wq[i] = 2 / ((1 - x * x) * dp * dp);
    }
    const int L = 2 * nl;
    std::vector<double> Pv(size_t(nl) * nq), dP(size_t(nl) * nq), g(nq);
    for (int i = 0; i < nq; i++) {
        double x = xq[i], th = std::acos(x), uu = 0, pp = 0;
        for (size_t j = 0; j < u.size(); j++) uu += u[j] * std::cos(2.0 * j * th);
        for (size_t j = 0; j < P.size(); j++) pp += P[j] * std::cos(2.0 * j * th);
        g[i] = pp * pp * std::exp(-2 * uu);
        std::vector<double> pl(L + 1), dl(L + 1);
        pl[0] = 1; pl[1] = x; dl[0] = 0; dl[1] = 1;
        for (int k = 2; k <= L; k++) { pl[k] = ((2 * k - 1) * x * pl[k - 1] - (k - 1) * pl[k - 2]) / k; dl[k] = dl[k - 2] + (2 * k - 1) * pl[k - 1]; }
        for (int a = 0; a < nl; a++) { Pv[size_t(a) * nq + i] = pl[2 * a]; dP[size_t(a) * nq + i] = dl[2 * a]; }
    }
    std::vector<cd> A(size_t(nl) * nl), B(size_t(nl) * nl);
    for (int a = 0; a < nl; a++)
        for (int b = 0; b < nl; b++) {
            double sa = 0, sb = 0;
            for (int i = 0; i < nq; i++) {
                sa += dP[size_t(a) * nq + i] * dP[size_t(b) * nq + i] * g[i] * (1 - xq[i] * xq[i]) * wq[i];
                sb += Pv[size_t(a) * nq + i] * Pv[size_t(b) * nq + i] * g[i] * wq[i];
            }
            A[size_t(a) * nl + b] = sa; B[size_t(a) * nl + b] = sb;
        }
    std::vector<cd> ev = la::eiggen(A, B, nl);
    std::vector<double> mu;
    for (auto& e : ev) mu.push_back(e.real());
    std::sort(mu.begin(), mu.end());
    std::map<int, double> out;
    for (int l = 2; l <= lmax && l / 2 < nl; l += 2) out[l] = 1 + (-n + std::sqrt(n * n + mu[l / 2])) / n;
    return out;
}

inline double pmaxabs(const Piece& p) {
    double m = 0.0;
    for (auto& z : p.a) m = std::max(m, std::abs(z));
    return m;
}

struct Ann {
    double key, mag;
};
// rows: the kernel's out[] for the series stripped at g; adds the announced keys (ckey: registered) to `add`, in Python's order
// the candidates (aligned centre, the piece's magnitude) in the rows of the series stripped at g, in Python's loop order -- computed from the
// rows at the centre's START (n2a_nl: rr from sources(strip(C, g))); registered (ckey) when the engine does: in every Newton iteration
inline std::vector<std::pair<double, double>> announce_candidates(const Ser rows[6], const Proj& P, double g, double gmax) {
    std::vector<std::pair<double, double>> cand;
    for (int i = 0; i < P.nn; i++) {
        int f = P.gfam[i];
        for (auto& kv : rows[f].d) {
            double sc = (kv.first.v - ROWOFF1[f]) - P.dE[i];
            if (g + 1e-9 < sc && sc <= gmax + 1e-9) cand.emplace_back(sc, pmaxabs(kv.second));
        }
    }
    return cand;
}
inline void announce_register(Ctx& c, const std::vector<std::pair<double, double>>& cand, std::vector<Ann>& ann, std::vector<Key>& add) {
    std::map<double, double> a;
    for (auto& cm : cand) {
        double r = std::nearbyint(cm.first * 1e9) / 1e9;
        a[r] = std::max(a.count(r) ? a[r] : 0.0, cm.second);
        if (cm.second > ANNOUNCE_MIN) add.push_back(c.reg(cm.first));      // n2a_nl: register(sc_)
    }
    for (auto& kv : a) ann.push_back(Ann{kv.first, kv.second});
}
inline void announce(Ctx& c, const Ser rows[6], const Proj& P, double g, double gmax, std::vector<Ann>& ann, std::vector<Key>& add) {
    announce_register(c, announce_candidates(rows, P, g, gmax), ann, add);
}

struct Lapse {
    std::vector<cd> eig_all;
    std::vector<double> evr;
    std::vector<std::pair<int, Key>> lapse;          // (l, the refined root's key)
    std::vector<std::pair<int, Key>> gauge;          // stage (ii) item 1 (c.gauge_reg): (L, the block root nearest L / n)
    std::vector<std::pair<int, double>> registry;    // (l, guess)
};
// Ms: 3 x NB x NB; bord: full (row, column) indices
inline void lapse_roots(Ctx& c, const std::vector<cd>& Ms, int NB, const std::vector<double>& rw, const std::vector<double>& cw,
                        const std::vector<std::pair<int, int>>& bord, double g, double n, double gmax, int lmax, Lapse& o) {
    int kb = int(bord.size()), nb = NB + kb, n2 = 2 * nb;
    size_t nm = size_t(NB) * NB;
    std::vector<std::vector<cd>> Q(3, std::vector<cd>(size_t(nb) * nb, cd(0.0, 0.0)));
    for (int m = 0; m < 3; m++)
        for (int i = 0; i < NB; i++)
            for (int j = 0; j < NB; j++)
                Q[m][size_t(i) * nb + j] = ct::div(ct::div(Ms[m * nm + size_t(i) * NB + j], cd(rw[i], 0.0)), cd(cw[j], 0.0));
    for (int q = 0; q < kb; q++) {
        Q[0][size_t(bord[q].first) * nb + NB + q] = 1.0;
        Q[0][size_t(NB + q) * nb + bord[q].second] = 1.0;
    }
    std::vector<cd> A(size_t(n2) * n2, cd(0.0, 0.0)), B(size_t(n2) * n2, cd(0.0, 0.0));
    for (int i = 0; i < nb; i++) {
        A[size_t(i) * n2 + nb + i] = 1.0;
        B[size_t(i) * n2 + i] = 1.0;
        for (int j = 0; j < nb; j++) {
            A[size_t(nb + i) * n2 + j] = -Q[0][size_t(i) * nb + j];
            A[size_t(nb + i) * n2 + nb + j] = -Q[1][size_t(i) * nb + j];
            B[size_t(nb + i) * n2 + nb + j] = Q[2][size_t(i) * nb + j];
        }
    }
    std::vector<cd> al, be;
    la::eiggen_ab(A, B, n2, al, be);
    o.eig_all.resize(n2);
    bool all_real_alpha = true;
    for (auto& z : al) all_real_alpha = all_real_alpha && z.imag() == 0.0;
    for (int i = 0; i < n2; i++) {                                // scipy's _make_eigvals
        if (be[i] != cd(0.0, 0.0)) o.eig_all[i] = ct::div(al[i], be[i]);
        else if (al[i] != cd(0.0, 0.0)) o.eig_all[i] = cd(std::numeric_limits<double>::infinity(), 0.0);
        else o.eig_all[i] = all_real_alpha ? cd(std::nan(""), 0.0) : cd(std::nan(""), std::nan(""));
    }
    for (auto& e : o.eig_all)
        if (std::isfinite(e.real()) && std::isfinite(e.imag()) && std::abs(e.imag()) < REAL_ROOT) o.evr.push_back(e.real() + g);
    for (int L = 4; L <= lmax; L += 2) {
        double guess = 1 + (-std::sqrt(2.0) + std::sqrt(2.0 + L * (L + 1))) / n;
        if (c.lapse_sl.count(L)) {                                // research round 1078: the Sturm-Liouville guess on the solved cylinder
            guess = c.lapse_sl.at(L);
            if (!o.evr.empty()) {
                size_t im = 0;
                for (size_t i = 1; i < o.evr.size(); i++)
                    if (std::abs(o.evr[i] - guess) < std::abs(o.evr[im] - guess)) im = i;
                std::printf("    [g 2] lapse l%d: Sturm-Liouville %.9f, six-field root %.9f (%+.1e)\n", L, guess, o.evr[im], o.evr[im] - guess);
            }
        }
        o.registry.emplace_back(L, guess);
        if (guess > gmax + GUESS_MARGIN || o.evr.empty()) continue;
        size_t im = 0;
        for (size_t i = 1; i < o.evr.size(); i++)
            if (std::abs(o.evr[i] - guess) < std::abs(o.evr[im] - guess)) im = i;
        double sgm = o.evr[im];
        if (c.lapse_sl.count(L)) sgm = guess;                       // research round 1088: registered AT the SL exponent (the block root logged)
        if (std::abs(sgm - guess) > ROOT_FAR || sgm > gmax) continue;
        if (const char* ff = std::getenv("MK_FAM_FORCE")) {       // diagnostic (round 1028): a family's centre forced ('L:value;...')
            std::string fs = std::string(";") + ff + ";", k = ";" + std::to_string(L) + ":";
            size_t at = fs.find(k);
            if (at != std::string::npos) sgm = std::stod(fs.substr(at + k.size()));
        }
        o.lapse.emplace_back(L, c.gauge_reg ? c.reg_exact(sgm) : c.reg(sgm));   // n2a_nl: (register_exact if _GAUGE_REG else register)(sgm)
    }
    if (c.gauge_reg) {
        // spec section 26: the gauge families at L / n (s = L in r units, aligned), each refined to the nearest root of the block operator;
        // registered exactly, the datum pinned to zero (n2a_nl: range(2, 2 ceil(gmax n / 2) + 2, 2), within 0.01 of L / n, <= gmax)
        int Lmax = 2 * int(std::ceil(gmax * n / 2));
        for (int L = 2; L <= Lmax; L += 2) {
            double guess = L / n;
            if (guess <= 2 + 1e-9 || guess > gmax + GUESS_MARGIN || o.evr.empty()) continue;
            size_t im = 0;
            for (size_t i = 1; i < o.evr.size(); i++)
                if (std::abs(o.evr[i] - guess) < std::abs(o.evr[im] - guess)) im = i;
            double sgm = o.evr[im];
            if (std::abs(sgm - guess) > 0.01 || sgm > gmax) continue;
            o.gauge.emplace_back(L, c.reg_exact(sgm));
        }
    }
}

// the families' lattice towers: [ckey(sf + k) for sf in fams for k in 0 .. int(gmax) if sf + k <= gmax + 1e-9]
// (stage (ii) item 1: exact -- register_exact, the gauge families' towers and, with c.gauge_reg, the lapse families')
inline std::vector<Key> towers(Ctx& c, const std::vector<Key>& fams, double gmax, bool exact = false) {
    std::vector<Key> t;
    for (auto& sf : fams)
        for (int k = 0; k <= int(gmax); k++)
            if (sf.v + k <= gmax + 1e-9) t.push_back(exact ? c.reg_exact(sf.v + k) : c.reg(sf.v + k));   // n2a_nl: register(sf + k)
    return t;
}

// the pending queue: Python's list, deduplicated and sorted by value at every pop
struct Queue {
    std::vector<Key> pending;
    std::vector<double> seen;
    void push(const std::vector<Key>& ks) { pending.insert(pending.end(), ks.begin(), ks.end()); }
    bool pop(Key& g) {
        while (!pending.empty()) {
            std::vector<Key> u;
            for (auto& k : pending) {                              // set(): equal values are one element
                bool dup = false;
                for (auto& x : u)
                    if (keq(x, k)) { dup = true; break; }
                if (!dup) u.push_back(k);
            }
            std::stable_sort(u.begin(), u.end(), [](const Key& a, const Key& b) { return a.v < b.v; });
            g = u.front();
            u.erase(u.begin());
            pending = u;
            bool s = false;
            for (double x : seen)
                if (x == g.v) { s = true; break; }
            if (s) continue;
            if (!manual) seen.push_back(g.v);
            return true;
        }
        return false;
    }
    // stage (ii) port, item 2: n2a_nl's own seen-set handling -- a centre is marked when the recursion decides it (stepped, or stored / routed
    // elsewhere), not when popped; a product re-queued behind the step that stores it; the redirected centre taken off the queue
    bool manual = false;
    bool has(double v) const {
        for (double x : seen)
            if (x == v) return true;
        return false;
    }
    void mark(double v) {
        if (!has(v)) seen.push_back(v);
    }
    void remove(double v) {
        std::vector<Key> u;
        for (auto& k : pending)
            if (std::abs(k.v - v) > 1e-12) u.push_back(k);
        pending = u;
    }
    std::vector<double> after() const {                           // sorted({float(x) for x in pending})
        std::set<double> s;
        for (auto& k : pending) s.insert(k.v);
        return std::vector<double>(s.begin(), s.end());
    }
};

// ---- the gauge Lie flow (n2a_nl.lie_xi / lie_L / lie_flow) ----
inline Ser theta_ser(Ctx& c, const std::vector<double>& vals) {   // theta_piece: MS({0: const_piece(vals)}, 0, BIG)
    Piece v = c.zeros(1);
    for (int q = 0; q < c.NF; q++) v.a[q] = vals[q];
    std::vector<std::pair<Key, Piece>> d;
    d.emplace_back(Key{0.0, true}, std::move(v));
    return c.make(std::move(d), true, 0, BIG);
}
struct Xi {
    Ser xr, xr_r, xt;
};
inline Xi lie_xi(Ctx& c, const std::vector<std::pair<int, double>>& cs, const std::vector<double>& TH) {
    Xi x{c.emptyser(), c.emptyser(), c.emptyser()};
    auto one = [&](double key, double cf, const std::vector<double>& f, int e) {
        Piece v = c.zeros(1);
        for (int q = 0; q < c.NF; q++) v.a[q] = ct::mul(cd(cf, 0.0), cd(f[q], 0.0));
        std::vector<std::pair<Key, Piece>> d;
        d.emplace_back(Key{key, false}, std::move(v));
        return c.make(std::move(d), true, e, BIG);
    };
    for (auto& lc : cs) {
        int l = lc.first;
        double key = l / c.n;
        std::vector<double> co(c.NF), si(c.NF);
        for (int q = 0; q < c.NF; q++) { co[q] = std::cos(l * TH[q]); si[q] = std::sin(l * TH[q]); }
        Ser a = one(key, lc.second, co, 1);
        x.xr = c.add(x.xr, a);
        Ser b = one(key, lc.second, co, 0);
        x.xr_r = c.add(x.xr_r, b);
        Ser d = one(key, lc.second, si, 0);
        x.xt = c.add(x.xt, d);
    }
    return x;
}
// F, out: psi2 Phb qf br bt Qb.  Every operation SEQUENCED in Python's left-to-right order (products register centres as they arrive, so the
// order of evaluation is part of the result; C++ leaves the order of function arguments unspecified -- no nested calls with side effects)
inline void lie_L(Ctx& c, const Ser F[6], const Xi& xi, double J, const std::vector<double>& TH, Ser out[6]) {
    std::vector<double> cot(c.NF), s3(c.NF);
    for (int q = 0; q < c.NF; q++) {
        cot[q] = std::cos(TH[q]) / std::sin(TH[q]);
        s3[q] = (6 * J) * std::pow(std::sin(TH[q]), 3.0);
    }
    Ser cotS = theta_ser(c, cot);
    auto dot = [&](const Ser& f) {                     // (xr * f.Drad() + xt * f.Dth()) if f.d else MS({})
        if (f.empty()) return c.emptyser();
        Ser dr = c.drad(f);
        Ser a = c.mul(xi.xr, dr);
        Ser dt = c.dth(f);
        Ser b = c.mul(xi.xt, dt);
        return c.add(a, b);
    };
    Ser xtc = c.mul(xi.xt, cotS);
    Ser w = c.add(xi.xr_r, xtc);
    for (int i = 0; i < 2; i++) {                       // psi2, Phb: dot(F) + F * w
        Ser a = dot(F[i]);
        Ser b = c.mul(F[i], w);
        out[i] = c.add(a, b);
    }
    {                                                   // qf: dot(qf) + xr.Drad() - w
        Ser a = dot(F[2]);
        Ser b = c.drad(xi.xr);
        Ser ab = c.add(a, b);
        out[2] = c.sub(ab, w);
    }
    for (int i = 3; i < 5; i++) {                       // br / bt: dot(F) - (F_br * xi.Drad() + (F_bt * xi.Dth() if F_bt.d else MS({})))
        const Ser& x = i == 3 ? xi.xr : xi.xt;
        Ser a = dot(F[i]);
        Ser xd = c.drad(x);
        Ser p = c.mul(F[3], xd);
        Ser q;
        if (F[4].empty()) q = c.emptyser();
        else {
            Ser xt_ = c.dth(x);
            q = c.mul(F[4], xt_);
        }
        Ser pq = c.add(p, q);
        out[i] = c.sub(a, pq);
    }
    {                                                   // Qb: dot(Qb) + xt * theta_piece(6 J sin^3)
        Ser a = dot(F[5]);
        Ser th = theta_ser(c, s3);
        Ser b = c.mul(xi.xt, th);
        out[5] = c.add(a, b);
    }
}
inline void lie_flow(Ctx& c, Ser F[6], const std::vector<std::pair<int, double>>& cs, double J, const std::vector<double>& TH, int steps) {
    Xi xi = lie_xi(c, cs, TH);
    double h = 1.0 / steps;
    auto addc = [&](const Ser A[6], const Ser B[6], double cf, Ser out[6]) {
        for (int i = 0; i < 6; i++) {
            Ser sb = c.scale(B[i], cd(cf, 0.0));
            out[i] = c.add(A[i], sb);
        }
    };
    for (int s = 0; s < steps; s++) {
        Ser k1[6], k2[6], k3[6], k4[6], T[6];
        lie_L(c, F, xi, J, TH, k1);
        addc(F, k1, h / 2, T);
        lie_L(c, T, xi, J, TH, k2);
        addc(F, k2, h / 2, T);
        lie_L(c, T, xi, J, TH, k3);
        addc(F, k3, h, T);
        lie_L(c, T, xi, J, TH, k4);
        for (int i = 0; i < 6; i++) {                   // F + k1 h/6 + k2 h/3 + k3 h/3 + k4 h/6, left to right
            Ser s1 = c.scale(k1[i], cd(h / 6, 0.0));
            Ser f = c.add(F[i], s1);
            Ser s2 = c.scale(k2[i], cd(h / 3, 0.0));
            f = c.add(f, s2);
            Ser s3 = c.scale(k3[i], cd(h / 3, 0.0));
            f = c.add(f, s3);
            Ser s4 = c.scale(k4[i], cd(h / 6, 0.0));
            F[i] = c.add(f, s4);
        }
    }
}

}  // namespace fm
}  // namespace mk

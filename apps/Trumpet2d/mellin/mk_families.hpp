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

inline double pmaxabs(const Piece& p) {
    double m = 0.0;
    for (auto& z : p.a) m = std::max(m, std::abs(z));
    return m;
}

struct Ann {
    double key, mag;
};
// rows: the kernel's out[] for the series stripped at g; adds the announced keys (ckey: registered) to `add`, in Python's order
inline void announce(Ctx& c, const Ser rows[6], const Proj& P, double g, double gmax, std::vector<Ann>& ann, std::vector<Key>& add) {
    std::map<double, double> a;
    for (int i = 0; i < P.nn; i++) {
        int f = P.gfam[i];
        for (auto& kv : rows[f].d) {
            double sc = (kv.first.v - ROWOFF1[f]) - P.dE[i];
            if (!(g + 1e-9 < sc && sc <= gmax + 1e-9)) continue;
            double m = pmaxabs(kv.second), r = std::nearbyint(sc * 1e9) / 1e9;
            a[r] = std::max(a.count(r) ? a[r] : 0.0, m);
            if (m > ANNOUNCE_MIN) add.push_back(ckey(c, sc));
        }
    }
    for (auto& kv : a) ann.push_back(Ann{kv.first, kv.second});
}

struct Lapse {
    std::vector<cd> eig_all;
    std::vector<double> evr;
    std::vector<std::pair<int, Key>> lapse;          // (l, the refined root's key)
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
        o.registry.emplace_back(L, guess);
        if (guess > gmax + GUESS_MARGIN || o.evr.empty()) continue;
        size_t im = 0;
        for (size_t i = 1; i < o.evr.size(); i++)
            if (std::abs(o.evr[i] - guess) < std::abs(o.evr[im] - guess)) im = i;
        double sgm = o.evr[im];
        if (std::abs(sgm - guess) > ROOT_FAR || sgm > gmax) continue;
        o.lapse.emplace_back(L, ckey(c, sgm));
    }
}

// the families' lattice towers: [ckey(sf + k) for sf in fams for k in 0 .. int(gmax) if sf + k <= gmax + 1e-9]
inline std::vector<Key> towers(Ctx& c, const std::vector<Key>& fams, double gmax) {
    std::vector<Key> t;
    for (auto& sf : fams)
        for (int k = 0; k <= int(gmax); k++)
            if (sf.v + k <= gmax + 1e-9) t.push_back(ckey(c, sf.v + k));
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
            seen.push_back(g.v);
            return true;
        }
        return false;
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

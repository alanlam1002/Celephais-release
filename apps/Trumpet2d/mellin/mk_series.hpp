// Mellin throat kernel (code round 463 M1, research round 1013, CELEPHAIS_MIGRATION.md M1): the graded Mellin series algebra.
//
// A port of spinning_trumpet/scripts/n2a_nl.py's MS class, operation by operation (the Python engine is the oracle):
//   a series  r^e sum_g x^g [moments (K) x theta grid (NF)],  valid through grade `top`;
//   pieces keyed by their centre g -- an INTEGER (lattice) or a REAL registered centre (off-lattice families), in INSERTION ORDER (Python's dict
//   order: it fixes the pair order of products, hence the order in which new centres are registered);
//   every piece may carry a leading BATCH axis (B copies of the series evaluated together; 1 broadcasts);
//   the canonical-centre registry (a merge radius Rm): a new centre within Rm of an integer or of a registered centre is re-expanded about it by
//   the binomial shift  x^(c + dl) sum_j v_j (ln x)^j / j!  =  x^c sum_m [sum_j C(m, j) dl^(m - j) v_j] (ln x)^m / m!   (moments >= K dropped).
// Nothing here depends on Kadath; production targets are untouched (this directory builds its own shared library).
#pragma once
#include <cstdio>
#include <string>

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mk {

using cd = std::complex<double>;
constexpr double BIG = 1e6;                     // n2a_nl.BIG (validity "unbounded")

struct Key {                                    // a centre: Python int or float (equal values are the SAME key, as 2 == 2.0)
    double v = 0.0;
    bool isint = true;
};
inline bool keq(const Key& a, const Key& b) { return a.v == b.v; }

struct Piece {                                  // (B, K, NF) complex, row-major
    int B = 1;
    std::vector<cd> a;
};

struct Ctx;

struct Ser {
    std::vector<std::pair<Key, Piece>> d;
    bool has_e = false;
    int e = 0;
    double top = BIG;
    bool empty() const { return d.empty(); }
    int find(const Key& g) const {
        for (size_t i = 0; i < d.size(); i++)
            if (keq(d[i].first, g)) return int(i);
        return -1;
    }
    double pmin() const {
        if (d.empty()) return top + 1;
        double m = d[0].first.v;
        for (auto& p : d) m = std::min(m, p.first.v);
        return m;
    }
};

struct Ctx {
    int K = 6, NF = 64;
    double n = std::sqrt(2.0), Rm = 0.0;
    std::vector<double> cent;                   // the registered off-integer centres (sorted)
    std::vector<cd> D;                          // d/dtheta on the grid (NF x NF): Dth(v) = v @ D^T
    std::vector<double> fact;

    void init(int K_, int NF_, double n_, double Rm_, const std::vector<cd>& D_) {
        K = K_; NF = NF_; n = n_; Rm = Rm_; D = D_;
        fact.assign(K, 1.0);
        for (int m = 1; m < K; m++) fact[m] = fact[m - 1] * m;
    }
    size_t psz() const { return size_t(K) * NF; }

    // ---- the registry (n2a_nl.canon) ----
    static double pyround(double g) { return std::nearbyint(g); }       // Python round(): half to even (the default rounding mode)
    double trace_above = 0.0;                                   // diagnostic (MK_TRACE_CANON): new registrations above this value reported
    std::string trace_tag;
    std::pair<Key, double> canon(double g) {
        double gi = pyround(g);
        if (std::abs(g - gi) < 1e-9 || (Rm > 0 && std::abs(g - gi) < Rm)) return {Key{gi, true}, g - gi};
        if (!cent.empty()) {
            double c = cent[0];
            for (double c_ : cent)
                if (std::abs(c_ - g) < std::abs(c - g)) c = c_;           // the first minimiser (Python min over the sorted list)
            if (std::abs(c - g) < std::max(Rm, 1e-9)) return {Key{c, false}, g - c};
        }
        cent.push_back(g);
        std::sort(cent.begin(), cent.end());
        if (trace_above > 0 && g > trace_above) std::fprintf(stderr, "mk: registered %.12f [%s]\n", g, trace_tag.c_str());   // diagnostic
        return {Key{g, false}, 0.0};
    }
    std::vector<double> shiftmat(double dl) const {
        std::vector<double> S(size_t(K) * K, 0.0);
        for (int m = 0; m < K; m++)
            for (int j = 0; j <= m; j++) S[m * K + j] = binom(m, j) * std::pow(dl, m - j);
        return S;
    }
    static double binom(int m, int j) {
        double r = 1.0;
        for (int i = 1; i <= j; i++) r = r * (m - j + i) / i;
        return r;
    }
    Piece shift(const Piece& p, double dl) const {      // S @ v per batch copy
        auto S = shiftmat(dl);
        Piece o{p.B, std::vector<cd>(p.a.size(), 0.0)};
        for (int b = 0; b < p.B; b++)
            for (int m = 0; m < K; m++)
                for (int j = 0; j <= m; j++) {
                    double s = S[m * K + j];
                    if (s == 0.0) continue;
                    const cd* src = &p.a[(size_t(b) * K + j) * NF];
                    cd* dst = &o.a[(size_t(b) * K + m) * NF];
                    for (int q = 0; q < NF; q++) dst[q] += s * src[q];
                }
        return o;
    }

    // ---- pieces ----
    Piece zeros(int B) const { return Piece{B, std::vector<cd>(size_t(B) * psz(), 0.0)}; }
    static int bb(int Ba, int Bb) {
        if (Ba == Bb || Bb == 1) return Ba;
        if (Ba == 1) return Bb;
        throw std::runtime_error("mk: batch shapes do not broadcast");
    }
    Piece padd(const Piece& x, const Piece& y) const {
        int B = bb(x.B, y.B);
        Piece o = zeros(B);
        size_t P = psz();
        for (int b = 0; b < B; b++) {
            const cd* xa = &x.a[(x.B == 1 ? 0 : b) * P];
            const cd* ya = &y.a[(y.B == 1 ? 0 : b) * P];
            cd* oa = &o.a[b * P];
            for (size_t i = 0; i < P; i++) oa[i] = xa[i] + ya[i];
        }
        return o;
    }
    void padd_inplace(Piece& x, const Piece& y) const {        // x += y (x broadcast up if needed)
        if (x.B == 1 && y.B > 1) x = padd(zeros(y.B), x);
        size_t P = psz();
        for (int b = 0; b < x.B; b++) {
            const cd* ya = &y.a[(y.B == 1 ? 0 : b) * P];
            cd* xa = &x.a[b * P];
            for (size_t i = 0; i < P; i++) xa[i] += ya[i];
        }
    }
    static bool anynz(const Piece& p) {
        for (auto& z : p.a)
            if (z != 0.0) return true;
        return false;
    }
    Piece scaled(const Piece& x) const {                       // moments / m!
        Piece o = x;
        for (int b = 0; b < x.B; b++)
            for (int k = 0; k < K; k++)
                for (int q = 0; q < NF; q++) o.a[(size_t(b) * K + k) * NF + q] /= fact[k];
        return o;
    }
    std::vector<bool> nzmom(const Piece& x) const {           // moment j nonzero anywhere in the batch (n2a_nl.mprod's test)
        std::vector<bool> nz(K, false);
        for (int b = 0; b < x.B; b++)
            for (int j = 0; j < K; j++)
                if (!nz[j])
                    for (int q = 0; q < NF; q++)
                        if (x.a[(size_t(b) * K + j) * NF + q] != 0.0) { nz[j] = true; break; }
        return nz;
    }
    // mprod on pre-scaled factors into a reused buffer (the same arithmetic as mprod)
    void mprod_into(Piece& o, const Piece& fa, const Piece& fb, const std::vector<bool>& nz) const {
        int B = bb(fa.B, fb.B);
        size_t P = psz();
        if (o.B != B || o.a.size() != size_t(B) * P) o = zeros(B);
        else std::fill(o.a.begin(), o.a.end(), cd(0.0));
        for (int b = 0; b < B; b++) {
            const cd* xa = &fa.a[(fa.B == 1 ? 0 : b) * P];
            const cd* ya = &fb.a[(fb.B == 1 ? 0 : b) * P];
            cd* w = &o.a[b * P];
            for (int j = 0; j < K; j++) {
                if (!nz[j]) continue;
                for (int m = j; m < K; m++) {
                    const cd* x = xa + j * NF;
                    const cd* y = ya + (m - j) * NF;
                    cd* wm = w + m * NF;
                    for (int q = 0; q < NF; q++) wm[q] += x[q] * y[q];
                }
            }
            for (int k = 0; k < K; k++)
                for (int q = 0; q < NF; q++) w[k * NF + q] *= fact[k];
        }
    }
    // the moment convolution gamma_m = sum_j C(m, j) alpha_j beta_(m - j) (n2a_nl.mprod: through the 1/m!-scaled moments)
    Piece mprod(const Piece& x, const Piece& y) const {
        int B = bb(x.B, y.B);
        size_t P = psz();
        std::vector<bool> nz(K, false);                         // nonzero moment j of x anywhere in its batch (the Python test)
        for (int b = 0; b < x.B; b++)
            for (int j = 0; j < K; j++)
                if (!nz[j])
                    for (int q = 0; q < NF; q++)
                        if (x.a[(size_t(b) * K + j) * NF + q] != 0.0) { nz[j] = true; break; }
        Piece o = zeros(B);
        std::vector<cd> fa(P), fb(P);
        for (int b = 0; b < B; b++) {
            const cd* xa = &x.a[(x.B == 1 ? 0 : b) * P];
            const cd* ya = &y.a[(y.B == 1 ? 0 : b) * P];
            for (int k = 0; k < K; k++)
                for (int q = 0; q < NF; q++) {
                    fa[k * NF + q] = xa[k * NF + q] / fact[k];
                    fb[k * NF + q] = ya[k * NF + q] / fact[k];
                }
            cd* w = &o.a[b * P];
            for (int j = 0; j < K; j++) {
                if (!nz[j]) continue;
                for (int m = j; m < K; m++)
                    for (int q = 0; q < NF; q++) w[m * NF + q] += fa[j * NF + q] * fb[(m - j) * NF + q];
            }
            for (int k = 0; k < K; k++)
                for (int q = 0; q < NF; q++) w[k * NF + q] *= fact[k];
        }
        return o;
    }

    // ---- MS(d, e, top): canonical keys (when the registry is active), validity filter ----
    Ser make(std::vector<std::pair<Key, Piece>>&& d, bool has_e, int e, double top) {
        Ser s;
        s.top = top;
        if (!cent.empty() || Rm > 0) {                          // n2a_nl.canon_dict
            std::vector<std::pair<Key, Piece>> out;
            for (auto& kv : d) {
                Key c;
                double dl = 0.0;
                if (kv.first.isint) c = Key{kv.first.v, true};
                else { auto r = canon(kv.first.v); c = r.first; dl = r.second; }
                Piece w = std::abs(dl) < 1e-14 ? std::move(kv.second) : shift(kv.second, dl);
                bool found = false;
                for (auto& o : out)
                    if (keq(o.first, c)) { o.second = padd(o.second, w); found = true; break; }
                if (!found) out.emplace_back(c, std::move(w));
            }
            d = std::move(out);
        }
        for (auto& kv : d)
            if (kv.first.v <= top + 1e-9) s.d.push_back(std::move(kv));
        s.has_e = !s.d.empty() && has_e;
        s.e = s.has_e ? e : 0;
        return s;
    }
    Ser emptyser(double top = BIG) { return make({}, false, 0, top); }

    // ---- operations ----
    Ser add(const Ser& a, const Ser& b) {
        if (!a.empty() && !b.empty() && a.e != b.e) throw std::runtime_error("mk: r-power mismatch in add");
        double top = std::min(a.top, b.top);
        std::vector<std::pair<Key, Piece>> d;
        for (auto& kv : a.d)
            if (kv.first.v <= top) d.push_back(kv);
        for (auto& kv : b.d) {
            if (kv.first.v > top) continue;
            bool found = false;
            for (auto& o : d)
                if (keq(o.first, kv.first)) { o.second = padd(o.second, kv.second); found = true; break; }
            if (!found) d.push_back(kv);
        }
        bool he = !a.empty() ? a.has_e : b.has_e;
        int e = !a.empty() ? a.e : b.e;
        return make(std::move(d), he, e, top);
    }
    Ser scale(const Ser& a, cd c) {
        std::vector<std::pair<Key, Piece>> d;
        for (auto& kv : a.d) {
            Piece p = kv.second;
            for (auto& z : p.a) z = c * z;
            d.emplace_back(kv.first, std::move(p));
        }
        return make(std::move(d), a.has_e, a.e, a.top);
    }
    Ser sub(const Ser& a, const Ser& b) { return add(a, scale(b, -1.0)); }
    Ser mul(const Ser& a, const Ser& b) {
        double top = std::min({a.pmin() + b.top, b.pmin() + a.top, BIG});
        if (a.empty() || b.empty()) return make({}, false, 0, top);
        std::vector<std::pair<Key, Piece>> d;
        std::unordered_map<double, int> at;                     // target centre -> position in d
        std::vector<Piece> fa(a.d.size()), fb(b.d.size());     // the 1/m!-scaled moments, once per piece
        std::vector<std::vector<bool>> nza(a.d.size());
        for (size_t i = 0; i < a.d.size(); i++) { fa[i] = scaled(a.d[i].second); nza[i] = nzmom(a.d[i].second); }
        for (size_t i = 0; i < b.d.size(); i++) fb[i] = scaled(b.d[i].second);
        Piece W;
        for (size_t ia = 0; ia < a.d.size(); ia++)              // the original pair order (the registry fills identically)
            for (size_t ib = 0; ib < b.d.size(); ib++) {
                const Key& kA = a.d[ia].first;
                const Key& kB = b.d[ib].first;
                double gv = kA.v + kB.v;
                if (gv > top + 1e-9) continue;
                Key g{gv, kA.isint && kB.isint};
                double dl = 0.0;
                if (!g.isint) { auto r = canon(gv); g = r.first; dl = r.second; }
                mprod_into(W, fa[ia], fb[ib], nza[ia]);
                if (std::abs(dl) >= 1e-14) W = shift(W, dl);
                auto it = at.find(g.v);
                if (it != at.end()) padd_inplace(d[it->second].second, W);
                else { at.emplace(g.v, int(d.size())); d.emplace_back(g, W); }
            }
        std::vector<std::pair<Key, Piece>> dz;                  // exact zeros pruned
        for (auto& kv : d)
            if (anynz(kv.second)) dz.push_back(std::move(kv));
        return make(std::move(dz), true, a.e + b.e, top);
    }
    Ser drad(const Ser& a) {
        if (a.empty()) return make({}, false, 0, a.top);
        std::vector<std::pair<Key, Piece>> d;
        for (auto& kv : a.d) {
            const Piece& v = kv.second;
            Piece w = zeros(v.B);
            double c0 = a.e + n * kv.first.v;
            for (int b = 0; b < v.B; b++)
                for (int k = 0; k < K; k++)
                    for (int q = 0; q < NF; q++) {
                        size_t i = (size_t(b) * K + k) * NF + q;
                        w.a[i] = c0 * v.a[i];
                        if (k < K - 1) w.a[i] += n * v.a[i + NF];
                    }
            d.emplace_back(kv.first, std::move(w));
        }
        return make(std::move(d), true, a.e - 1, a.top);
    }
    Ser dth(const Ser& a) {
        std::vector<std::pair<Key, Piece>> d;
        for (auto& kv : a.d) {
            const Piece& v = kv.second;
            Piece w = zeros(v.B);
            for (int bk = 0; bk < v.B * K; bk++) {
                const cd* src = &v.a[size_t(bk) * NF];
                cd* dst = &w.a[size_t(bk) * NF];
                for (int i = 0; i < NF; i++) {
                    cd s = 0.0;
                    for (int j = 0; j < NF; j++) s += src[j] * D[size_t(i) * NF + j];
                    dst[i] = s;
                }
            }
            d.emplace_back(kv.first, std::move(w));
        }
        return make(std::move(d), a.has_e, a.e, a.top);
    }
    Ser one() {
        Piece v = zeros(1);
        for (int q = 0; q < NF; q++) v.a[q] = 1.0;
        std::vector<std::pair<Key, Piece>> d;
        d.emplace_back(Key{0.0, true}, std::move(v));
        return make(std::move(d), true, 0, BIG);
    }
    Ser const_ser(int e, const std::vector<cd>& vals, double top = BIG) {
        Piece v = zeros(1);
        for (int q = 0; q < NF; q++) v.a[q] = vals[q];
        std::vector<std::pair<Key, Piece>> d;
        d.emplace_back(Key{0.0, true}, std::move(v));
        return make(std::move(d), true, e, top);
    }
    static Key ksub(const Key& g, const Key& p) { return Key{g.v - p.v, g.isint && p.isint}; }

    // a piece divided by a00 (B', NF) per batch copy and grid point
    Piece pdiv(const Piece& v, const Piece& a00) const {   // a00 stored as (B, 1, NF) in a Piece with K = 1 layout: a[b * NF + q]
        int B = bb(v.B, a00.B);
        Piece o = zeros(B);
        for (int b = 0; b < B; b++)
            for (int k = 0; k < K; k++)
                for (int q = 0; q < NF; q++)
                    o.a[(size_t(b) * K + k) * NF + q] = v.a[((v.B == 1 ? 0 : b) * size_t(K) + k) * NF + q] / a00.a[(a00.B == 1 ? 0 : b) * size_t(NF) + q];
        return o;
    }
    // lead: pm, a00 (moment 0 of the leading piece), X = a / (a00 x^pm) - 1 (the leading piece's log moments into X's grade 0)
    void lead(const Ser& a, Key& pm, Piece& a00, Ser& X) {
        int ip = 0;
        for (size_t i = 1; i < a.d.size(); i++)
            if (a.d[i].first.v < a.d[ip].first.v) ip = int(i);
        pm = a.d[ip].first;
        const Piece& L = a.d[ip].second;
        a00 = Piece{L.B, std::vector<cd>(size_t(L.B) * NF)};
        for (int b = 0; b < L.B; b++)
            for (int q = 0; q < NF; q++) a00.a[size_t(b) * NF + q] = L.a[size_t(b) * K * NF + q];
        std::vector<std::pair<Key, Piece>> d;
        for (auto& kv : a.d)
            if (kv.first.v > pm.v) d.emplace_back(ksub(kv.first, pm), pdiv(kv.second, a00));
        Piece lp = L;
        for (int b = 0; b < L.B; b++)
            for (int q = 0; q < NF; q++) lp.a[size_t(b) * K * NF + q] = 0.0;
        if (anynz(lp)) d.emplace_back(Key{0.0, true}, pdiv(lp, a00));
        X = make(std::move(d), true, 0, a.top - pm.v);
    }
    Ser inv(const Ser& a) {
        Key pm; Piece a00; Ser X;
        lead(a, pm, a00, X);
        Ser out = one(), term = one();
        int it = 0;
        for (; it < 400; it++) {
            term = scale(mul(term, X), -1.0);
            if (term.empty() || term.pmin() > X.top) break;
            out = add(out, term);
        }
        if (it == 400) throw std::runtime_error("mk: inv did not terminate");
        std::vector<std::pair<Key, Piece>> d;
        for (auto& kv : out.d) d.emplace_back(ksub(kv.first, pm), pdiv(kv.second, a00));
        return make(std::move(d), true, -a.e, out.top - pm.v);
    }
    Ser expser(const Ser& a) {
        if (a.empty()) {
            std::vector<cd> ones(NF, 1.0);
            return const_ser(0, ones, a.top);
        }
        if (a.e != 0 || a.pmin() < 0) throw std::runtime_error("mk: expser of a series with r-power or negative grades");
        std::vector<std::pair<Key, Piece>> rd;
        for (auto& kv : a.d)
            if (kv.first.v > 0) rd.push_back(kv);
        Piece e0;
        int i0 = a.find(Key{0.0, true});
        if (i0 >= 0) {
            const Piece& a0 = a.d[i0].second;
            e0 = Piece{a0.B, std::vector<cd>(size_t(a0.B) * NF)};
            for (int b = 0; b < a0.B; b++)
                for (int q = 0; q < NF; q++) e0.a[size_t(b) * NF + q] = std::exp(a0.a[size_t(b) * K * NF + q]);
            Piece lp = a0;
            for (int b = 0; b < a0.B; b++)
                for (int q = 0; q < NF; q++) lp.a[size_t(b) * K * NF + q] = 0.0;
            if (anynz(lp)) rd.emplace_back(Key{0.0, true}, std::move(lp));
        } else {
            e0 = Piece{1, std::vector<cd>(NF, 1.0)};
        }
        Ser rest = make(std::move(rd), true, 0, a.top);
        Ser out = one(), term = one();
        int k = 1;
        for (; k < 400; k++) {
            term = scale(mul(term, rest), 1.0 / k);
            if (term.empty() || term.pmin() > rest.top) break;
            out = add(out, term);
        }
        if (k == 400) throw std::runtime_error("mk: expser did not terminate");
        std::vector<std::pair<Key, Piece>> d;
        for (auto& kv : out.d) {
            int B = bb(kv.second.B, e0.B);
            Piece o = zeros(B);
            for (int b = 0; b < B; b++)
                for (int kk = 0; kk < K; kk++)
                    for (int q = 0; q < NF; q++)
                        o.a[(size_t(b) * K + kk) * NF + q] = kv.second.a[((kv.second.B == 1 ? 0 : b) * size_t(K) + kk) * NF + q] *
                                                            e0.a[(e0.B == 1 ? 0 : b) * size_t(NF) + q];
            d.emplace_back(kv.first, std::move(o));
        }
        return make(std::move(d), true, 0, std::min(out.top, a.top));
    }
    Ser logser(const Ser& a) {
        Key pm; Piece a00; Ser X;
        lead(a, pm, a00, X);
        Piece v0 = zeros(a00.B);
        for (int b = 0; b < a00.B; b++)
            for (int q = 0; q < NF; q++) {
                v0.a[size_t(b) * K * NF + q] = std::log(a00.a[size_t(b) * NF + q]);
                if (K > 1) v0.a[(size_t(b) * K + 1) * NF + q] = a.e / n + pm.v;
            }
        std::vector<std::pair<Key, Piece>> d0;
        d0.emplace_back(Key{0.0, true}, std::move(v0));
        Ser out = make(std::move(d0), true, 0, BIG);
        Ser term = one();
        int k = 1;
        for (; k < 400; k++) {
            term = mul(term, X);
            if (term.empty() || term.pmin() > X.top) break;
            out = add(out, scale(term, (k % 2 ? 1.0 : -1.0) / k));
        }
        if (k == 400) throw std::runtime_error("mk: logser did not terminate");
        std::vector<std::pair<Key, Piece>> d = out.d;
        return make(std::move(d), true, 0, std::min(out.top, a.top - pm.v));
    }
    // a series truncated to a validity (the linear response is valid where its value is)
    Ser trunc(const Ser& a, double top) {
        std::vector<std::pair<Key, Piece>> d = a.d;
        return make(std::move(d), a.has_e, a.e, std::min(a.top, top));
    }
};

}  // namespace mk

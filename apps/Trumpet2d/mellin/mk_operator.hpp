// Mellin throat recursion, M2a (code round 463; research rounds 1018-1020): the OPERATOR at a centre.
//
// spinning_trumpet/scripts/n2a_nl.py's meff: the rows' linearisation at aligned g about the current series Cx, a quadratic in (sigma - g),
// M(sigma) = M0 + (sigma - g) M1 + (sigma - g)^2 M2, for the square Galerkin rows (Ms, NB x NB) and the condition rows (Mc, nc x NB).  A
// moment-2 unit perturbation of column j at its aligned centre returns moments (2, 1, 0) = (M, M', M''/2) e_j.  Two modes (research round
// 1019): FD -- central differences in h (n2a_nl's default h 1e-2), the chunk's columns perturbed +-h in ONE batched assembly (--meff-chunk);
// JVP -- the rows' exact linear response (M1's forward mode), the chunk's directions batched.  The projections are n2a_nl's galerkin_src /
// stored_src, with the setup bundle's Gauss tables contracted once per row (W_i = P.I^T P.T[i]).
#pragma once
#include "mk_kernel.hpp"
#include "mk_contour.hpp"

namespace mk {

static const int ROWOFF1[6] = {0, 0, 0, 1, 0, 0};    // n2a_nl.ROWOFF[fam][1], in the kernel's row order H_c E_sigma L_K E_sh^r E_sh^th E_Q

struct Proj {
    int nn = 0, nc = 0, nb = 1;
    std::vector<int> gfam, dE;                  // Galerkin row i: its row family, its leading level (O['dE'])
    std::vector<cd> W;                          // nn x NF
    std::vector<int> cfam, cdE;                 // condition row q (Oc['dE'], >= 99: absent)
    std::vector<cd> Wc;                         // nc x NF
    std::vector<int> kofs;                      // column j's offset kE (nb = 1, no --stagger)
    std::vector<int> gtfam, gtdE;               // M2c: the gate rows (glab, Og['dE']), their weights, live mask (the truncation measure)
    std::vector<cd> Wg;                         // ng x NF
    std::vector<char> live;
};

// n2a_nl.ckey: an int within 1e-9 of an integer, else the registered (canonical) centre -- canon may register it
inline Key ckey(Ctx& c, double x) {
    double xi = std::nearbyint(x);
    if (std::abs(x - xi) < 1e-9) return Key{xi, true};
    return c.canon(x).first;
}

// the piece of a row family at grade Gr (n2a_nl.piece_at): nullptr when absent (zeros); beyond the family's validity an error.  Exact merge
// (stage (i)): the pieces within Rm of Gr and nearer to it than to any integer or registered centre, each re-expanded once from its exact
// exponent, summed in the series' order (into buf)
inline const Piece* piece_at(Ctx& c, const Ser& s, double Gr, Piece& buf) {
    if (Gr > s.top) throw std::runtime_error("mk: grade beyond validity");
    if (!c.exact) {
        Key k = ckey(c, Gr);
        int i = s.find(k);
        return i < 0 ? nullptr : &s.d[i].second;
    }
    double rm = std::max(c.Rm, 1e-9);
    bool any = false;
    for (auto& kv : s.d) {
        double e = kv.first.v, de = e - Gr;
        if (std::abs(de) >= rm) continue;
        bool near = false;
        for (double c_ : c.cent)
            if (std::abs(c_ - e) < std::abs(de) - 1e-12) { near = true; break; }
        double gi = Ctx::pyround(e);
        if (std::abs(e - gi) < std::abs(de) - 1e-12) near = true;
        if (near) continue;
        if (!any) { buf = std::abs(de) < 1e-14 ? kv.second : c.shift(kv.second, de); any = true; }
        else buf = c.padd(buf, std::abs(de) < 1e-14 ? kv.second : c.shift(kv.second, de));
    }
    return any ? &buf : nullptr;
}

// stage (ii) port, item 2 (n2a_nl.piece_at's cluster branch, research round 1032 / spec section 28): at a cluster's grade the content at its
// MEMBERS' exact exponents -- each piece whose nearest centre is a (shifted) member, or an exponent merged into a pole, re-expanded about its
// nearest member, the blocks STACKED (members x K); an absent block zeros.  Returns false when no piece is read (all zeros)
inline bool piece_at_cl(Ctx& c, const Ser& s, double Gr, std::vector<Piece>& blocks) {
    if (Gr > s.top) throw std::runtime_error("mk: grade beyond validity");
    double rm = std::max(c.Rm, 1e-9), key;
    std::vector<double> mem;
    c.clus_at(Gr, key, mem);
    std::vector<double> T{Gr};
    T.insert(T.end(), mem.begin(), mem.end());
    auto ab = c.clus_abs_at(Gr);
    T.insert(T.end(), ab.begin(), ab.end());
    auto inT = [&](double x) {
        for (double t : T)
            if (std::abs(x - t) < 1e-9) return true;
        return false;
    };
    blocks.assign(mem.size(), Piece{0, {}});
    bool any = false;
    for (auto& kv : s.d) {
        double e = kv.first.v, dm = 1e300;
        for (double t : T) dm = std::min(dm, std::abs(e - t));
        if (dm >= rm) continue;
        bool skip = false;
        for (double c_ : c.cent)
            if (std::abs(c_ - e) < dm - 1e-12 && !inT(c_)) { skip = true; break; }
        if (skip) continue;
        double gi = Ctx::pyround(e);
        if (std::abs(e - gi) < dm - 1e-12 && !inT(gi)) continue;
        size_t im = 0;
        for (size_t i = 1; i < mem.size(); i++)
            if (std::abs(e - mem[i]) < std::abs(e - mem[im])) im = i;
        double de = e - mem[im];
        Piece w = std::abs(de) < 1e-14 ? kv.second : c.shift(kv.second, de);
        if (blocks[im].B == 0) blocks[im] = w;
        else blocks[im] = c.padd(blocks[im], w);
        any = true;
    }
    return any;
}

// rows projected at aligned g: (B, K, nrows), rows' weights W (nrows x NF); the family / level per row; B the batch of the result.  Stage
// (ii): at a cluster's grade (B, L K, nrows), the members' blocks stacked (n2a_nl's stacked readout)
inline std::vector<cd> project(Ctx& c, const Ser rows[6], double g, int B, const std::vector<int>& fam, const std::vector<int>& lev,
                               const std::vector<cd>& W, bool absent99) {
    int nr = int(fam.size()), K = c.K, NF = c.NF;
    if (c.is_clus(g)) {
        int L = c.clus_L(g), LK = L * K;
        std::vector<cd> out(size_t(B) * LK * nr, 0.0);
        for (int i = 0; i < nr; i++) {
            if (absent99 && lev[i] >= 99) continue;
            std::vector<Piece> bl;
            if (!piece_at_cl(c, rows[fam[i]], ROWOFF1[fam[i]] + lev[i] + g, bl)) continue;
            const cd* w = &W[size_t(i) * NF];
            for (int l = 0; l < L; l++) {
                if (bl[l].B == 0) continue;
                for (int b = 0; b < B; b++)
                    for (int k = 0; k < K; k++) {
                        const cd* v = &bl[l].a[((bl[l].B == 1 ? 0 : b) * size_t(K) + k) * NF];
                        cd s = 0.0;
                        for (int q = 0; q < NF; q++) s += v[q] * w[q];
                        out[(size_t(b) * LK + l * K + k) * nr + i] = s;
                    }
            }
        }
        return out;
    }
    std::vector<cd> out(size_t(B) * K * nr, 0.0);
    for (int i = 0; i < nr; i++) {
        if (absent99 && lev[i] >= 99) continue;
        Piece buf_;
        const Piece* p = piece_at(c, rows[fam[i]], ROWOFF1[fam[i]] + lev[i] + g, buf_);
        if (!p) continue;
        const cd* w = &W[size_t(i) * NF];
        for (int b = 0; b < B; b++)
            for (int k = 0; k < K; k++) {
                const cd* v = &p->a[((p->B == 1 ? 0 : b) * size_t(K) + k) * NF];
                cd s = 0.0;
                for (int q = 0; q < NF; q++) s += v[q] * w[q];
                out[(size_t(b) * K + k) * nr + i] = s;
            }
    }
    return out;
}

// the operator at a centre: Ms (3, NB, NB), Mc (3, nc, NB), row-major [m][row][col]
inline void meff(Kernel& kr, const Proj& P, const std::vector<Entry>& Cx, double g, double top, int mode, double h, int chunk,
                 std::vector<cd>& Ms, std::vector<cd>& Mc) {
    Ctx& c = kr.c;
    int K = c.K, nn = P.nn, nc = P.nc, NB = P.nb * nn;
    if (P.nb != 1) throw std::runtime_error("mk: the block recursion (nb = 2) is not ported");
    Ms.assign(size_t(3) * NB * NB, 0.0);
    Mc.assign(size_t(3) * nc * NB, 0.0);
    std::vector<std::pair<int, Key>> colsb;
    for (int j = 0; j < nn; j++) colsb.emplace_back(j, ckey(c, g + P.kofs[j]));
    auto put = [&](int col, const std::vector<cd>& Rg, const std::vector<cd>& Rc, int B, int ip, int im, double scale) {
        for (int i = 0; i < nn; i++)
            for (int m = 0; m < 3; m++) {                       // M[m] <- moment (2 - m)
                size_t a = (size_t(ip) * K + (2 - m)) * nn + i;
                cd d = im < 0 ? Rg[a] : (Rg[a] - Rg[(size_t(im) * K + (2 - m)) * nn + i]);
                Ms[(size_t(m) * NB + i) * NB + col] = d * scale;
            }
        for (int q = 0; q < nc; q++)
            for (int m = 0; m < 3; m++) {
                size_t a = (size_t(ip) * K + (2 - m)) * nc + q;
                cd d = im < 0 ? Rc[a] : (Rc[a] - Rc[(size_t(im) * K + (2 - m)) * nc + q]);
                Mc[(size_t(m) * nc + q) * NB + col] = d * scale;
            }
        (void)B;
    };
    for (size_t c0 = 0; c0 < colsb.size(); c0 += size_t(chunk)) {
        size_t c1 = std::min(colsb.size(), c0 + size_t(chunk));
        int nq = int(c1 - c0);
        if (mode == 0) {                                        // FD: +-h at moment 2, one batched assembly
            int Bt = 2 * nq;
            std::vector<Entry> Cp = Cx;
            for (int q = 0; q < nq; q++) {
                int j = colsb[c0 + q].first;
                const Key& t = colsb[c0 + q].second;
                int at = -1;
                for (size_t e = 0; e < Cp.size(); e++)
                    if (Cp[e].j == j && keq(Cp[e].t, t)) { at = int(e); break; }
                if (at < 0) {                                   // a new centre of column j: appended (Python's setdefault order)
                    Cp.push_back(Entry{j, t, 1, std::vector<cd>(K, 0.0)});
                    at = int(Cp.size()) - 1;
                }
                Entry& en = Cp[at];
                if (en.nb == 1) {                               // broadcast to the batch
                    std::vector<cd> arr(size_t(Bt) * K);
                    for (int b = 0; b < Bt; b++)
                        for (int k = 0; k < K; k++) arr[size_t(b) * K + k] = en.m[k];
                    en.m = std::move(arr);
                    en.nb = Bt;
                }
                en.m[size_t(2 * q) * K + 2] += h;
                en.m[size_t(2 * q + 1) * K + 2] -= h;
            }
            kr.run(Cp, nullptr, top);
            auto Rg = project(c, kr.out, g, Bt, P.gfam, P.dE, P.W, false);
            auto Rc = project(c, kr.out, g, Bt, P.cfam, P.cdE, P.Wc, true);
            for (int q = 0; q < nq; q++) put(colsb[c0 + q].first, Rg, Rc, Bt, 2 * q, 2 * q + 1, 1.0 / (2 * h));
        } else {                                                // JVP: the exact linear response, the chunk's directions batched
            std::vector<Entry> dC;
            for (int q = 0; q < nq; q++) {
                std::vector<cd> arr(size_t(nq) * K, 0.0);
                arr[size_t(q) * K + 2] = 1.0;
                dC.push_back(Entry{colsb[c0 + q].first, colsb[c0 + q].second, nq, std::move(arr)});
            }
            kr.run(Cx, &dC, top);
            auto Rg = project(c, kr.dout, g, nq, P.gfam, P.dE, P.W, false);
            auto Rc = project(c, kr.dout, g, nq, P.cfam, P.cdE, P.Wc, true);
            for (int q = 0; q < nq; q++) put(colsb[c0 + q].first, Rg, Rc, nq, q, -1, 1.0);
        }
    }
}

// n2a_nl.with_grade (nb = 1): the series with the aligned-g columns set to X's moments -- replaced in place when the centre is present,
// else appended (the column's dict order: the kernel reads each column's centres in order, the columns in family order)
inline std::vector<Entry> with_grade(Ctx& c, const std::vector<Entry>& Cx, const std::vector<cd>& X, double g, const std::vector<int>& kofs,
                                     const std::vector<cd>& e1, double R) {
    int nn = int(kofs.size()), K = c.K;
    auto Mo = ct::moments_of(X, K, nn, e1, R);
    std::vector<Entry> Cy = Cx;
    for (int j = 0; j < nn; j++) {
        Key t = ckey(c, g + kofs[j]);
        if (!(t.v >= 1)) continue;
        std::vector<cd> m(K);
        for (int k = 0; k < K; k++) m[k] = Mo[size_t(k) * nn + j];
        bool found = false;
        for (auto& en : Cy)
            if (en.j == j && keq(en.t, t)) { en.nb = 1; en.m = m; found = true; break; }
        if (!found) Cy.push_back(Entry{j, t, 1, std::move(m)});
    }
    return Cy;
}

// an off-integer centre's operator: the last integer centre g0's quadratic re-centred at g (n2a_nl: d = g - g0,
// [M0 + d M1 + d^2 M2, M1 + 2 d M2, M2]) -- exact for a pure background; the defect correction against the full rows follows
inline void recentre(const std::vector<cd>& M0, std::vector<cd>& M, size_t nm, double d) {
    M.assign(3 * nm, 0.0);
    for (size_t i = 0; i < nm; i++) {
        M[i] = M0[i] + d * M0[nm + i] + d * d * M0[2 * nm + i];
        M[nm + i] = M0[nm + i] + 2 * d * M0[2 * nm + i];
        M[2 * nm + i] = M0[2 * nm + i];
    }
}

}  // namespace mk

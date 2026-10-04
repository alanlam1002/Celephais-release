// Round 345 (research round 675): the stage-B throat expansion evaluator (hand-written around the generated
// stageB_gen.hpp).  Mirrors scripts/stageB_expansion.py and scripts/stageB_kadath.py exactly:
//   unknowns u -> state (amplitude / companion coefficients, n; the frozen grade-4/5 completion about the seed)
//   -> per theta node: amplitude functions (value, d/dth, d2/dth2) -> on the Laurent grid r = e^{i phi},
//   x = 0.25 e^{i psi} (32 x 64): the six fields' jets (generated), the companions' first-order jets
//   DJ = dJ/da . L and their commutator CJ (round 320), each equation's value E, partials P, second partials H
//   (generated): y0 = E + P.CJ, y1 = P.DJ -> the r^c0 rho^(c1+g) coefficients (separable DFT) -> Re -> the equation's
//   orthonormal projection (Gauss weights) -> the throat rows / nr.
//   The Jacobian: analytic in the amplitude symbols, companions and n (second-order terms included), chained to the
//   coefficients (angular functions) and to the unknowns (the completion's linear map).
//   The matching: the ansatz at r_m (value and r-derivative, + ln rho_m DJ and its r-derivative), least squares on
//   each field's padded basis, + the off-lattice images; its Jacobian by a complex step.
// Round 346 (research round 676, STAGE_B_SPEC section 8): grades 4-5 are unknowns -- the data's completion sections are
//   empty (nlow = nhigh = 0) and the state map is the identity on the slots -- and an optional ORTH section adds the
//   fixed truncation-orthogonality rows, O (u - u_anchor), appended after the throat rows (NROW = throat rows + NORTH);
//   the anchor is the seed they were built at (the fitted W), carried in the section so a run seeded elsewhere keeps them.
// Data: scripts/stageB_kadath.py export.  Lives in the Celephais clone (apps/Trumpet2d/src) next to the generated stageB_gen.hpp.
#pragma once
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "stageB_gen.hpp"

namespace sbe {
typedef std::complex<double> cplx;

struct KeyInfo { int f, gi, kind, nfull; };            // kind 0 ce, 1 qs, 2 se, 3 co
struct EqInfo { std::string name; int c0, c1, nth, nb; std::vector<double> Q; };
struct RowInfo { int e, g, i, part, twist; double nr; };
struct UnkInfo { char t; int key, ia; double u0; };
struct Compl { int nlow = 0, nhigh = 0; std::vector<int> ltype, lkey, lia, hkey, hia; std::vector<double> Phi, Vlow, corr; };
struct MField { int kind = 0, nm = 0; std::vector<int> modes; std::vector<double> pinv; };
struct MKey { int F, d, k, kept; std::vector<double> off; };

inline int kind_of(const std::string& s)
{
    if (s == "ce") return 0;
    if (s == "qs") return 1;
    if (s == "se") return 2;
    if (s == "co") return 3;
    throw std::runtime_error("stageB: unknown angular kind " + s);
}

// value, d/dth, d2/dth2 of the angular function (stageB_expansion.angular)
inline void angular(int kind, int j, double th, double* o)
{
    if (kind == 0 || kind == 2) {
        const double w = 2.0 * j;
        if (kind == 0) { o[0] = std::cos(w * th); o[1] = -w * std::sin(w * th); o[2] = -w * w * std::cos(w * th); }
        else           { o[0] = std::sin(w * th); o[1] =  w * std::cos(w * th); o[2] = -w * w * std::sin(w * th); }
        return;
    }
    if (kind == 1) {
        double c[3];
        angular(0, j, th, c);
        const double s = std::sin(th) * std::sin(th), s1 = std::sin(2 * th), s2 = 2 * std::cos(2 * th);
        o[0] = s * c[0]; o[1] = s1 * c[0] + s * c[1]; o[2] = s2 * c[0] + 2 * s1 * c[1] + s * c[2];
        return;
    }
    const double w = 2.0 * j + 1.0;
    const double c0 = std::cos(w * th), c1 = -w * std::sin(w * th), c2 = -w * w * std::cos(w * th);
    const double sn = std::sin(th), cs = std::cos(th);
    const double s = sn * sn * sn * sn, s1 = 4 * sn * sn * sn * cs, s2 = 12 * sn * sn * cs * cs - 4 * sn * sn * sn * sn;
    o[0] = s * c0; o[1] = s1 * c0 + s * c1; o[2] = s2 * c0 + 2 * s1 * c1 + s * c2;
}

typedef void (*Jets1)(const cplx*, cplx, cplx, cplx, cplx*, cplx*, cplx*);
typedef void (*Jets2)(const cplx*, cplx, cplx, cplx, cplx*, cplx*);
typedef void (*EqFn)(const cplx*, cplx, double, double, cplx&, cplx*, cplx*);

class Eval {
  public:
    int NT = 0, NR = 0, NX = 0, NTH = 0, NK = 0, NE = 0, NU = 0, NROW = 0, NMK = 0, NOFF = 0, NTHM = 0;
    int NTHR = 0, NORTH = 0;                              // throat rows, orthogonality rows (NROW = NTHR + NORTH)
    std::vector<double> orth, orthu0;                     // NORTH x NU, row-major; the anchor (the seed they were built at)
    double RM = 0, RX = 0;
    std::vector<double> th, sw, thm;
    std::vector<KeyInfo> keys;
    std::vector<EqInfo> eqs;
    std::vector<RowInfo> rows;
    std::vector<UnkInfo> unk;
    std::vector<std::vector<double>> seedamp;
    Compl cs, ct;
    std::vector<MField> mf;
    std::vector<MKey> mk;
    int keyof[6][7];
    double tval = 0, tjac = 0, tmatch = 0;

    explicit Eval(const std::string& path) { read(path); }

    // ---------------------------------------------------------------- data
    void read(const std::string& path)
    {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("stageB: cannot read " + path);
        std::string tag;
        auto need = [&](const char* t) { in >> tag; if (tag != t) throw std::runtime_error(std::string("stageB data: expected ") + t + ", got " + tag); };
        need("NT"); in >> NT;
        need("RM"); in >> RM;
        need("GRID"); in >> NR >> NX >> RX;
        need("THETA"); in >> NTH;
        th.resize(NTH); sw.resize(NTH);
        for (int t = 0; t < NTH; t++) in >> th[t] >> sw[t];
        need("KEYS"); in >> NK;
        keys.resize(NK);
        for (auto& r : keyof) for (int& v : r) v = -1;
        for (int k = 0; k < NK; k++) {
            std::string kd;
            in >> keys[k].f >> keys[k].gi >> kd >> keys[k].nfull;
            keys[k].kind = kind_of(kd);
            keyof[keys[k].f][keys[k].gi] = k;
        }
        need("EQS"); in >> NE;
        eqs.resize(NE);
        for (auto& e : eqs) {
            in >> e.name >> e.c0 >> e.c1 >> e.nth >> e.nb;
            e.Q.resize(static_cast<size_t>(e.nth) * e.nb);
            for (double& v : e.Q) in >> v;
        }
        need("ROWS"); in >> NROW;
        rows.resize(NROW);
        for (auto& r : rows) in >> r.e >> r.g >> r.i >> r.part >> r.nr >> r.twist;
        need("UNK"); in >> NU;
        unk.resize(NU);
        for (auto& u : unk) { std::string t; in >> t >> u.key >> u.ia >> u.u0; u.t = t[0]; }
        need("SEEDAMP");
        seedamp.resize(NK);
        for (int k = 0; k < NK; k++) { seedamp[k].resize(keys[k].nfull); for (double& v : seedamp[k]) in >> v; }
        for (int sct = 0; sct < 2; sct++) {
            Compl& C = sct == 0 ? cs : ct;
            std::string sec;
            need("COMPL"); in >> sec >> C.nlow >> C.nhigh;
            int namp = 0;
            for (int j = 0; j < C.nlow; j++) {
                std::string t; in >> t;
                if (t == "DN") { C.ltype.push_back(1); C.lkey.push_back(-1); C.lia.push_back(-1); }
                else if (t == "LOG") { C.ltype.push_back(2); C.lkey.push_back(-1); C.lia.push_back(-1); }
                else { int k, ia; in >> k >> ia; C.ltype.push_back(0); C.lkey.push_back(k); C.lia.push_back(ia); namp++; }
            }
            for (int j = 0; j < C.nhigh; j++) { int k, ia; in >> k >> ia; C.hkey.push_back(k); C.hia.push_back(ia); }
            C.Phi.resize(static_cast<size_t>(C.nhigh) * C.nlow);
            for (double& v : C.Phi) in >> v;
            if (sct == 0) {
                C.Vlow.resize(namp); for (double& v : C.Vlow) in >> v;
                C.corr.resize(C.nhigh); for (double& v : C.corr) in >> v;
            }
        }
        need("MATCH"); in >> NTHM;
        thm.resize(NTHM);
        for (double& v : thm) in >> v;
        mf.resize(6);
        for (int f = 0; f < 6; f++) {
            std::string F, kd; in >> F >> kd >> mf[f].nm;
            mf[f].kind = kind_of(kd);
            mf[f].modes.resize(mf[f].nm);
            for (int& j : mf[f].modes) in >> j;
            mf[f].pinv.resize(static_cast<size_t>(mf[f].nm) * NTHM);
            for (double& v : mf[f].pinv) in >> v;
        }
        need("MKEYS"); in >> NMK >> NOFF;
        mk.resize(NMK);
        for (auto& m : mk) { in >> m.F >> m.d >> m.k >> m.kept; m.off.resize(NOFF); for (double& v : m.off) in >> v; }
        if (!in) throw std::runtime_error("stageB data: truncated " + path);
        NTHR = NROW;
        if (in >> tag) {
            if (tag != "ORTH") throw std::runtime_error("stageB data: expected ORTH or the end, got " + tag);
            in >> NORTH;
            orthu0.resize(NU);
            for (double& v : orthu0) in >> v;
            orth.resize(static_cast<size_t>(NORTH) * NU);
            for (double& v : orth) in >> v;
            if (!in) throw std::runtime_error("stageB data: truncated ORTH in " + path);
            NROW = NTHR + NORTH;
        }
    }

    // ---------------------------------------------------------------- states
    // amp / lc: per key, full-mode coefficients; complex so the matching can take a complex step
    struct State { std::vector<std::vector<cplx>> amp, lc; cplx n; std::vector<cplx> off; };

    State state(const std::vector<cplx>& u) const
    {
        State s;
        s.amp.resize(NK); s.lc.resize(NK);
        for (int k = 0; k < NK; k++) {
            s.amp[k].assign(seedamp[k].begin(), seedamp[k].end());
            s.lc[k].assign(keys[k].nfull, cplx(0.0));
        }
        s.n = std::sqrt(2.0);
        for (int q = 0; q < NU; q++) {
            const UnkInfo& U = unk[q];
            if (U.t == 'a') s.amp[U.key][U.ia] = u[q];
            else if (U.t == 'l') s.lc[U.key][U.ia] = u[q];
            else if (U.t == 'n') s.n = u[q];
            else s.off.push_back(u[q]);
        }
        complete(s, cs, true);
        complete(s, ct, false);
        return s;
    }

    void complete(State& s, const Compl& C, bool scalar) const
    {
        if (C.nhigh == 0) return;                         // round 346: no completion (grades 4-5 are unknowns)
        std::vector<cplx> low(C.nlow), llow;
        std::vector<int> ampidx;
        for (int j = 0; j < C.nlow; j++) {
            if (C.ltype[j] == 0) {
                low[j] = s.amp[C.lkey[j]][C.lia[j]] - seedamp[C.lkey[j]][C.lia[j]];
                llow.push_back(s.lc[C.lkey[j]][C.lia[j]]);
                ampidx.push_back(j);
            }
        }
        cplx lam = 0.0;
        if (scalar) {
            double vv = 0.0;
            for (size_t a = 0; a < llow.size(); a++) { lam += C.Vlow[a] * llow[a]; vv += C.Vlow[a] * C.Vlow[a]; }
            lam /= vv;
            for (int j = 0; j < C.nlow; j++) {
                if (C.ltype[j] == 1) low[j] = s.n - std::sqrt(2.0);
                if (C.ltype[j] == 2) low[j] = lam;
            }
        }
        for (int h = 0; h < C.nhigh; h++) {
            cplx hi = 0.0, lhi = 0.0;
            for (int j = 0; j < C.nlow; j++) hi += C.Phi[static_cast<size_t>(h) * C.nlow + j] * low[j];
            for (size_t a = 0; a < ampidx.size(); a++) lhi += C.Phi[static_cast<size_t>(h) * C.nlow + ampidx[a]] * llow[a];
            if (scalar) lhi += C.corr[h] * lam;
            s.amp[C.hkey[h]][C.hia[h]] = seedamp[C.hkey[h]][C.hia[h]] + hi;
            s.lc[C.hkey[h]][C.hia[h]] = lhi;
        }
    }

    // ---------------------------------------------------------------- generated dispatch
    static Jets1 jets1(int f)
    {
        static const Jets1 t[6] = {sbg::jets1_psi2, sbg::jets1_Phb, sbg::jets1_qf, sbg::jets1_br, sbg::jets1_bt, sbg::jets1_Qb};
        return t[f];
    }
    static Jets2 jets2(int f)
    {
        static const Jets2 t[6] = {sbg::jets2_psi2, sbg::jets2_Phb, sbg::jets2_qf, sbg::jets2_br, sbg::jets2_bt, sbg::jets2_Qb};
        return t[f];
    }
    static const int* jpos(int f)
    {
        static const int* t[6] = {sbg::JPOS_psi2, sbg::JPOS_Phb, sbg::JPOS_qf, sbg::JPOS_br, sbg::JPOS_bt, sbg::JPOS_Qb};
        return t[f];
    }
    static int d2n(int f)
    {
        static const int t[6] = {sbg::D2N_psi2, sbg::D2N_Phb, sbg::D2N_qf, sbg::D2N_br, sbg::D2N_bt, sbg::D2N_Qb};
        return t[f];
    }
    static const int (*d2p(int f))[3]
    {
        static const int (*t[6])[3] = {sbg::D2P_psi2, sbg::D2P_Phb, sbg::D2P_qf, sbg::D2P_br, sbg::D2P_bt, sbg::D2P_Qb};
        return t[f];
    }
    static EqFn eqfn(const std::string& nm)
    {
        if (nm == "E_q") return sbg::eq_Eq;
        if (nm == "E_sigma") return sbg::eq_Esigma;
        if (nm == "E_Phi") return sbg::eq_EPhi;
        if (nm == "E_sh^r") return sbg::eq_Eshr;
        if (nm == "E_sh^th") return sbg::eq_Eshth;
        if (nm == "E_Q") return sbg::eq_EQ;
        if (nm == "M") return sbg::eq_M;
        throw std::runtime_error("stageB: no generated equation " + nm);
    }
    static int hn(const std::string& nm)
    {
        if (nm == "E_q") return sbg::HN_Eq;
        if (nm == "E_sigma") return sbg::HN_Esigma;
        if (nm == "E_Phi") return sbg::HN_EPhi;
        if (nm == "E_sh^r") return sbg::HN_Eshr;
        if (nm == "E_sh^th") return sbg::HN_Eshth;
        if (nm == "E_Q") return sbg::HN_EQ;
        return sbg::HN_M;
    }
    static const int (*hp(const std::string& nm))[2]
    {
        if (nm == "E_q") return sbg::HP_Eq;
        if (nm == "E_sigma") return sbg::HP_Esigma;
        if (nm == "E_Phi") return sbg::HP_EPhi;
        if (nm == "E_sh^r") return sbg::HP_Eshr;
        if (nm == "E_sh^th") return sbg::HP_Eshth;
        if (nm == "E_Q") return sbg::HP_EQ;
        return sbg::HP_M;
    }

    // ---------------------------------------------------------------- throat rows
    // coefficient space: amp coefficients (key, mode) -> ca[key] offset; lc -> cl[key] offset; n -> the last
    std::vector<int> caoff, cloff;
    int NC = 0;
    void coef_layout()
    {
        if (NC) return;
        caoff.resize(NK); cloff.resize(NK);
        int o = 0;
        for (int k = 0; k < NK; k++) { caoff[k] = o; o += keys[k].nfull; }
        for (int k = 0; k < NK; k++) { cloff[k] = o; o += keys[k].nfull; }
        NC = o + 1;
    }

    // values (rows[NROW]) and, if G != nullptr, d rows / d coefficients (NROW x NC, row-major)
    void throat(const State& s, double Jv, std::vector<double>& out, std::vector<double>* G)
    {
        coef_layout();
        const int NS = 217;                       // 108 A symbols, 108 L symbols, n
        const bool grad = (G != nullptr);
        bool anyL = false;
        for (int k = 0; k < NK; k++) for (const cplx& v : s.lc[k]) if (v != 0.0) anyL = true;
        const cplx n = s.n;
        out.assign(NTHR, 0.0);
        if (grad) G->assign(static_cast<size_t>(NTHR) * NC, 0.0);
        const int NV = grad ? NS + 1 : 1;         // slot 0 the value, 1.. the gradient
        std::vector<int> pmin(NE), pmax(NE);
        for (int e = 0; e < NE; e++) { pmin[e] = eqs[e].c1 - 2; pmax[e] = eqs[e].c1 + 3; }
        // grid
        std::vector<cplx> rg(NR), xg(NX);
        for (int a = 0; a < NR; a++) rg[a] = std::polar(1.0, 2 * M_PI * a / NR);
        for (int b = 0; b < NX; b++) xg[b] = RX * std::polar(1.0, 2 * M_PI * b / NX);
        std::vector<cplx> coef(static_cast<size_t>(NE) * 2 * 6 * NV);
        std::vector<cplx> Sx(static_cast<size_t>(NE) * 2 * NV);
        double angv[3];
        for (int t = 0; t < NTH; t++) {
            // amplitude functions on this node: per field the 18 symbols (grade index x derivative)
            cplx A[6][18], L[6][18];
            std::vector<std::vector<std::array<double, 3>>> ang(NK);
            for (int f = 0; f < 6; f++) for (int s_ = 0; s_ < 18; s_++) { A[f][s_] = 0.0; L[f][s_] = 0.0; }
            for (int k = 0; k < NK; k++) {
                ang[k].resize(keys[k].nfull);
                for (int j = 0; j < keys[k].nfull; j++) {
                    angular(keys[k].kind, j + (keys[k].kind == 2 ? 1 : 0), th[t], angv);
                    ang[k][j] = {angv[0], angv[1], angv[2]};
                    for (int d = 0; d < 3; d++) {
                        A[keys[k].f][3 * keys[k].gi + d] += s.amp[k][j] * angv[d];
                        L[keys[k].f][3 * keys[k].gi + d] += s.lc[k][j] * angv[d];
                    }
                }
            }
            std::fill(coef.begin(), coef.end(), cplx(0.0));
            for (int b = 0; b < NX; b++) {
                std::fill(Sx.begin(), Sx.end(), cplx(0.0));
                for (int a = 0; a < NR; a++) {
                    const cplx r = rg[a], x = xg[b];
                    cplx Jt[36], dJ[6][108], dJn[6][6], d2J[6][400], d2Jn[6][108];
                    for (int f = 0; f < 6; f++) {
                        cplx J6[6];
                        for (int q = 0; q < 108; q++) dJ[f][q] = 0.0;
                        for (int q = 0; q < 6; q++) dJn[f][q] = 0.0;
                        jets1(f)(A[f], r, x, n, J6, dJ[f], dJn[f]);
                        for (int q = 0; q < 6; q++) Jt[jpos(f)[q]] = J6[q];
                        if (anyL) {
                            for (int q = 0; q < 108; q++) d2Jn[f][q] = 0.0;
                            jets2(f)(A[f], r, x, n, d2J[f], d2Jn[f]);
                        }
                    }
                    // companions' jets: DJ = dJ/da . L ; CJ the commutator; and their derivatives
                    cplx DJ[36], CJ[36];
                    for (int q = 0; q < 36; q++) { DJ[q] = 0.0; CJ[q] = 0.0; }
                    cplx dDJ[6][6][18], dDJn[6][6];       // d DJ_(f, jet) / d A_s ; d / dn
                    if (anyL) {
                        for (int f = 0; f < 6; f++) {
                            for (int q = 0; q < 6; q++) {
                                cplx acc = 0.0;
                                for (int s_ = 0; s_ < 18; s_++) acc += dJ[f][18 * q + s_] * L[f][s_];
                                DJ[jpos(f)[q]] = acc;
                                cplx an = 0.0;
                                for (int s_ = 0; s_ < 18; s_++) an += d2Jn[f][18 * q + s_] * L[f][s_];
                                dDJn[f][q] = an;
                                for (int s_ = 0; s_ < 18; s_++) dDJ[f][q][s_] = 0.0;
                            }
                            const int (*pp)[3] = d2p(f);
                            for (int m = 0; m < d2n(f); m++) {
                                const int q = pp[m][0], s1 = pp[m][1], s2 = pp[m][2];
                                dDJ[f][q][s1] += d2J[f][m] * L[f][s2];
                                if (s1 != s2) dDJ[f][q][s2] += d2J[f][m] * L[f][s1];
                            }
                            const int* jp = jpos(f);
                            CJ[jp[1]] = n / r * DJ[jp[0]];
                            CJ[jp[3]] = 2.0 * n / r * DJ[jp[1]] - n / (r * r) * DJ[jp[0]];
                            CJ[jp[4]] = n / r * DJ[jp[2]];
                        }
                    }
                    for (int e = 0; e < NE; e++) {
                        cplx E, P[36], H[200];
                        for (int q = 0; q < 36; q++) P[q] = 0.0;
                        const int nh = hn(eqs[e].name);
                        eqfn(eqs[e].name)(Jt, r, th[t], Jv, E, P, H);
                        cplx y0 = E, y1 = 0.0;
                        for (int q = 0; q < 36; q++) { y0 += P[q] * CJ[q]; y1 += P[q] * DJ[q]; }
                        const cplx w = std::pow(r, -eqs[e].c0);
                        cplx* S0 = &Sx[(static_cast<size_t>(e) * 2 + 0) * NV];
                        cplx* S1 = &Sx[(static_cast<size_t>(e) * 2 + 1) * NV];
                        S0[0] += y0 * w;
                        S1[0] += y1 * w;
                        if (!grad) continue;
                        // hc = H.CJ, hd = H.DJ
                        cplx hc[36], hd[36];
                        for (int q = 0; q < 36; q++) { hc[q] = 0.0; hd[q] = 0.0; }
                        if (anyL) {
                            const int (*hpp)[2] = hp(eqs[e].name);
                            for (int m = 0; m < nh; m++) {
                                const int i1 = hpp[m][0], i2 = hpp[m][1];
                                hc[i1] += H[m] * CJ[i2]; hd[i1] += H[m] * DJ[i2];
                                if (i1 != i2) { hc[i2] += H[m] * CJ[i1]; hd[i2] += H[m] * DJ[i1]; }
                            }
                        }
                        for (int f = 0; f < 6; f++) {
                            const int* jp = jpos(f);
                            for (int s_ = 0; s_ < 18; s_++) {
                                cplx gA0 = 0.0, gA1 = 0.0, gL0 = 0.0, gL1 = 0.0;
                                for (int q = 0; q < 6; q++) {
                                    const cplx d = dJ[f][18 * q + s_];
                                    gA0 += (P[jp[q]] + hc[jp[q]]) * d;
                                    gA1 += hd[jp[q]] * d;
                                    gL1 += P[jp[q]] * d;
                                }
                                // d CJ / d L_s from the commutator of dJ[.][s]
                                gL0 += P[jp[1]] * (n / r * dJ[f][18 * 0 + s_])
                                     + P[jp[3]] * (2.0 * n / r * dJ[f][18 * 1 + s_] - n / (r * r) * dJ[f][18 * 0 + s_])
                                     + P[jp[4]] * (n / r * dJ[f][18 * 2 + s_]);
                                if (anyL) {
                                    for (int q = 0; q < 6; q++) gA1 += P[jp[q]] * dDJ[f][q][s_];
                                    gA0 += P[jp[1]] * (n / r * dDJ[f][0][s_])
                                         + P[jp[3]] * (2.0 * n / r * dDJ[f][1][s_] - n / (r * r) * dDJ[f][0][s_])
                                         + P[jp[4]] * (n / r * dDJ[f][2][s_]);
                                }
                                S0[1 + 18 * f + s_] += gA0 * w;
                                S1[1 + 18 * f + s_] += gA1 * w;
                                S0[1 + 108 + 18 * f + s_] += gL0 * w;
                                S1[1 + 108 + 18 * f + s_] += gL1 * w;
                            }
                        }
                        cplx gn0 = 0.0, gn1 = 0.0;
                        for (int f = 0; f < 6; f++) {
                            const int* jp = jpos(f);
                            for (int q = 0; q < 6; q++) {
                                gn0 += (P[jp[q]] + hc[jp[q]]) * dJn[f][q];
                                gn1 += hd[jp[q]] * dJn[f][q];
                            }
                            if (anyL) {
                                for (int q = 0; q < 6; q++) gn1 += P[jp[q]] * dDJn[f][q];
                                gn0 += P[jp[1]] * (DJ[jp[0]] / r + n / r * dDJn[f][0])
                                     + P[jp[3]] * (2.0 / r * DJ[jp[1]] + 2.0 * n / r * dDJn[f][1] - DJ[jp[0]] / (r * r) - n / (r * r) * dDJn[f][0])
                                     + P[jp[4]] * (DJ[jp[2]] / r + n / r * dDJn[f][2]);
                            }
                        }
                        S0[1 + 216] += gn0 * w;
                        S1[1 + 216] += gn1 * w;
                    }
                }
                // fold this x-column into the rho^p coefficients
                for (int e = 0; e < NE; e++)
                    for (int g = 0; g < 6; g++) {
                        const int p = pmin[e] + g;
                        const cplx w = std::pow(xg[b], -p) / static_cast<double>(NR * NX);
                        for (int y = 0; y < 2; y++) {
                            const cplx* S = &Sx[(static_cast<size_t>(e) * 2 + y) * NV];
                            cplx* C = &coef[((static_cast<size_t>(e) * 2 + y) * 6 + g) * NV];
                            for (int v = 0; v < NV; v++) C[v] += S[v] * w;
                        }
                    }
            }
            // project this node onto the rows
            for (int ro = 0; ro < NTHR; ro++) {
                const RowInfo& R = rows[ro];
                const EqInfo& Eq = eqs[R.e];
                const int g = R.g + 2;                 // g from -2
                const double qw = Eq.Q[static_cast<size_t>(t) * Eq.nb + R.i] * sw[t] / R.nr;
                const cplx* C = &coef[((static_cast<size_t>(R.e) * 2 + R.part) * 6 + g) * NV];
                out[ro] += qw * C[0].real();
                if (!grad) continue;
                double* Gr = &(*G)[static_cast<size_t>(ro) * NC];
                for (int k = 0; k < NK; k++) {
                    const int f = keys[k].f, gi = keys[k].gi;
                    for (int j = 0; j < keys[k].nfull; j++) {
                        double ga = 0.0, gl = 0.0;
                        for (int d = 0; d < 3; d++) {
                            ga += C[1 + 18 * f + 3 * gi + d].real() * ang[k][j][d];
                            gl += C[1 + 108 + 18 * f + 3 * gi + d].real() * ang[k][j][d];
                        }
                        Gr[caoff[k] + j] += qw * ga;
                        Gr[cloff[k] + j] += qw * gl;
                    }
                }
                Gr[NC - 1] += qw * C[1 + 216].real();
            }
        }
    }

    // the linear chain coefficients <- unknowns: CU (NC x NU): exact (the state map is affine)
    std::vector<double> chain()
    {
        coef_layout();
        std::vector<double> CU(static_cast<size_t>(NC) * NU, 0.0);
        std::vector<cplx> u0(NU);
        for (int q = 0; q < NU; q++) u0[q] = unk[q].u0;
        const State s0 = state(u0);
        for (int q = 0; q < NU; q++) {
            if (unk[q].t == 'c') continue;
            std::vector<cplx> u1 = u0;
            u1[q] += 1.0;
            const State s1 = state(u1);
            for (int k = 0; k < NK; k++)
                for (int j = 0; j < keys[k].nfull; j++) {
                    CU[static_cast<size_t>(caoff[k] + j) * NU + q] = (s1.amp[k][j] - s0.amp[k][j]).real();
                    CU[static_cast<size_t>(cloff[k] + j) * NU + q] = (s1.lc[k][j] - s0.lc[k][j]).real();
                }
            CU[static_cast<size_t>(NC - 1) * NU + q] = (s1.n - s0.n).real();
        }
        return CU;
    }

    // ---------------------------------------------------------------- matching
    std::vector<cplx> matching(const State& s) const
    {
        std::vector<cplx> out(NMK, 0.0);
        const cplx n = s.n;
        const cplx x = std::pow(cplx(RM), n);
        const cplx lnr = n * std::log(RM);
        double angv[3];
        // per field, values at the nodes
        std::vector<std::vector<cplx>> v0(6, std::vector<cplx>(NTHM)), v1(6, std::vector<cplx>(NTHM));
        for (int t = 0; t < NTHM; t++) {
            cplx A[6][18], L[6][18];
            for (int f = 0; f < 6; f++) for (int q = 0; q < 18; q++) { A[f][q] = 0.0; L[f][q] = 0.0; }
            for (int k = 0; k < NK; k++)
                for (int j = 0; j < keys[k].nfull; j++) {
                    angular(keys[k].kind, j + (keys[k].kind == 2 ? 1 : 0), thm[t], angv);
                    for (int d = 0; d < 3; d++) {
                        A[keys[k].f][3 * keys[k].gi + d] += s.amp[k][j] * angv[d];
                        L[keys[k].f][3 * keys[k].gi + d] += s.lc[k][j] * angv[d];
                    }
                }
            for (int f = 0; f < 6; f++) {
                cplx J6[6], dJ[108], dJn[6];
                for (int q = 0; q < 108; q++) dJ[q] = 0.0;
                jets1(f)(A[f], cplx(RM), x, n, J6, dJ, dJn);
                cplx D0 = 0.0, D1 = 0.0;
                for (int q = 0; q < 18; q++) { D0 += dJ[q] * L[f][q]; D1 += dJ[18 + q] * L[f][q]; }
                v0[f][t] = J6[0] + lnr * D0;
                v1[f][t] = J6[1] + lnr * D1 + n / RM * D0;
            }
        }
        for (int q = 0; q < NMK; q++) {
            const MKey& K = mk[q];
            const MField& M = mf[K.F];
            int row = -1;
            for (int a = 0; a < M.nm; a++) if (M.modes[a] == K.k) row = a;
            if (row < 0) throw std::runtime_error("stageB: matching key outside the padded modes");
            cplx acc = 0.0;
            const std::vector<cplx>& v = K.d ? v1[K.F] : v0[K.F];
            for (int t = 0; t < NTHM; t++) acc += M.pinv[static_cast<size_t>(row) * NTHM + t] * v[t];
            for (int c = 0; c < NOFF; c++) acc += K.off[c] * s.off[c];
            out[q] = acc;
        }
        return out;
    }

    // the orthogonality rows' values, O (u - u_seed), appended to the throat rows
    void append_orth(const std::vector<double>& u, std::vector<double>& rv) const
    {
        rv.resize(NROW);
        for (int o = 0; o < NORTH; o++) {
            double acc = 0.0;
            for (int q = 0; q < NU; q++) acc += orth[static_cast<size_t>(o) * NU + q] * (u[q] - orthu0[q]);
            rv[NTHR + o] = acc;
        }
    }

    // ---------------------------------------------------------------- the public calls
    void values(const std::vector<double>& u, double Jv, std::vector<double>& rv, std::vector<double>& mv)
    {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<cplx> uc(u.begin(), u.end());
        const State s = state(uc);
        throat(s, Jv, rv, nullptr);
        append_orth(u, rv);
        auto t1 = std::chrono::steady_clock::now();
        const std::vector<cplx> m = matching(s);
        mv.resize(NMK);
        for (int q = 0; q < NMK; q++) mv[q] = m[q].real();
        auto t2 = std::chrono::steady_clock::now();
        tval = std::chrono::duration<double>(t1 - t0).count();
        tmatch = std::chrono::duration<double>(t2 - t1).count();
    }

    // Jr (NROW x NU), Jm (NMK x NU), row-major
    void jacobian(const std::vector<double>& u, double Jv, std::vector<double>& Jr, std::vector<double>& Jm)
    {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<cplx> uc(u.begin(), u.end());
        const State s = state(uc);
        std::vector<double> rv, G;
        throat(s, Jv, rv, &G);
        static std::vector<double> CU;
        static int CUfor = -1;
        if (CUfor != NU) { CU = chain(); CUfor = NU; }
        Jr.assign(static_cast<size_t>(NROW) * NU, 0.0);
        for (int ro = 0; ro < NTHR; ro++)
            for (int c = 0; c < NC; c++) {
                const double g = G[static_cast<size_t>(ro) * NC + c];
                if (g == 0.0) continue;
                const double* cu = &CU[static_cast<size_t>(c) * NU];
                double* jr = &Jr[static_cast<size_t>(ro) * NU];
                for (int q = 0; q < NU; q++) jr[q] += g * cu[q];
            }
        for (int o = 0; o < NORTH; o++)
            for (int q = 0; q < NU; q++) Jr[static_cast<size_t>(NTHR + o) * NU + q] = orth[static_cast<size_t>(o) * NU + q];
        Jm.assign(static_cast<size_t>(NMK) * NU, 0.0);
        const double h = 1e-30;
        for (int q = 0; q < NU; q++) {
            std::vector<cplx> u1 = uc;
            u1[q] += cplx(0.0, h);
            const std::vector<cplx> m = matching(state(u1));
            for (int k = 0; k < NMK; k++) Jm[static_cast<size_t>(k) * NU + q] = m[k].imag() / h;
        }
        tjac = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};
}  // namespace sbe

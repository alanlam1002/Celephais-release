// Round 463 part 20 (research round 1005, N3b): the LINEARISED EXTERNAL THROAT -- a drop-in for sbe::Eval under TRUMPET_STAGEB_LIN.
// The throat is the Mellin series (scripts/n2a_nl.py, the cylinder-centred recursion) linearised about its parameters p0 by the engine's
// validated tangents, written by scripts/n3b_linthroat.py:
//   unknowns   the Mellin parameters p (grade-1 data, the aligned-2 genuine amplitudes, the families' amplitudes)
//   rows       the Mellin throat's equations (the aligned-1 Galerkin rows and the l = 0 conditions): h(p) = H0 + JH (p - p0)
//   matching   the series' stage-B matching keys at r_m (the data's own theta nodes and least-squares projectors): m(p) = M0 + JM (p - p0)
// A Python outer loop re-linearises about the solve's p and repeats (the throat exact at convergence).  The interface mirrors sbe::Eval's
// public members as finiteJ_probe_main.cpp uses them (NU, unk, NROW, NTHR, rows, eqs, NMK, mk, RM, values, jacobian, timings).
// File (whitespace-separated):  LINTHROAT 1 | RM r | UNK NU, NU x (name u0) | EQS NE, NE x name | ROWS NROW, NROW x (e g i part)
//                               | MKEYS NMK, NMK x (F d k kept) | P0 NU | H0 NROW | JH NROW x NU | M0 NMK | JM NMK x NU
#pragma once
#include "stageB_eval.hpp"

namespace sbe {
class LinEval {
  public:
    int NU = 0, NROW = 0, NTHR = 0, NMK = 0, NE = 0;
    double RM = 0;
    std::vector<EqInfo> eqs;
    std::vector<RowInfo> rows;
    std::vector<UnkInfo> unk;
    std::vector<MKey> mk;
    std::vector<std::string> uname;
    std::vector<double> p0, H0, JH, M0, JM;
    double tval = 0, tjac = 0, tmatch = 0;

    explicit LinEval(const std::string& path)
    {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("stageB-lin: cannot read " + path);
        std::string tag;
        int ver = 0;
        auto need = [&](const char* t) { in >> tag; if (tag != t) throw std::runtime_error(std::string("stageB-lin: expected ") + t + ", got " + tag); };
        need("LINTHROAT"); in >> ver;
        need("RM"); in >> RM;
        need("UNK"); in >> NU;
        unk.resize(NU); uname.resize(NU);
        for (int q = 0; q < NU; q++) { in >> uname[q] >> unk[q].u0; unk[q].t = 'p'; unk[q].key = q; unk[q].ia = 0; }
        need("EQS"); in >> NE;
        eqs.resize(NE);
        for (auto& e : eqs) { in >> e.name; e.c0 = e.c1 = e.nth = e.nb = 0; }
        need("ROWS"); in >> NROW;
        rows.resize(NROW);
        for (auto& r : rows) { in >> r.e >> r.g >> r.i >> r.part; r.nr = 1.0; r.twist = 0; }
        NTHR = NROW;
        need("MKEYS"); in >> NMK;
        mk.resize(NMK);
        for (auto& m : mk) in >> m.F >> m.d >> m.k >> m.kept;
        auto vec = [&](const char* t, std::vector<double>& v, size_t n) { need(t); v.resize(n); for (double& x : v) in >> x; };
        vec("P0", p0, NU);
        vec("H0", H0, NROW);
        vec("JH", JH, static_cast<size_t>(NROW) * NU);
        vec("M0", M0, NMK);
        vec("JM", JM, static_cast<size_t>(NMK) * NU);
        if (!in) throw std::runtime_error("stageB-lin: truncated " + path);
    }

    void values(const std::vector<double>& u, double, std::vector<double>& rv, std::vector<double>& mv)
    {
        rv.assign(NROW, 0.0);
        mv.assign(NMK, 0.0);
        for (int r = 0; r < NROW; r++) {
            double acc = H0[r];
            const double* j = &JH[static_cast<size_t>(r) * NU];
            for (int q = 0; q < NU; q++) acc += j[q] * (u[q] - p0[q]);
            rv[r] = acc;
        }
        for (int k = 0; k < NMK; k++) {
            double acc = M0[k];
            const double* j = &JM[static_cast<size_t>(k) * NU];
            for (int q = 0; q < NU; q++) acc += j[q] * (u[q] - p0[q]);
            mv[k] = acc;
        }
    }

    void jacobian(const std::vector<double>&, double, std::vector<double>& Jr, std::vector<double>& Jm)
    {
        Jr = JH;
        Jm = JM;
    }
};
}  // namespace sbe

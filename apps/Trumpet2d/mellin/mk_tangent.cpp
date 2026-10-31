// Mellin throat recursion, M3 (code round 463): the tangent step (mk_tangent.hpp).  Compiled with -ffp-contract=off.
#include "mk_tangent.hpp"

#include <algorithm>
#include <cmath>

namespace mk {
namespace tn {

using ct::mul;

static inline double cabs_(cd z) { return std::hypot(z.real(), z.imag()); }
static inline cd rc(double r) { return cd(r, 0.0); }
static double absmax(const Vec& v) {
    double m = 0.0;
    for (auto& z : v) m = std::max(m, cabs_(z));
    return m;
}
// dX + sum(dl[q] * Wq[q] for q in range(rk))
static Vec add_dirs(const Vec& dX, const Vec& dl, const std::vector<Vec>& Wq) {
    Vec S(dX.size()), out(dX.size());
    for (size_t q = 0; q < Wq.size(); q++)
        for (size_t i = 0; i < dX.size(); i++) S[i] = q == 0 ? mul(dl[0], Wq[0][i]) : S[i] + mul(dl[q], Wq[q][i]);
    for (size_t i = 0; i < dX.size(); i++) out[i] = dX[i] + S[i];
    return out;
}

void tan_step(const StepState& st, const Seed& seed, double sc, double ctol, int iters, int K, int NB, int nn, const Vec& e1, double R,
              const std::function<Vec(const Vec*)>& drows, StepOut& o) {
    o = StepOut();
    int rk = st.rk;
    bool amp_done = false;
    // the residue amplitudes' derivatives from the step's data rule
    auto amp = [&](const Vec& dX) -> Vec {
        if (!rk) return dX;
        Vec dl(rk, cd(0.0, 0.0));
        if (st.isint) {
            if (!st.has_src) return dX;
            int m = 2 * NB;
            Vec t0d(m, cd(0.0, 0.0));
            for (int i = 0; i < m; i++)
                if (!st.src_tgt[i].empty()) t0d[i] = rc(seed.kind == "tgt" && seed.name == st.src_tgt[i] ? 1.0 : 0.0);
            Vec rho = dt::tg(dX, NB, e1, R);
            std::vector<Vec> Atl(rk);
            for (int q = 0; q < rk; q++) Atl[q] = dt::tg(st.Wq[q], NB, e1, R);
            Vec A(size_t(m) * rk), b(m);
            for (int i = 0; i < m; i++) {
                double w = std::sqrt(st.wt[i]);
                for (int q = 0; q < rk; q++) A[size_t(i) * rk + q] = mul(rc(w), Atl[q][i]);
                b[i] = -mul(rc(w), rho[i] - t0d[i]);                // lstsq(A_, -b_)
            }
            dl = la::lstsq(A, m, rk, b, 1);
        } else {
            // the forward: X0 + c_datum W, the corrections held -- the datum's derivative ADDED once to the start's own component
            if (amp_done) return dX;
            amp_done = true;
            if (st.has_label && seed.kind == "lap" && seed.name == st.label) dl[0] = rc(1.0);
        }
        return add_dirs(dX, dl, st.Wq);
    };
    // the forward's step, linearised: X0 = -M^-1 S(strip(C)), the amplitudes from the data rule, then the kept grade's rows corrected
    // against the full rows (amplitudes held)
    Vec dX = st::correction(drows(nullptr), st.Minv, K, NB, e1, R);
    for (int it = 0; it < iters; it++) {
        dX = amp(dX);
        Vec r = drows(&dX);
        if (st.kept_only)
            for (int m = 0; m < K; m++)
                for (int j = nn; j < NB; j++) r[size_t(m) * NB + j] = 0.0;
        Vec upd = st::correction(r, st.Minv, K, NB, e1, R);
        if (rk) st::hold(upd, st.Wq, st.kept_only ? nn : NB, NB, e1, R);
        double du = absmax(ct::moments_of(upd, K, NB, e1, R)), xm = absmax(ct::moments_of(dX, K, NB, e1, R));
        double sx = std::max(xm, sc);
        o.hh.push_back(sx != 0.0 ? du / std::max(sx, 1e-300) : du);
        o.hha.push_back(du);
        for (size_t i = 0; i < dX.size(); i++) dX[i] = dX[i] + upd[i];
        size_t n = o.hh.size();
        if (du <= ctol * std::max(sx, 1e-300) || (sx == 0.0 && du == 0.0) || (n > 1 && o.hh[n - 1] > 0.5 * o.hh[n - 2])) break;
    }
    o.dX = amp(dX);                                                     // the data rule on the final coefficients (as the forward's)
}

}  // namespace tn
}  // namespace mk

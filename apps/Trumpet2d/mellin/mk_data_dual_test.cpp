// Unit test of mk_data_dual.hpp (research round 1092, stage 1(c) part 2): the dual data rule (--cond-fit, production) and data fix against
// central differences of dt::data_rule / dt::data_fix, the start X0, the contour's directions Wq and the targets tg0 moved along two random
// directions.
#include "mk_data_dual.hpp"
#include <cstdio>
#include <random>

using namespace mk;
using cd = std::complex<double>;
using Vec = std::vector<cd>;

static std::mt19937_64 rng(10922);
static Vec rnd(size_t n, double a = 1.0) {
    std::normal_distribution<double> g(0.0, 1.0);
    Vec v(n);
    for (auto& z : v) z = a * cd(g(rng), g(rng));
    return v;
}
static Vec axpy(const Vec& x, double h, const Vec& d) { Vec o(x.size()); for (size_t i = 0; i < x.size(); i++) o[i] = x[i] + h * d[i]; return o; }
static double nrm(const Vec& v) { double s = 0; for (auto& z : v) s += std::norm(z); return std::sqrt(s); }
static double rel(const Vec& a, const Vec& b) { return nrm(axpy(a, -1.0, b)) / std::max(nrm(b), 1e-300); }

int main() {
    std::printf("mk_data_dual unit test (research round 1092, stage 1(c) part 2): the dual data rule / fix vs central differences of dt\n");
    const int nn = 5, NB = 5, N = 16, ND = 2;
    int fails = 0;
    for (int rk = 1; rk <= 2; rk++) {
        dt::DataIn in;
        in.nn = nn; in.NB = NB; in.rk = rk; in.R = 0.06; in.cond_fit = true; in.notgt = false; in.g = 3.0;
        in.e1.resize(N);
        for (int k = 0; k < N; k++) in.e1[k] = std::polar(1.0, 2 * M_PI * (k + 0.5) / N);
        in.X0 = rnd(size_t(N) * NB, 0.1);
        in.Wq.resize(rk);
        for (int r = 0; r < rk; r++) in.Wq[r] = rnd(size_t(N) * NB, 3.0);
        in.tgtv = rnd(2 * NB, 0.01); in.prevv = Vec(2 * NB, 0.0); in.tg0 = rnd(2 * NB, 0.01);
        in.wt.assign(2 * NB, 0.0);
        for (int i = 0; i < 2 * NB; i++) in.wt[i] = (i % 3 == 0) ? 1.0 : (i % 3 == 1 ? 0.25 : 0.0);
        in.lc.assign(nn, 0);
        dt::Eval ev;
        ev.cond = [](const Vec&) { return Vec(4, 0.0); };
        ev.gate = [](const Vec*, double) { return 0.0; };
        std::vector<Vec> dX0(ND), dtg0(ND);
        std::vector<std::vector<Vec>> dW(ND, std::vector<Vec>(rk));
        for (int q = 0; q < ND; q++) {
            dX0[q] = rnd(in.X0.size(), 0.1); dtg0[q] = rnd(2 * NB, 0.01);
            for (int r = 0; r < rk; r++) dW[q][r] = rnd(in.Wq[r].size(), 0.5);
        }
        dt::DataOut d;
        dt::data_rule(in, ev, d);
        ddl::DataT t;
        ddl::data_rule_t(in, d, dX0, dW, dtg0, ND, t);
        // the fix at a point X with its own tangent
        Vec X = axpy(d.Xg, 1.0, rnd(d.Xg.size(), 0.01));
        std::vector<Vec> dXf(ND);
        for (int q = 0; q < ND; q++) dXf[q] = rnd(X.size(), 0.1);
        std::vector<Vec> dfx;
        Vec fx = ddl::data_fix_t(X, dXf, in, d, t.dWn, dtg0, ND, dfx);
        double w[3][2] = {{0}};
        const double hs[2] = {1e-5, 1e-6};
        for (int q = 0; q < ND; q++)
            for (int ih = 0; ih < 2; ih++) {
                double h = hs[ih];
                auto moved = [&](double s) {
                    dt::DataIn i2 = in;
                    i2.X0 = axpy(in.X0, s, dX0[q]);
                    for (int r = 0; r < rk; r++) i2.Wq[r] = axpy(in.Wq[r], s, dW[q][r]);
                    i2.tg0 = axpy(in.tg0, s, dtg0[q]);
                    return i2;
                };
                dt::DataIn ip = moved(h), im = moved(-h);
                dt::DataOut dp, dm;
                dt::data_rule(ip, ev, dp);
                dt::data_rule(im, ev, dm);
                auto fd = [&](const Vec& a, const Vec& b) { Vec r = axpy(a, -1.0, b); for (auto& z : r) z /= 2 * h; return r; };
                w[0][ih] = std::max(w[0][ih], rel(t.dXg[q], fd(dp.Xg, dm.Xg)));
                w[1][ih] = std::max(w[1][ih], rel(t.dc[q], fd(dp.c, dm.c)));
                Vec fp = dt::data_fix(axpy(X, h, dXf[q]), ip, dp), fm = dt::data_fix(axpy(X, -h, dXf[q]), im, dm);
                w[2][ih] = std::max(w[2][ih], rel(dfx[q], fd(fp, fm)));
            }
        bool vok = rel(fx, dt::data_fix(X, in, d)) < 1e-14;
        const char* nm[3] = {"the start Xg", "the amplitudes c", "the data fix"};
        for (int i = 0; i < 3; i++) {
            bool ok = std::min(w[i][0], w[i][1]) < 1e-7 && (i < 2 || vok);
            if (!ok) fails++;
            std::printf("  rank %d  %-18s rel. error  h 1e-5: %.2e   h 1e-6: %.2e   %s\n", rk, nm[i], w[i][0], w[i][1], ok ? "ok" : "FAIL");
        }
    }
    // research round 1093: the off-integer start (dt::offl_start, --cond-fit) on duals, the datum moved too
    for (int rk = 1; rk <= 2; rk++) {
        Vec e1(N);
        for (int k = 0; k < N; k++) e1[k] = std::polar(1.0, 2 * M_PI * (k + 0.5) / N);
        Vec X0 = rnd(size_t(N) * NB, 0.1);
        std::vector<Vec> W(rk);
        for (int r = 0; r < rk; r++) W[r] = rnd(size_t(N) * NB, 3.0);
        double datum = 0.37;
        dt::Eval ev;
        ev.cond = [](const Vec&) { return Vec(4, 0.0); };
        ev.gate = [](const Vec*, double) { return 0.0; };
        std::vector<Vec> Wq;
        Vec Xg;
        bool hf = false;
        cd amp;
        dt::offl_start(X0, W, nn, NB, e1, 0.06, true, datum, true, ev, Wq, Xg, hf, amp);
        std::vector<Vec> dX0(ND);
        std::vector<std::vector<Vec>> dW(ND, std::vector<Vec>(rk));
        std::vector<double> dd(ND);
        for (int q = 0; q < ND; q++) {
            dX0[q] = rnd(X0.size(), 0.1);
            for (int r = 0; r < rk; r++) dW[q][r] = rnd(W[r].size(), 0.5);
            dd[q] = 0.3 + q;
        }
        ddl::OfflT t;
        ddl::offl_start_t(X0, W, nn, NB, e1, 0.06, true, datum, dX0, dW, dd, ND, Wq, t);
        double w1[2] = {0, 0}, w2[2] = {0, 0};
        const double hs[2] = {1e-5, 1e-6};
        for (int q = 0; q < ND; q++)
            for (int ih = 0; ih < 2; ih++) {
                double h = hs[ih];
                auto at = [&](double s, std::vector<Vec>& Wo, Vec& Xo) {
                    std::vector<Vec> W2(rk);
                    for (int r = 0; r < rk; r++) W2[r] = axpy(W[r], s, dW[q][r]);
                    bool h2; cd a2;
                    dt::offl_start(axpy(X0, s, dX0[q]), W2, nn, NB, e1, 0.06, true, datum + s * dd[q], true, ev, Wo, Xo, h2, a2);
                };
                std::vector<Vec> Wp, Wm;
                Vec Xp, Xm;
                at(h, Wp, Xp);
                at(-h, Wm, Xm);
                auto fd = [&](const Vec& a, const Vec& b) { Vec r = axpy(a, -1.0, b); for (auto& z : r) z /= 2 * h; return r; };
                w1[ih] = std::max(w1[ih], rel(t.dXg[q], fd(Xp, Xm)));
                for (int r = 0; r < rk; r++) w2[ih] = std::max(w2[ih], rel(t.dWq[q][r], fd(Wp[r], Wm[r])));
            }
        bool ok1 = std::min(w1[0], w1[1]) < 1e-7, ok2 = std::min(w2[0], w2[1]) < 1e-7;
        fails += !ok1 + !ok2;
        std::printf("  rank %d  %-18s rel. error  h 1e-5: %.2e   h 1e-6: %.2e   %s\n", rk, "offl start Xg", w1[0], w1[1], ok1 ? "ok" : "FAIL");
        std::printf("  rank %d  %-18s rel. error  h 1e-5: %.2e   h 1e-6: %.2e   %s\n", rk, "offl directions", w2[0], w2[1], ok2 ? "ok" : "FAIL");
    }
    std::printf("%s\n", fails ? "FAILURES" : "all ok");
    return fails ? 1 : 0;
}

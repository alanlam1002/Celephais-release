// Unit test of mk_contour_dual.hpp (research rounds 1074 / 1089 / 1092, the forward-mode replay's stage 1(c), part 1): the dual contour step
// against central differences of ct::contour itself, the operator (3 x NB x NB) and the sources (N x NB) moved along two random directions.
// The operator has one root inside the contour (residue rank 1) in case A and two in case B; the residue directions are compared through
// the projector onto their span (the SVD's phase and basis within the span are a convention).
#include "mk_contour_dual.hpp"
#include <cstdio>
#include <random>

using namespace mk;
using cd = std::complex<double>;
using Vec = std::vector<cd>;

static std::mt19937_64 rng(1092);
static Vec rnd(size_t n, double a = 1.0) {
    std::normal_distribution<double> g(0.0, 1.0);
    Vec v(n);
    for (auto& z : v) z = a * cd(g(rng), g(rng));
    return v;
}
static Vec axpy(const Vec& x, double h, const Vec& d) { Vec o(x.size()); for (size_t i = 0; i < x.size(); i++) o[i] = x[i] + h * d[i]; return o; }
static double nrm(const Vec& v) { double s = 0; for (auto& z : v) s += std::norm(z); return std::sqrt(s); }
static double rel(const Vec& a, const Vec& b) { return nrm(axpy(a, -1.0, b)) / std::max(nrm(b), 1e-300); }

// the projector onto span{W_r} (W_r: N x NB flattened), (N NB) x (N NB)
static Vec proj(const std::vector<Vec>& W) {
    int rk = int(W.size());
    size_t n = W[0].size();
    Vec G(size_t(rk) * rk);
    for (int a = 0; a < rk; a++)
        for (int b = 0; b < rk; b++) {
            cd s = 0.0;
            for (size_t i = 0; i < n; i++) s += std::conj(W[a][i]) * W[b][i];
            G[size_t(a) * rk + b] = s;
        }
    Vec Gi = la::inv(G, rk);
    Vec P(n * n, 0.0);
    for (int a = 0; a < rk; a++)
        for (int b = 0; b < rk; b++)
            for (size_t i = 0; i < n; i++)
                for (size_t j = 0; j < n; j++) P[i * n + j] += W[a][i] * Gi[size_t(a) * rk + b] * std::conj(W[b][j]);
    return P;
}

int main() {
    std::printf("mk_contour_dual unit test (research round 1092, stage 1(c) part 1): the dual contour vs central differences of ct::contour\n");
    const int NB = 6, N = 24, ND = 2;
    const double R = 0.3;
    Vec e1(N);
    for (int k = 0; k < N; k++) e1[k] = std::polar(1.0, 2 * M_PI * (k + 0.5) / N);
    std::vector<double> rw(NB, 1.0), cw(NB, 1.0);
    for (int j = 0; j < NB; j++) { rw[j] = 1.0 + 0.3 * j; cw[j] = 1.0 / (1.0 + 0.2 * j); }
    std::vector<char> act(NB, 1);
    int fails = 0;
    for (int cs = 0; cs < 2; cs++) {
        // M(z) = A0 + z A1 + z^2 A2 with roots inside |z| < R: A0 = -z_i A1 on chosen directions (A0 = -A1 Z, Z = diag(z_1..))
        Vec A1 = rnd(size_t(NB) * NB), A2 = rnd(size_t(NB) * NB, 0.1);
        for (int i = 0; i < NB; i++) A1[size_t(i) * NB + i] += 3.0;
        std::vector<cd> zr(NB);
        for (int i = 0; i < NB; i++) zr[i] = cd(1.5 + 0.3 * i, 0.2);        // outside the contour
        zr[0] = cd(0.05, 0.0);
        if (cs == 1) zr[1] = cd(-0.12, 0.0);
        Vec A0(size_t(NB) * NB, 0.0);
        for (int i = 0; i < NB; i++)
            for (int j = 0; j < NB; j++) A0[size_t(i) * NB + j] = -A1[size_t(i) * NB + j] * zr[j];
        Vec Ms(3 * size_t(NB) * NB);
        for (size_t i = 0; i < size_t(NB) * NB; i++) { Ms[i] = A0[i]; Ms[size_t(NB) * NB + i] = A1[i]; Ms[2 * size_t(NB) * NB + i] = A2[i]; }
        Vec Sn = rnd(size_t(N) * NB, 0.1);
        std::vector<Vec> dMs(ND), dSn(ND);
        for (int q = 0; q < ND; q++) { dMs[q] = rnd(Ms.size(), 0.05); dSn[q] = rnd(Sn.size(), 0.1); }
        ct::Contour o;
        ct::contour(Ms, NB, rw, cw, e1, R, act, Sn, -1, o);
        cdl::ContourT t;
        cdl::contour_t(Ms, dMs, NB, e1, R, Sn, dSn, o, ND, t);
        double w[4][2] = {{0}};
        const double hs[2] = {1e-5, 1e-6};
        for (int q = 0; q < ND; q++)
            for (int ih = 0; ih < 2; ih++) {
                double h = hs[ih];
                ct::Contour op, om;
                ct::contour(axpy(Ms, h, dMs[q]), NB, rw, cw, e1, R, act, axpy(Sn, h, dSn[q]), o.rk, op);
                ct::contour(axpy(Ms, -h, dMs[q]), NB, rw, cw, e1, R, act, axpy(Sn, -h, dSn[q]), o.rk, om);
                auto fd = [&](const Vec& a, const Vec& b) { Vec r = axpy(a, -1.0, b); for (auto& z : r) z /= 2 * h; return r; };
                double e0 = 0;
                for (int k = 0; k < N; k++) e0 = std::max(e0, rel(t.dMinv[k][q], fd(op.Minv[k], om.Minv[k])));
                w[0][ih] = std::max(w[0][ih], e0);
                w[1][ih] = std::max(w[1][ih], rel(t.dX0[q], fd(op.X0, om.X0)));
                w[2][ih] = std::max(w[2][ih], rel(t.dRes[q], fd(op.Res, om.Res)));
                // the span: dP from the dual (P(W + h dW) - P(W - h dW)) / 2h against the FD of P(Wq)
                std::vector<Vec> Wp(o.rk), Wm(o.rk);
                for (int r = 0; r < o.rk; r++) { Wp[r] = axpy(o.Wq[r], h, t.dWq[q][r]); Wm[r] = axpy(o.Wq[r], -h, t.dWq[q][r]); }
                w[3][ih] = std::max(w[3][ih], rel(fd(proj(Wp), proj(Wm)), fd(proj(op.Wq), proj(om.Wq))));
            }
        const char* nm[4] = {"dMinv (every node)", "dX0", "dRes", "span{Wq} (projector)"};
        for (int i = 0; i < 4; i++) {
            bool ok = std::min(w[i][0], w[i][1]) < 1e-7;
            if (!ok) fails++;
            std::printf("  case %c (rank %d) %-24s rel. error  h 1e-5: %.2e   h 1e-6: %.2e   %s\n", cs ? 'B' : 'A', o.rk, nm[i], w[i][0], w[i][1], ok ? "ok" : "FAIL");
        }
    }
    std::printf("%s\n", fails ? "FAILURES" : "all ok");
    return fails ? 1 : 0;
}

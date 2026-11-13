// Unit test of mk_steps_dual.hpp (research rounds 1074 / 1089, the forward-mode replay's stage 1(b)): the dual defect correction against
// central differences of mk_steps' OWN defect_int / defect_off (the production step), every input moved along two random directions at
// once -- the start X0, the contour inverses Minv, the residue directions Wq, the rows' data and (integer centres) the data fix.  The rows are
// a synthetic nonlinear map of the step's moments (the real ones come from the kernel's tape; the step's arithmetic is what is tested).
#include "mk_steps_dual.hpp"
#include <cstdio>
#include <random>

using namespace mk;
using sd::Tv;
using sd::Vec;
using cd = std::complex<double>;

static std::mt19937_64 rng(1089);
static Vec rnd(size_t n, double a = 1.0) {
    std::normal_distribution<double> g(0.0, 1.0);
    Vec v(n);
    for (auto& z : v) z = a * cd(g(rng), g(rng));
    return v;
}
static Vec axpy(const Vec& x, double h, const Vec& d) { Vec o(x.size()); for (size_t i = 0; i < x.size(); i++) o[i] = x[i] + h * d[i]; return o; }
static double nrm(const Vec& v) { double s = 0; for (auto& z : v) s += std::norm(z); return std::sqrt(s); }

int main() {
    std::printf("mk_steps_dual unit test (research rounds 1074 / 1089, stage 1(b)): the dual step vs central differences of mk_steps\n");
    const int K = 3, NB = 4, N = 8, ND = 2, KN = K * NB;
    const double R = 0.05;
    Vec e1(N);
    for (int k = 0; k < N; k++) e1[k] = std::polar(1.0, 2 * M_PI * (k + 0.5) / N);
    // the inputs and their two directions
    Vec X0 = rnd(size_t(N) * NB, 1e-3), A = rnd(size_t(KN) * KN, 0.05), c = rnd(KN, 1e-3);
    for (int i = 0; i < KN; i++) A[size_t(i) * KN + i] += 1.0;
    std::vector<Vec> Minv(N);
    for (int k = 0; k < N; k++) {
        Minv[k] = rnd(size_t(NB) * NB, 0.02);
        for (int i = 0; i < NB; i++) Minv[k][size_t(i) * NB + i] += 0.6;
    }
    std::vector<Vec> Wq = {rnd(size_t(N) * NB, 1.0)};
    std::vector<Vec> dX0(ND), dA(ND), dc(ND);
    std::vector<std::vector<Vec>> dMin(ND), dW(ND);
    for (int q = 0; q < ND; q++) {
        dX0[q] = rnd(X0.size(), 1e-3); dA[q] = rnd(A.size(), 0.05); dc[q] = rnd(c.size(), 1e-3);
        for (int k = 0; k < N; k++) dMin[q].push_back(rnd(size_t(NB) * NB, 0.05));
        dW[q] = {rnd(Wq[0].size(), 0.3)};
    }
    // the rows: A mom + c + 0.3 mom * mom (elementwise), mom = moments_of(X); h moves the data along direction q
    auto rows_at = [&](const Vec& X, int q, double h) {
        Vec mom = ct::moments_of(X, K, NB, e1, R), Ah = q < 0 ? A : axpy(A, h, dA[q]), ch = q < 0 ? c : axpy(c, h, dc[q]), r(KN);
        for (int i = 0; i < KN; i++) {
            cd s = ch[i] + 0.3 * mom[i] * mom[i];
            for (int j = 0; j < KN; j++) s += Ah[size_t(i) * KN + j] * mom[j];
            r[i] = s;
        }
        return r;
    };
    sd::RowsT rowsT = [&](const Vec& X, const Tv& dX, Tv& dR) {
        Vec mom = ct::moments_of(X, K, NB, e1, R);
        dR.assign(ND, Vec(KN));
        for (int q = 0; q < ND; q++) {
            Vec dm = ct::moments_of(dX[q], K, NB, e1, R);
            for (int i = 0; i < KN; i++) {
                cd s = dc[q][i] + 0.6 * mom[i] * dm[i];
                for (int j = 0; j < KN; j++) s += A[size_t(i) * KN + j] * dm[j] + dA[q][size_t(i) * KN + j] * mom[j];
                dR[q][i] = s;
            }
        }
        return rows_at(X, -1, 0.0);
    };
    // the data fix: X + 0.2 X^2 (elementwise) and its tangent
    auto fix = [](const Vec& X) { Vec o(X.size()); for (size_t i = 0; i < X.size(); i++) o[i] = X[i] + 0.2 * X[i] * X[i]; return o; };
    sd::FixT fixT = [&](const Vec& X, const Tv& dX, Tv& dO) {
        dO.assign(dX.size(), Vec(X.size()));
        for (size_t q = 0; q < dX.size(); q++)
            for (size_t i = 0; i < X.size(); i++) dO[q][i] = (1.0 + 0.4 * X[i]) * dX[q][i];
        return fix(X);
    };
    std::vector<ad::Tans> dMinv(N, ad::Tans(ND));
    for (int k = 0; k < N; k++) for (int q = 0; q < ND; q++) dMinv[k][q] = dMin[q][k];
    std::vector<Tv> dWq(ND, Tv{Vec()});
    for (int q = 0; q < ND; q++) dWq[q][0] = dW[q][0];
    std::vector<char> act(NB, 1);
    act[NB - 1] = 0;                                              // one inactive column (masked rows)
    int fails = 0;
    for (int variant = 0; variant < 3; variant++) {               // integer without / with the data fix; off-integer
        const bool off = variant == 2, withfix = variant == 1;
        const int ndc = 4;
        sd::DefectT dt;
        if (off) sd::defect_off_t(X0, dX0, Minv, dMinv, Wq, dWq, K, NB, e1, R, ndc, false, rowsT, ND, dt);
        else sd::defect_int_t(X0, dX0, Minv, dMinv, Wq, dWq, act, 1.0, K, NB, e1, R, ndc, false, withfix ? &fixT : nullptr, rowsT, ND, dt);
        // the production step at the base and moved
        auto prod = [&](int q, double h) {
            Vec X0h = q < 0 ? X0 : axpy(X0, h, dX0[q]);
            std::vector<Vec> Mh = Minv, Wh = Wq;
            if (q >= 0) { for (int k = 0; k < N; k++) Mh[k] = axpy(Minv[k], h, dMin[q][k]); Wh[0] = axpy(Wq[0], h, dW[q][0]); }
            st::Rows rw;
            rw.gal = [&, q, h](const Vec& X) { return rows_at(X, q, h); };
            rw.stored = [&, q, h](const Vec& X) { return rows_at(X, q, h); };
            st::DefectOut o;
            std::function<Vec(const Vec&)> df = withfix ? std::function<Vec(const Vec&)>(fix) : std::function<Vec(const Vec&)>();
            if (off) st::defect_off(X0h, Mh, Wh, K, NB, e1, R, ndc, false, rw, o);
            else st::defect_int(X0h, Mh, Wh, act, 1.0, K, NB, e1, R, ndc, false, df, rw, o);
            return o.X;
        };
        Vec base = prod(-1, 0.0);
        double vrel = nrm(axpy(dt.X, -1.0, base)) / nrm(base), worst[2] = {0, 0};
        const double hs[2] = {1e-5, 1e-6};
        for (int q = 0; q < ND; q++)
            for (int ih = 0; ih < 2; ih++) {
                Vec f = axpy(prod(q, hs[ih]), -1.0, prod(q, -hs[ih]));
                for (auto& z : f) z /= 2 * hs[ih];
                worst[ih] = std::max(worst[ih], nrm(axpy(dt.dX[q], -1.0, f)) / nrm(f));
            }
        bool ok = std::min(worst[0], worst[1]) < 1e-7 && vrel < 1e-13;
        if (!ok) fails++;
        std::printf("  %-30s corrections %d: rel. error  h 1e-5: %.2e   h 1e-6: %.2e   (value vs mk_steps: %.1e)   %s\n",
                    off ? "off-integer (defect_off)" : withfix ? "integer + data fix" : "integer (defect_int)", dt.corrections, worst[0], worst[1],
                    vrel, ok ? "ok" : "FAIL");
    }
    std::printf("%s\n", fails ? "FAILURES" : "all ok");
    return fails ? 1 : 0;
}

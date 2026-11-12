// Unit tests of mk_ad.hpp's derivative rules (research round 1074): each rule against central finite differences of the primitive itself,
// along random complex directions, in isolation.  Prints the relative error per rule (|rule - FD| / |FD|) for two step sizes (the FD's own
// truncation shows as the h -> h/10 change).  Exit 1 if any rule misses by more than its tolerance.
#include "mk_ad.hpp"
#include <cstdio>
#include <functional>
#include <random>

using namespace mk;
using ad::Mat;
using ad::Tans;
using cd = std::complex<double>;

static std::mt19937_64 rng(1074);
static Mat rnd(size_t n) {
    std::normal_distribution<double> g(0.0, 1.0);
    Mat m(n);
    for (auto& x : m) x = cd(g(rng), g(rng));
    return m;
}
static double nrm(const Mat& a) { double s = 0; for (auto& x : a) s += std::norm(x); return std::sqrt(s); }
static Mat diff(const Mat& a, const Mat& b, double sc) { Mat r(a.size()); for (size_t i = 0; i < a.size(); i++) r[i] = (a[i] - b[i]) * sc; return r; }
static Mat plus(const Mat& a, const Mat& d, double h) { Mat r(a.size()); for (size_t i = 0; i < a.size(); i++) r[i] = a[i] + h * d[i]; return r; }
static double relerr(const Mat& rule, const Mat& fd) { return nrm(diff(rule, fd, 1.0)) / std::max(nrm(fd), 1e-300); }

static int fails = 0;
static void report(const char* name, double e1, double e2, double tol) {
    bool ok = std::min(e1, e2) < tol;
    if (!ok) fails++;
    std::printf("  %-44s rel. error  h: %.2e   h/10: %.2e   %s\n", name, e1, e2, ok ? "ok" : "FAIL");
}

// FD of a matrix-valued f along direction t, central, step h (relative to |x|)
static Mat fd(const std::function<Mat(double)>& f, double h) { return diff(f(h), f(-h), 0.5 / h); }

int main() {
    std::printf("mk_ad unit tests (research round 1074): rule vs central differences\n");
    const double H = 1e-5;
    // inv
    {
        int n = 7; Mat A = rnd(n * n); for (int i = 0; i < n; i++) A[i * n + i] += 4.0;
        Mat dA = rnd(n * n); Tans dAi;
        ad::inv_t(A, n, {dA}, dAi);
        auto f = [&](double h) { return la::inv(plus(A, dA, h), n); };
        report("inv", relerr(dAi[0], fd(f, H)), relerr(dAi[0], fd(f, H / 10)), 1e-7);
    }
    // solve (dA and dB, multiple rhs)
    {
        int n = 9, r = 3; Mat A = rnd(n * n), B = rnd(n * r), dA = rnd(n * n), dB = rnd(n * r); Tans dX;
        ad::solve_t(A, n, B, r, {dA, Mat()}, {dB, dB}, dX);
        auto f0 = [&](double h) { return la::solve(plus(A, dA, h), n, plus(B, dB, h), r); };
        auto f1 = [&](double h) { return la::solve(A, n, plus(B, dB, h), r); };
        report("solve (dA + dB)", relerr(dX[0], fd(f0, H)), relerr(dX[0], fd(f0, H / 10)), 1e-7);
        report("solve (dB only)", relerr(dX[1], fd(f1, H)), relerr(dX[1], fd(f1, H / 10)), 1e-7);
    }
    // lstsq: overdetermined full rank, underdetermined full row rank, rank deficient with a gap (rank frozen)
    for (int cs = 0; cs < 3; cs++) {
        int m = cs == 1 ? 6 : 12, n = cs == 1 ? 10 : 7, r = 2;
        Mat A = rnd(size_t(m) * n);
        int rk_imp = -1;
        if (cs == 2) {                                            // A = U diag(s) V^H with s = (10 .. 1, 1e-9, 1e-11): rank 5 imposed
            std::vector<double> s; Mat U, Vh;
            la::svd(A, m, n, s, &U, &Vh, false);
            int k = std::min(m, n);
            for (int q = 0; q < k; q++) s[q] = q < 5 ? 10.0 - q : 1e-9 * std::pow(1e-2, q - 5);
            Mat A2(size_t(m) * n, 0.0);
            for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) for (int q = 0; q < k; q++) A2[size_t(i) * n + j] += U[size_t(i) * k + q] * s[q] * Vh[size_t(q) * n + j];
            A = A2; rk_imp = 5;
        }
        Mat B = rnd(size_t(m) * r), dA = rnd(size_t(m) * n), dB = rnd(size_t(m) * r); Tans dX;
        ad::lstsq_t(A, m, n, B, r, {dA}, {dB}, dX, rk_imp);
        auto f = [&](double h) {
            Mat Ah = plus(A, dA, h), Bh = plus(B, dB, h);
            if (rk_imp < 0) return la::lstsq(Ah, m, n, Bh, r);
            int rr = rk_imp; Mat P = ad::pinv_r(Ah, m, n, rr);
            return ad::mul(P, Bh, n, m, r);
        };
        const char* nm[] = {"lstsq (overdetermined, full rank)", "lstsq (underdetermined, full row rank)", "lstsq (rank 5 of 7 frozen, gap 1e9)"};
        report(nm[cs], relerr(dX[0], fd(f, H)), relerr(dX[0], fd(f, H / 10)), cs == 2 ? 1e-6 : 1e-7);
    }
    // singular values
    {
        int m = 9, n = 6; Mat A = rnd(size_t(m) * n), dA = rnd(size_t(m) * n);
        std::vector<std::vector<double>> ds;
        ad::svals_t(A, m, n, {dA}, ds);
        auto f = [&](double h) { std::vector<double> s; la::svd(plus(A, dA, h), m, n, s); Mat r(s.size()); for (size_t i = 0; i < s.size(); i++) r[i] = s[i]; return r; };
        Mat rule(ds[0].size()); for (size_t i = 0; i < rule.size(); i++) rule[i] = ds[0][i];
        report("singular values", relerr(rule, fd(f, H)), relerr(rule, fd(f, H / 10)), 1e-7);
    }
    // null vector of a nearly singular square matrix (smallest singular value 1e-8), and of a tall one
    for (int cs = 0; cs < 2; cs++) {
        int m = cs ? 10 : 7, n = 7;
        Mat A = rnd(size_t(m) * n);
        { std::vector<double> s; Mat U, Vh; la::svd(A, m, n, s, &U, &Vh, false); s[n - 1] = 1e-8; Mat A2(size_t(m) * n, 0.0);
          for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) for (int q = 0; q < n; q++) A2[size_t(i) * n + j] += U[size_t(i) * n + q] * s[q] * Vh[size_t(q) * n + j];
          A = A2; }
        Mat dA = rnd(size_t(m) * n); Tans dv;
        Mat v = ad::nullvec_t(A, m, n, {dA}, dv);
        int p = 0; for (int j = 1; j < n; j++) if (std::abs(v[j]) > std::abs(v[p])) p = j;
        auto f = [&](double h) {     // the same convention at the frozen entry p: v_p real positive, unit norm
            std::vector<double> s; Mat U, Vh; la::svd(plus(A, dA, h), m, n, s, &U, &Vh, true);
            Mat w(n); for (int j = 0; j < n; j++) w[j] = std::conj(Vh[size_t(n - 1) * n + j]);
            cd ph = std::abs(w[p]) / w[p]; for (auto& x : w) x *= ph; return w; };
        report(cs ? "null vector (tall 10 x 7)" : "null vector (square, s_min 1e-8)", relerr(dv[0], fd(f, H)), relerr(dv[0], fd(f, H / 10)), 1e-6);
    }
    // eigenvalues (matched to the base by nearest)
    {
        int n = 8; Mat A = rnd(size_t(n) * n), dA = rnd(size_t(n) * n); Tans dl;
        std::vector<cd> w = ad::eigvals_t(A, n, {dA}, dl);
        auto f = [&](double h) { std::vector<cd> e = la::eigvals(plus(A, dA, h), n); Mat o(n);
            for (int i = 0; i < n; i++) { int b = 0; for (int j = 1; j < n; j++) if (std::abs(e[j] - w[i]) < std::abs(e[b] - w[i])) b = j; o[i] = e[b]; } return o; };
        report("eigenvalues", relerr(dl[0], fd(f, H)), relerr(dl[0], fd(f, H / 10)), 1e-7);
    }
    // generalized eigenvalues (dA and dB)
    {
        int n = 8; Mat A = rnd(size_t(n) * n), B = rnd(size_t(n) * n), dA = rnd(size_t(n) * n), dB = rnd(size_t(n) * n); Tans dl;
        std::vector<cd> w = ad::eiggen_t(A, B, n, {dA}, {dB}, dl);
        auto f = [&](double h) { std::vector<cd> e = la::eiggen(plus(A, dA, h), plus(B, dB, h), n); Mat o(n);
            for (int i = 0; i < n; i++) { int b = 0; for (int j = 1; j < n; j++) if (std::abs(e[j] - w[i]) < std::abs(e[b] - w[i])) b = j; o[i] = e[b]; } return o; };
        report("generalized eigenvalues (dA + dB)", relerr(dl[0], fd(f, H)), relerr(dl[0], fd(f, H / 10)), 1e-7);
    }
    std::printf("%s: %d rule(s) failed\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}

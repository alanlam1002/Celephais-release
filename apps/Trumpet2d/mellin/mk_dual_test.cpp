// Unit tests of mk_dual.hpp (research rounds 1074 / 1089, the forward-mode replay's stage 1): every dual-series rule against central finite
// differences of the same mk_series operations, along two random directions at once (ND = 2), in the exact-merge mode (production's).  The
// FD's decisions are the value's (the steps are small: the keys and the validity do not move).  Prints the relative error per rule for two
// step sizes; exit 1 if any rule misses by more than its tolerance.
#include "mk_dual.hpp"
#include <cstdio>
#include <functional>
#include <map>
#include <random>

using namespace mk;

static std::mt19937_64 rng(1089);
static double gauss() { static std::normal_distribution<double> g(0.0, 1.0); return g(rng); }

static Ctx C;
static Piece rpiece(double amp, int B = 1) {
    Piece p = C.zeros(B);
    for (auto& z : p.a) z = amp * cd(gauss(), gauss());
    return p;
}
// a random series r^e sum over keys, moments damped (as a converged series' are)
static Ser rser(const std::vector<double>& keys, int e, double top, double amp) {
    std::vector<std::pair<Key, Piece>> d;
    for (double g : keys) {
        Piece p = rpiece(amp);
        for (int m = 0; m < C.K; m++)
            for (int q = 0; q < C.NF; q++) p.a[size_t(m) * C.NF + q] *= std::pow(0.2, m);
        d.emplace_back(Key{g, std::abs(g - std::nearbyint(g)) < 1e-12}, std::move(p));
    }
    return C.make(std::move(d), true, e, top);
}
static Ser axpy(const Ser& a, double h, const Ser& b) { return C.add(a, C.scale(b, h)); }
// batch copy k of a tangent series as a batch-1 series
static Ser copyk(const Ser& t, int k) {
    std::vector<std::pair<Key, Piece>> d;
    for (auto& kv : t.d) {
        Piece p = C.zeros(1);
        std::copy(kv.second.a.begin() + size_t(k) * C.psz(), kv.second.a.begin() + size_t(k + 1) * C.psz(), p.a.begin());
        d.emplace_back(kv.first, std::move(p));
    }
    Ser s;
    s.d = std::move(d); s.has_e = t.has_e; s.e = t.e; s.top = t.top;
    return s;
}
// || a - b || / || b || over the union of keys
static double rel(const Ser& a, const Ser& b) {
    std::map<double, std::vector<cd>> A, Bm;
    for (auto& kv : a.d) A[kv.first.v] = kv.second.a;
    for (auto& kv : b.d) Bm[kv.first.v] = kv.second.a;
    double num = 0, den = 0;
    for (auto& kv : Bm) {
        auto it = A.find(kv.first);
        for (size_t i = 0; i < kv.second.size(); i++) {
            cd x = it == A.end() ? cd(0.0) : it->second[i];
            num += std::norm(x - kv.second[i]);
            den += std::norm(kv.second[i]);
        }
    }
    for (auto& kv : A)
        if (!Bm.count(kv.first))
            for (auto& x : kv.second) num += std::norm(x);
    return std::sqrt(num) / std::max(std::sqrt(den), 1e-300);
}

static int fails = 0;
// f on dual inputs vs central FD of f on values along each direction k: x_i(h) = x_i + h dx_i[k]
static void check(const char* name, const std::vector<Ser>& x, const std::vector<std::vector<Ser>>& dx,
                  const std::function<DSer(Dual&, const std::vector<DSer>&)>& fd_, const std::function<Ser(const std::vector<Ser>&)>& fv,
                  double tol = 1e-7) {
    const int ND = 2;
    Dual D(C, ND);
    std::vector<DSer> xd;
    for (size_t i = 0; i < x.size(); i++) {
        DSer s = D.seed(x[i], 0, dx[i][0]);
        DSer s1 = D.seed(x[i], 1, dx[i][1]);
        s.t = D.tadd(s.t, s1.t);
        xd.push_back(s);
    }
    DSer out = fd_(D, xd);
    double worst[2] = {0, 0};
    const double hs[2] = {1e-5, 1e-6};
    for (int k = 0; k < ND; k++)
        for (int ih = 0; ih < 2; ih++) {
            double h = hs[ih];
            std::vector<Ser> xp, xm;
            for (size_t i = 0; i < x.size(); i++) { xp.push_back(axpy(x[i], h, dx[i][k])); xm.push_back(axpy(x[i], -h, dx[i][k])); }
            Ser fdk = C.scale(C.sub(fv(xp), fv(xm)), 0.5 / h);
            worst[ih] = std::max(worst[ih], rel(copyk(out.t, k), fdk));
        }
    double vr = rel(out.v, fv(x));
    bool ok = std::min(worst[0], worst[1]) < tol && vr < 1e-14;
    if (!ok) fails++;
    std::printf("  %-34s rel. error  h 1e-5: %.2e   h 1e-6: %.2e   (value vs plain: %.1e)   %s\n", name, worst[0], worst[1], vr, ok ? "ok" : "FAIL");
}

int main() {
    std::printf("mk_dual unit tests (research rounds 1074 / 1089, stage 1): dual-series rules vs central differences\n");
    const int K = 5, NF = 12;
    std::vector<cd> Dm(size_t(NF) * NF);
    for (auto& z : Dm) z = cd(gauss(), 0.0) * 0.3;
    C.init(K, NF, 1.389005375683986, 0.0, Dm);
    C.exact = true;
    const double top = 5.0;
    // grade-0-led series (e = 0) for exp / log / inv; keys on and off the lattice
    std::vector<double> ks = {0.0, 1.0, 2.0, 2.370639771, 3.0, 3.370639771, 4.0, 4.741279542};
    Ser a = rser(ks, 0, top, 0.3), b = rser(ks, 0, top, 0.3);
    // a's grade-0 moment 0 offset away from zero (inv / log need a nonvanishing leading coefficient)
    for (int q = 0; q < NF; q++) a.d[0].second.a[q] += 2.0;
    auto dir = [&](const Ser& s) { return std::vector<Ser>{rser(ks, s.e, top, 1.0), rser(ks, s.e, top, 1.0)}; };
    std::vector<std::vector<Ser>> da = {dir(a)}, dab = {dir(a), dir(b)};
    for (auto& d : da[0]) for (int q = 0; q < NF; q++) d.d[0].second.a[q] *= 0.1;
    dab[0] = da[0];
    Ser rb = rser({1.0, 2.370639771, 3.0}, 1, top, 0.5);           // an r-power-1 series (drad's e)
    std::vector<std::vector<Ser>> drb = {dir(rb)};

    check("add", {a, b}, dab, [](Dual& D, auto& x) { return D.add(x[0], x[1]); }, [](auto& x) { return C.add(x[0], x[1]); });
    check("scale", {a}, da, [](Dual& D, auto& x) { return D.scale(x[0], cd(0.7, -0.2)); }, [](auto& x) { return C.scale(x[0], cd(0.7, -0.2)); });
    check("mul", {a, b}, dab, [](Dual& D, auto& x) { return D.mul(x[0], x[1]); }, [](auto& x) { return C.mul(x[0], x[1]); });
    check("drad", {rb}, drb, [](Dual& D, auto& x) { return D.drad(x[0]); }, [](auto& x) { return C.drad(x[0]); });
    check("dth", {a}, da, [](Dual& D, auto& x) { return D.dth(x[0]); }, [](auto& x) { return C.dth(x[0]); });
    check("inv", {a}, da, [](Dual& D, auto& x) { return D.inv(x[0]); }, [](auto& x) { return C.inv(x[0]); });
    check("expser", {b}, {dab[1]}, [](Dual& D, auto& x) { return D.expser(x[0]); }, [](auto& x) { return C.expser(x[0]); });
    check("logser", {a}, da, [](Dual& D, auto& x) { return D.logser(x[0]); }, [](auto& x) { return C.logser(x[0]); });
    check("trunc", {a}, da, [](Dual& D, auto& x) { return D.trunc(x[0], 3.2); }, [](auto& x) { return C.trunc(x[0], 3.2); });
    // a composite as the engine's expressions are: exp(b) / a + log(a b) * drad(dth(a) r)
    check("composite", {a, b}, dab,
          [&](Dual& D, auto& x) {
              DSer r1 = D.mul(D.expser(x[1]), D.inv(x[0]));
              DSer r2 = D.logser(D.mul(x[0], x[1]));
              DSer r3 = D.drad(D.mul(D.dth(x[0]), D.constant(rb)));
              return D.add(D.mul(r1, r2), D.scale(r3, 1.0));
          },
          [&](auto& x) {
              Ser r1 = C.mul(C.expser(x[1]), C.inv(x[0]));
              Ser r2 = C.logser(C.mul(x[0], x[1]));
              Ser r3 = C.drad(C.mul(C.dth(x[0]), rb));
              return C.add(C.mul(r1, r2), r3);
          });
    // key_tangent: a piece at a moving centre g(p) = g0 + p dg, against FD of the binomial shift (the piece re-expanded about g0)
    {
        Dual D(C, 2);
        Piece v = rpiece(1.0);
        std::vector<double> dg = {0.37, -1.2};
        Piece t = D.key_tangent(v, dg);
        double worst = 0;
        for (int k = 0; k < 2; k++) {
            const double h = 1e-6;
            Piece pp = C.shift(v, h * dg[k]), pm = C.shift(v, -h * dg[k]);
            double num = 0, den = 0;
            for (size_t i = 0; i < v.a.size(); i++) {
                cd f = (pp.a[i] - pm.a[i]) / (2 * h);
                num += std::norm(t.a[size_t(k) * C.psz() + i] - f);
                den += std::norm(f);
            }
            worst = std::max(worst, std::sqrt(num / den));
        }
        bool ok = worst < 1e-8;
        if (!ok) fails++;
        std::printf("  %-34s rel. error  h 1e-6: %.2e   %s\n", "key_tangent (moving centre)", worst, ok ? "ok" : "FAIL");
    }
    std::printf("%s\n", fails ? "FAILURES" : "all rules ok");
    return fails ? 1 : 0;
}

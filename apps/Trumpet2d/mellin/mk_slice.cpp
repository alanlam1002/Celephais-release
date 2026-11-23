// Mellin throat recursion (code round 463; research rounds 1187-1189, S2b part B): the l = 2 slicing flow applied to the engine's bare forward --
// its own translation unit, linked after mk_kernel.cpp (mk_slice_api.hpp; the generator: mk_slice.hpp, the port of slice_s1.build; the flow: the
// port of scripts/s2b_flow.py, checked against it to round-off).
#include "mk_slice_api.hpp"
#include "mk_slice.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>

namespace mk {
namespace sl {
namespace {
std::string jnum(double x) {
    char b[40];
    std::snprintf(b, sizeof b, "%.17g", x);
    return b;
}
void npyw(const std::string& path, const void* data, const std::string& descr, const std::vector<long>& shape, size_t item) {
    std::ostringstream h;
    h << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); i++) h << shape[i] << (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
    h << "), }";
    std::string hs = h.str();
    size_t tot = 10 + hs.size() + 1;
    hs += std::string((64 - tot % 64) % 64, ' ') + "\n";
    std::ofstream f(path, std::ios::binary);
    f.write("\x93NUMPY\x01\x00", 8);
    unsigned short hl = static_cast<unsigned short>(hs.size());
    f.write(reinterpret_cast<const char*>(&hl), 2);
    f.write(hs.data(), long(hs.size()));
    size_t n = 1;
    for (long s : shape) n *= size_t(s);
    f.write(static_cast<const char*>(data), long(n * item));
}
}  // namespace

void flow_apply(Kernel& k_, Ser F[6], const FlowIn& in) {
    Ctx& c = k_.c;
    const std::vector<double>& TH = *in.TH;
    const int NF = c.NF;
    // the generator's top grade: MK_S2B_TOP, default min(the fields' top, 8) -- below e2^4's 4 sigma_2 = 8.19, which the third order does not supply
    const double T = std::getenv("MK_S2B_TOP") ? std::atof(std::getenv("MK_S2B_TOP")) : std::min(F[0].top, 8.0);
    const double eps = std::getenv("MK_S2B_EPS") ? std::atof(std::getenv("MK_S2B_EPS")) : 5e-4;
    auto t0 = std::chrono::steady_clock::now();
    for (int f = 0; f < 6; f++) F[f] = c.trunc(F[f], f == 1 ? T + 1 : T);
    std::vector<double> u(NF / 4, 0.0), P(NF / 4, 0.0);              // the cylinder's cos 2j theta modes (as SIGMA2_'s)
    for (int j = 0; j < NF / 4; j++)
        for (int q = 0; q < NF; q++) {
            const double w = (j ? 2.0 : 1.0) / NF * std::cos(2.0 * j * TH[q]);
            u[j] += w * k_.cyl.U[q];
            P[j] += w * k_.cyl.P[q];
        }
    if (!in.quiet) {                                                // the bare (gauge-flowed, truncated) fields: OUT/fields_bare_<f>
        static const char* FN[6] = {"psi2", "Phb", "qf", "br", "bt", "Qb"};
        for (int f = 0; f < 6; f++) {
            std::vector<double> ks;
            std::vector<int> ii;
            std::vector<cd> pc;
            for (auto& kv : F[f].d) { ks.push_back(kv.first.v); ii.push_back(kv.first.isint); pc.insert(pc.end(), kv.second.a.begin(), kv.second.a.end()); }
            const std::string b = in.out + "/fields_bare_" + FN[f];
            npyw(b + "_key.npy", ks.data(), "<f8", {long(ks.size())}, 8);
            npyw(b + "_isint.npy", ii.data(), "<i4", {long(ii.size())}, 4);
            npyw(b + "_pieces.npy", pc.data(), "<c16", {long(ks.size()), c.K, NF}, 16);
            std::ofstream(b + "_meta.txt") << (F[f].has_e ? std::to_string(F[f].e) : "None") << " " << jnum(F[f].top) << "\n";
        }
    }
    const sl::Modes M = sl::sl_modes(u, P);
    const double Ph = k_.cyl.v_Ph_r.empty() ? 1.0 : k_.cyl.v_Ph_r[0].real();
    Ser G[6];
    sl::Rep rep;
    sl::build(c, F, TH, M, Ph, in.J, T, 2, G, rep);
    // G's PH2 coefficient at sigma_2: Phibar's piece / (K P(theta)), least squares on the PH columns' bases
    const int fPH = Kernel::fam("PH");
    const int ncp = k_.cols.ncol[fPH];
    const Key ks = c.xkey(rep.sigma);
    const int ip = G[1].find(ks);
    if (ip < 0) throw std::runtime_error("mk: MK_S2B: G has no Phibar piece at sigma_2");
    std::vector<cd> A(size_t(NF) * ncp), b(NF);
    for (int q = 0; q < NF; q++) {
        b[q] = G[1].d[ip].second.a[q] / (Ph * k_.cyl.P[q]);
        for (int j = 0; j < ncp; j++) A[size_t(q) * ncp + j] = k_.cols.B[fPH][size_t(j) * NF + q];
    }
    auto gco = la::lstsq_rc(A, NF, ncp, b, 1, std::numeric_limits<double>::epsilon() * NF);
    const int jph = in.jph;
    const cd a1 = in.a1;
    const double a = (a1 / gco[jph]).real();
    auto addF = [&](const Ser* A_, const Ser* B_, double cf, Ser* out) {
        for (int i = 0; i < 6; i++) { Ser sb = c.scale(B_[i], cd(cf, 0.0)); out[i] = c.add(A_[i], sb); }
    };
    Ser Fp[6], Fm_[6], Gp[6], Gm[6], DG[6];
    addF(F, G, eps * a, Fp);
    addF(F, G, -eps * a, Fm_);
    sl::Rep rp, rm;
    sl::build(c, Fp, TH, M, Ph, in.J, T, 2, Gp, rp);
    sl::build(c, Fm_, TH, M, Ph, in.J, T, 2, Gm, rm);
    for (int i = 0; i < 6; i++) { Ser d = c.sub(Gp[i], Gm[i]); DG[i] = c.scale(d, cd(1.0 / (2 * eps * a), 0.0)); }
    // round 1189: the THIRD order (a^3 / 6) D(DG.G).G -- along the flow dF/da = G[F], d^2F/da^2 = H[F] := DG[F].G[F], d^3F/da^3 = DH[F].G[F] --
    // by central differences of H (itself at eps) along G at 2 eps (scripts/s2b_flow.py --order 3); MK_S2B_ORDER (default 3 when the top
    // grade reaches e2^3's 3 sigma_2 = 6.14, else 2)
    const int order = std::getenv("MK_S2B_ORDER") ? std::atoi(std::getenv("MK_S2B_ORDER")) : (T >= 3 * rep.sigma ? 3 : 2);
    Ser D3[6];
    double d3max = 0.0;
    if (order >= 3) {
        auto Hof = [&](const Ser* Fx, Ser* Hx) {
            Ser Gx[6], Fa[6], Fb[6], Ga[6], Gb[6];
            sl::Rep r0, r1, r2;
            sl::build(c, Fx, TH, M, Ph, in.J, T, 2, Gx, r0);
            addF(Fx, Gx, eps * a, Fa);
            addF(Fx, Gx, -eps * a, Fb);
            sl::build(c, Fa, TH, M, Ph, in.J, T, 2, Ga, r1);
            sl::build(c, Fb, TH, M, Ph, in.J, T, 2, Gb, r2);
            for (int i = 0; i < 6; i++) { Ser d = c.sub(Ga[i], Gb[i]); Hx[i] = c.scale(d, cd(1.0 / (2 * eps * a), 0.0)); }
        };
        Ser Fa[6], Fb[6], Ha[6], Hb[6];
        addF(F, G, 2 * eps * a, Fa);
        addF(F, G, -2 * eps * a, Fb);
        Hof(Fa, Ha);
        Hof(Fb, Hb);
        for (int i = 0; i < 6; i++) {
            Ser d = c.sub(Ha[i], Hb[i]);
            D3[i] = c.scale(d, cd(1.0 / (4 * eps * a), 0.0));
            for (auto& kv : D3[i].d)
                for (auto& z : kv.second.a) d3max = std::max(d3max, std::abs(z) * std::abs(a * a * a) / 6);
        }
    }
    Ser F1[6];
    addF(F, G, a, F1);
    if (order >= 3) {
        Ser F2[6];
        addF(F1, DG, 0.5 * a * a, F2);
        addF(F2, D3, a * a * a / 6, F);
    } else
        addF(F1, DG, 0.5 * a * a, F);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!in.quiet) {
        std::printf("  [s2b flow] sigma_2 %.12f; G at top %g (h: %d keys, residual %.1e; zeta residual %.1e); a = %+.13e (PH2 a_1 %+.13e / G_PH2 %+.13e); "
                    "DG.G at eps %.1e; order %d (third-order term's largest piece x a^3 / 6: %.1e); %.1f s\n", rep.sigma, T, rep.h_iters, rep.h_res, rep.zeta_res, a,
                    a1.real(), gco[jph].real(), eps, order, d3max, secs);
        std::ofstream(in.out + "/s2b_flow.json") << "{\"sigma2\":" << jnum(rep.sigma) << ",\"top\":" << jnum(T) << ",\"a\":" << jnum(a)
                                              << ",\"a1_PH2\":" << jnum(a1.real()) << ",\"G_PH2\":" << jnum(gco[jph].real()) << ",\"eps\":" << jnum(eps)
                                              << ",\"order\":" << order << ",\"third_max\":" << jnum(d3max) << ",\"h_residual\":" << jnum(rep.h_res) << ",\"zeta_residual\":" << jnum(rep.zeta_res) << "}\n";
    }
}

}  // namespace sl
}  // namespace mk

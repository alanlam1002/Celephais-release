/*
 * t3a_point_main.cpp -- research round 665 (code round 336), step 3a.
 *
 * A STANDALONE smoke test, part of no production binary: does Kadath's existing
 * point evaluation (System_of_eqs::add_eq_point -> Ope_point, Base_spectral::summation)
 * deliver a field's value, its r-derivative and its Jacobian column at an INTERIOR
 * point (r, theta) of a domain -- the evaluation the horizon rows at r_H need?
 *
 * Two spaces, stock Kadath domains only:
 *   STOCK  Space_polar (nucleus + two linear shells, bounds 0.3 / 1 / 2), nr 17, nt 7;
 *          points in domains 1 and 2: r = 0.7793271081 (r_H), 0.45, 1.5
 *   PROD   Space_polar_trumpet with the production layout (backbone_deep.dat: LOG shells
 *          [0.0054, 0.23643], [0.23643, 3.93890], compact), nr 33 / 33 / 17, nt 7;
 *          points r = 0.7793271081 (r_H, domain 1), 2.5 (domain 1), 0.05 (domain 0)
 * theta = 0.02 (near the axis), 0.7, 1.4.  Three unknown fields, one per theta basis the
 * bulk uses: F std_base (COS_EVEN), Q std_anti_base() (COS_ODD), B std_anti_base(1)
 * (SIN_EVEN), each a smooth analytic profile.
 *
 * Rows, each one Eq_int by add_eq_point(dom, expr, num), num = the domain's own
 * absol_to_num of (rho, z) = (r sin th, r cos th):
 *   "X" and "dr(X)" for X = F, Q, B   -> against a DIRECT Chebyshev x {cos 2j, cos(2j+1), sin 2j}
 *                                        evaluation of the same coefficients (this file's own
 *                                        sums; dr by the map's dx/dr)
 *   "exp(F) * dr(Q) * B"              -> nonlinear: its Jacobian column only
 * Jacobian: do_col_J of chosen columns (each field, coefficients (i, j) = (0, 0), (5, 1),
 * (nr-1, nt-1) in the point's domain; the column is identified from Kadath's own column
 * metadata) against a 4th-order central difference of sec_member in that coefficient
 * (h = 1e-3), and -- for the linear rows -- against the basis function's direct value.
 * (For dr in a LOG shell the reference is Kadath's own der_r of the unit coefficient, summed at the point:
 * dr = (1 / (alpha r)) d/dx there is a spectral product and differs from the exact derivative on the top
 * radial modes -- printed beside it, a property of the bulk's dr, not of the point evaluation.)
 * GATES: value and derivative 1e-12 relative (denominator: the absolute sum
 * sum |c_ij phi_ij| -- a plain relative error is printed beside it), Jacobian 1e-8.
 */

#include <mpi.h>

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "space/space_polar_trumpet.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace Kadath;

namespace {

const char* FN[3] = {"F", "Q", "B"};

// the theta basis function j of kind k (0 COS_EVEN, 1 COS_ODD, 2 SIN_EVEN), and its
// value is what Base_spectral::summation multiplies coefficient j by
double tbas(int k, int j, double th)
{
    if (k == 0) return std::cos(2.0 * j * th);
    if (k == 1) return std::cos((2.0 * j + 1.0) * th);
    return j == 0 ? 0.0 : std::sin(2.0 * j * th);
}

// smooth analytic profiles, finite at r = infinity (the compact domain)
double prof(int k, double r, double th)
{
    const double g0 = 1.0 / (1.0 + r), g1 = 1.0 / ((1.0 + r) * (1.0 + r)), g2 = 1.0 / (2.0 + r);
    if (k == 0) return 1.2 * g0 + 0.4 * g1 * std::cos(2 * th) + 0.1 * g2 * std::cos(4 * th) + 0.3;
    if (k == 1) return 0.7 * g0 * std::cos(th) + 0.2 * g2 * std::cos(3 * th);
    return 0.5 * g1 * std::sin(2 * th) + 0.15 * g0 * std::sin(4 * th);
}

void cheb(int n, double x, std::vector<double>& T, std::vector<double>& dT)
{
    T.assign(n, 0.0);
    dT.assign(n, 0.0);
    T[0] = 1.0;
    if (n > 1) { T[1] = x; dT[1] = 1.0; }
    for (int i = 2; i < n; i++) {
        T[i] = 2 * x * T[i - 1] - T[i - 2];
        dT[i] = 2 * T[i - 1] + 2 * x * dT[i - 1] - dT[i - 2];
    }
}

struct Pt { double r, th; int dom; };

struct Layout {
    std::string name;
    Space* space;
    std::vector<double> lo, hi;
    std::vector<bool> logm;
    int dmin, dmax;
    std::vector<int> nr;
    int nt;
};

// a residual evaluation on fresh fields; (pf, pd, pidx, h) perturbs one coefficient
struct Run {
    std::vector<std::unique_ptr<Scalar>> X;
    std::unique_ptr<System_of_eqs> syst;
};

std::unique_ptr<Run> build(const Layout& L, const std::vector<Pt>& pts, const std::vector<std::string>& exprs,
                           int pf = -1, int pd = -1, int pi = 0, int pj = 0, double h = 0.0)
{
    auto R = std::make_unique<Run>();
    for (int k = 0; k < 3; k++) {
        auto s = std::make_unique<Scalar>(*L.space);
        for (int d = 0; d < L.space->get_nbr_domains(); d++) {
            const Domain* dm = L.space->get_domain(d);
            Val_domain& v = s->set_domain(d);
            v.allocate_conf();
            Index idx(dm->get_nbr_points());
            do {
                const double rr = dm->get_radius()(idx);
                const double th = dm->get_coloc(2)(idx(1));
                v.set(idx) = prof(k, rr, th);
            } while (idx.inc());
        }
        if (k == 0) s->std_base();
        else if (k == 1) s->std_anti_base();
        else s->std_anti_base(1);
        if (k == pf) {
            Val_domain& v = s->set_domain(pd);
            v.coef();
            Index ix(L.space->get_domain(pd)->get_nbr_coefs());
            ix.set(0) = pi;
            ix.set(1) = pj;
            v.set_coef(ix) += h;
        }
        R->X.push_back(std::move(s));
    }
    R->syst = std::make_unique<System_of_eqs>(*L.space, L.dmin, L.dmax);
    for (int k = 0; k < 3; k++)
        R->syst->add_var(FN[k], *R->X[k]);
    for (const Pt& p : pts)
        for (const std::string& e : exprs) {
            Point M(2);
            M.set(1) = p.r * std::sin(p.th);
            M.set(2) = p.r * std::cos(p.th);
            Point num(L.space->get_domain(p.dom)->absol_to_num(M));
            R->syst->add_eq_point(p.dom, e.c_str(), num);
        }
    return R;
}

}  // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const double RH = 0.7793271081;
    const int nt = 7;
    int fails = 0;

    // ---- the two spaces ----
    Point cr(2);
    cr.set(1) = 0.0;
    cr.set(2) = 0.0;
    Dim_array res17(2);
    res17.set(0) = 17;
    res17.set(1) = nt;
    Array<double> b3(3);
    b3.set(0) = 0.3;
    b3.set(1) = 1.0;
    b3.set(2) = 2.0;
    Space_polar stock(CHEB_TYPE, cr, res17, b3);
    const std::vector<double> pb = {0.0054000000019038905, 0.23643048567470654, 3.9388952741521037};
    std::vector<Dim_array> pres;
    for (int n : {33, 33, 17}) {
        Dim_array d(2);
        d.set(0) = n;
        d.set(1) = nt;
        pres.push_back(d);
    }
    Trumpet::Space_polar_trumpet prod(CHEB_TYPE, cr, pres, pb, {true, true, false});

    std::vector<Layout> lays = {
        {"STOCK Space_polar", &stock, {0.0, 0.3, 1.0}, {0.3, 1.0, 2.0}, {false, false, false}, 1, 2, {17, 17, 17}, nt},
        {"PROD log layout", &prod, {pb[0], pb[1], pb[2]}, {pb[1], pb[2], 0.0}, {true, true, false}, 0, 1, {33, 33, 17}, nt}};
    const std::vector<std::vector<double>> radii = {{RH, 0.45, 1.5}, {RH, 2.5, 0.05}};
    const std::vector<double> ths = {0.02, 0.7, 1.4};
    const std::vector<std::string> ex = {"F", "dr(F)", "Q", "dr(Q)", "B", "dr(B)", "exp(F) * dr(Q) * B"};
    const int NE = static_cast<int>(ex.size());

    double wv = 0, wvp = 0, wj = 0, wjl = 0;
    for (std::size_t li = 0; li < lays.size(); li++) {
        const Layout& L = lays[li];
        std::vector<Pt> pts;
        for (double r : radii[li])
            for (double th : ths) {
                int dom = -1;
                for (int d = L.dmin; d <= L.dmax; d++)
                    if (r > L.lo[d] && r < L.hi[d]) dom = d;
                pts.push_back({r, th, dom});
            }
        auto R0 = build(L, pts, ex);
        Array<double> s0(R0->syst->sec_member());
        if (rank == 0)
            std::cout << "  ==== " << L.name << ": " << pts.size() << " points x " << NE << " rows = "
                      << s0.get_size(0) << " Eq_int rows; unknowns " << R0->syst->get_nbr_unknowns() << "\n"
                      << "   r            th     dom  x (absol_to_num | map)        row        Kadath                 direct"
                         "                 rel(abs-sum)  rel(plain)\n";
        // ---- value and derivative vs the direct evaluation ----
        for (std::size_t p = 0; p < pts.size(); p++) {
            const Pt& P = pts[p];
            const int d = P.dom;
            const Domain* dm = L.space->get_domain(d);
            Point M(2);
            M.set(1) = P.r * std::sin(P.th);
            M.set(2) = P.r * std::cos(P.th);
            const Point num(dm->absol_to_num(M));
            double xm, dxdr;
            if (L.logm[d]) {
                xm = 2.0 * std::log(P.r / L.lo[d]) / std::log(L.hi[d] / L.lo[d]) - 1.0;
                dxdr = 2.0 / (P.r * std::log(L.hi[d] / L.lo[d]));
            } else {
                xm = 2.0 * (P.r - L.lo[d]) / (L.hi[d] - L.lo[d]) - 1.0;
                dxdr = 2.0 / (L.hi[d] - L.lo[d]);
            }
            std::vector<double> T, dT;
            cheb(L.nr[d], num(1), T, dT);
            for (int k = 0; k < 3; k++) {
                (*R0->X[k])(d).coef();
                const Array<double> cf((*R0->X[k])(d).get_coef());
                double v = 0, dv = 0, sv = 0, sdv = 0;
                for (int j = 0; j < nt; j++)
                    for (int i = 0; i < L.nr[d]; i++) {
                        Index ix(dm->get_nbr_coefs());
                        ix.set(0) = i;
                        ix.set(1) = j;
                        const double c = cf(ix), tb = tbas(k, j, num(2));
                        v += c * T[i] * tb;
                        dv += c * dT[i] * dxdr * tb;
                        sv += std::fabs(c * T[i] * tb);
                        sdv += std::fabs(c * dT[i] * dxdr * tb);
                    }
                for (int q = 0; q < 2; q++) {
                    const int row = static_cast<int>(p) * NE + 2 * k + q;
                    const double kad = s0(row), dir = q ? dv : v, sc = q ? sdv : sv;
                    const double ra = std::fabs(kad - dir) / sc, rp = std::fabs(kad - dir) / std::max(std::fabs(dir), 1e-300);
                    wv = std::max(wv, ra);
                    wvp = std::max(wvp, rp);
                    if (ra > 1e-12) fails++;
                    if (rank == 0) {
                        char buf[400];
                        std::snprintf(buf, sizeof buf, "   %-12.10g %-5.2f  %d    %+.15f | %+.15f  %-9s  %+.15e  %+.15e  %.1e       %.1e%s\n",
                                      P.r, P.th, d, num(1), xm, ex[2 * k + q].c_str(), kad, dir, ra, rp,
                                      ra > 1e-12 ? "  FAIL" : "");
                        std::cout << buf;
                    }
                }
            }
        }
        // ---- the Jacobian columns: identify (var, domain, mode) from Kadath's own metadata ----
        std::ostringstream ros, cos_;
        R0->syst->dump_tagged_jacobian_metadata_csv(ros, cos_);
        std::map<std::tuple<std::string, int, int>, int> colof;
        {
            std::istringstream is(cos_.str());
            std::string ln;
            std::getline(is, ln);
            while (std::getline(is, ln)) {
                std::vector<std::string> f;
                std::stringstream ss(ln);
                std::string c;
                while (std::getline(ss, c, ',')) f.push_back(c);
                if (f.size() < 9) continue;
                std::string vn = f[7];
                while (!vn.empty() && vn.back() == ' ') vn.pop_back();
                colof[{vn, std::stoi(f[2]), std::stoi(f[8])}] = std::stoi(f[0]);
            }
        }
        if (rank == 0)
            std::cout << "   Jacobian columns (" << colof.size() << " field columns in the metadata); per column: max over "
                      << "the point rows of |Kadath - FD4| / |Kadath|; linear rows |Kadath - phi_ij| / max|col| (phi_ij: T_i b_j, and for dr the\n"
                      << "   summation of Kadath's own der_r of the unit coefficient)\n";
        std::vector<int> doms;
        for (const Pt& P : pts)
            if (std::find(doms.begin(), doms.end(), P.dom) == doms.end()) doms.push_back(P.dom);
        for (int d : doms) {
            const int nrd = L.nr[d];
            for (int k = 0; k < 3; k++)
                for (auto ij : std::vector<std::pair<int, int>>{{0, 0}, {5, 1}, {nrd - 1, nt - 1}}) {
                    if (k == 2 && ij.second == 0) ij.second = 1;          // SIN_EVEN has no j = 0 function
                    if (k > 0 && ij.second == nt - 1) ij.second = nt - 2;  // COS_ODD / SIN_EVEN: top unknown mode nt - 2
                    // the column metadata's basis_mode counts the UNKNOWN theta modes: SIN_EVEN's start at sin 2 th (j = 1)
                    const int mode = (k == 2 ? ij.second - 1 : ij.second) * nrd + ij.first;
                    auto it = colof.find({FN[k], d, mode});
                    if (it == colof.end()) {
                        if (rank == 0) std::cout << "     " << FN[k] << " d" << d << " (i,j)=(" << ij.first << "," << ij.second
                                                 << "): no column (mode " << mode << ")\n";
                        continue;
                    }
                    Array<double> col(R0->syst->do_col_J(it->second));
                    const double h = 1e-3;
                    const double off[4] = {-2, -1, 1, 2};
                    std::vector<Array<double>> sq;
                    for (int q = 0; q < 4; q++) {
                        auto Rq = build(L, pts, ex, k, d, ij.first, ij.second, off[q] * h);
                        sq.emplace_back(Rq->syst->sec_member());
                    }
                    // Kadath's OWN dr of the unit coefficient (i, j): in a LOG shell dr = (1 / (alpha r)) d/dx is a spectral
                    // product, so dr of a top radial mode is not the exact derivative -- the point layer is gated against
                    // the dr field Kadath builds, the exact derivative is printed beside it
                    Val_domain uv((*R0->X[k])(d));
                    uv.coef();
                    {
                        Index z(L.space->get_domain(d)->get_nbr_coefs());
                        do { uv.set_coef(z) = 0.0; } while (z.inc());
                        z.set(0) = ij.first;
                        z.set(1) = ij.second;
                        uv.set_coef(z) = 1.0;
                    }
                    Val_domain udr(uv.der_r());
                    udr.coef();
                    double wlin = 0, wex = 0, kmax = 0, wrow = 0;
                    for (std::size_t p = 0; p < pts.size(); p++) {
                        if (pts[p].dom != d) continue;
                        for (int e = 0; e < NE; e++) {
                            const int row = static_cast<int>(p) * NE + e;
                            kmax = std::max(kmax, std::fabs(col(row)));
                        }
                    }
                    for (std::size_t p = 0; p < pts.size(); p++) {
                        if (pts[p].dom != d) continue;
                        const Domain* dm = L.space->get_domain(d);
                        Point M(2);
                        M.set(1) = pts[p].r * std::sin(pts[p].th);
                        M.set(2) = pts[p].r * std::cos(pts[p].th);
                        const Point num(dm->absol_to_num(M));
                        std::vector<double> T, dT;
                        cheb(nrd, num(1), T, dT);
                        const double dxdr = L.logm[d] ? 2.0 / (pts[p].r * std::log(L.hi[d] / L.lo[d]))
                                                      : 2.0 / (L.hi[d] - L.lo[d]);
                        for (int e = 0; e < NE; e++) {
                            const int row = static_cast<int>(p) * NE + e;
                            const double fd = (sq[0](row) - 8 * sq[1](row) + 8 * sq[2](row) - sq[3](row)) / (12 * h);
                            const double kd = col(row);
                            const double den = std::max(std::fabs(kd), 1e-14 * kmax);
                            if (std::fabs(kd) > 1e-12 * kmax || std::fabs(fd) > 1e-12 * kmax)
                                wrow = std::max(wrow, std::fabs(kd - fd) / std::max(den, 1e-300));
                            if (e / 2 == k && e < 6) {
                                const double ex_ = (e % 2 ? dT[ij.first] * dxdr : T[ij.first]) * tbas(k, ij.second, num(2));
                                const double phi = e % 2 ? (udr.check_if_zero() ? 0.0 : udr.get_base().summation(num, udr.get_coef()))
                                                         : ex_;
                                wlin = std::max(wlin, std::fabs(kd - phi) / std::max(kmax, 1e-300));
                                if (e % 2) wex = std::max(wex, std::fabs(kd - ex_) / std::max(kmax, 1e-300));
                                if (std::getenv("T3A_VERBOSE") && rank == 0)
                                    std::printf("        p%zu %-8s Kadath %+.6e  FD4 %+.6e  phi %+.6e  exact %+.6e\n", p, ex[e].c_str(),
                                                kd, fd, phi, ex_);
                            }
                            if (e / 2 != k && e < 6 && std::fabs(kd) > 0)
                                wlin = std::max(wlin, std::fabs(kd) / std::max(kmax, 1e-300));   // other fields: must be 0
                        }
                    }
                    wj = std::max(wj, wrow);
                    wjl = std::max(wjl, wlin);
                    if (wrow > 1e-8 || wlin > 1e-12) fails++;
                    if (rank == 0) {
                        char buf[400];
                        std::snprintf(buf, sizeof buf, "     %s d%d (i,j)=(%2d,%d) column %5d:  vs FD4 %.1e   linear rows vs phi_ij %.1e"
                                      "   (dr rows vs the EXACT basis derivative %.1e)%s\n",
                                      FN[k], d, ij.first, ij.second, it->second, wrow, wlin, wex,
                                      (wrow > 1e-8 || wlin > 1e-12) ? "  FAIL" : "");
                        std::cout << buf;
                    }
                }
        }
    }
    if (rank == 0) {
        std::printf("  3a worst: value / derivative %.1e (abs-sum relative; plain relative %.1e);  Jacobian vs FD4 %.1e;"
                    "  linear-row columns vs phi_ij %.1e\n", wv, wvp, wj, wjl);
        std::cout << "  3a " << (fails ? "FAIL" : "PASS") << " (" << fails << " checks beyond the gates)\n";
    }
    MPI_Finalize();
    return fails ? 1 : 0;
}

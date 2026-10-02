/*
 * t1_nscalar_main.cpp -- research round 664 (code round 335), gate T1.
 *
 * A STANDALONE smoke test, part of no production binary: does a scalar unknown n
 * (add_var "NN") inside a matching-type row's exp(k n ln r) linearise correctly
 * in Kadath?
 *
 * Space_polar (nucleus + two shells, bounds 0.3 / 1.0 / 2.0), stock Kadath only.
 * One row per case, imposed by add_eq_mode at the cos 0 mode of a boundary:
 *
 *     F - GG * exp(K * NN * log(RR)) = 0
 *
 * F a known theta-dependent field (add_cst), RR the radius (add_cst), K a
 * constant (add_cst), GG and NN scalar unknowns (add_var).  Kadath's Jacobian
 * column for NN (do_col_J) is compared with a central finite difference of the
 * residual (sec_member) in NN, and with the analytic derivative
 *     d/dNN = -GG K ln(r) exp(K NN ln r)   (the cos 0 coefficient; F drops out).
 * Cases: K = 1, 3 at r = 0.3 (shell 1 inner face) and r = 2.0 (shell 2 outer face).
 * Gate: |Kadath - FD| / |Kadath| <= 1e-8, FD the 4th-order central difference (h = 1e-3); the 2nd-order one
 * (h = 1e-6) is printed too -- it is roundoff-limited at ~1e-8 when |dR/dNN| << |R|.
 */

#include <mpi.h>

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace Kadath;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int nr = 13, nt = 7;
    Dim_array res(2);
    res.set(0) = nr;
    res.set(1) = nt;
    Point cr(2);
    cr.set(1) = 0.0;
    cr.set(2) = 0.0;
    Array<double> bounds(3);
    bounds.set(0) = 0.3;
    bounds.set(1) = 1.0;
    bounds.set(2) = 2.0;
    Space_polar space(CHEB_TYPE, cr, res, bounds);

    Scalar F(space), RR(space);
    for (int d = 0; d < space.get_nbr_domains(); d++) {
        const Domain* dm = space.get_domain(d);
        Val_domain& vf = F.set_domain(d);
        Val_domain& vr = RR.set_domain(d);
        vf.allocate_conf();
        vr.allocate_conf();
        Index idx(dm->get_nbr_points());
        do {
            const double rr = dm->get_radius()(idx);
            const double th = dm->get_coloc(2)(idx(1));
            vf.set(idx) = 1.5 + 0.2 * rr + 0.3 * std::cos(2.0 * th);
            vr.set(idx) = rr;
        } while (idx.inc());
    }
    F.std_base();
    RR.std_base();

    struct Case { double k; int dom; int bc; double r; };
    const std::vector<Case> cases = {{1.0, 1, INNER_BC, 0.3}, {3.0, 1, INNER_BC, 0.3},
                                     {1.0, 2, OUTER_BC, 2.0}, {3.0, 2, OUTER_BC, 2.0}};
    int fails = 0;
    if (rank == 0)
        std::cout << "  T1: row F - GG exp(K NN log RR) = 0, cos 0 mode, NN = sqrt2, GG = 0.7\n"
                  << "     K     r     Kadath dR/dNN          central FD (h = 1e-6)     analytic"
                     "                 rel(Kadath, FD2)  rel(Kadath, analytic)\n";
    for (const Case& c : cases) {
        double nn = std::sqrt(2.0), gg = 0.7;
        Scalar KK(space);
        for (int d = 0; d < space.get_nbr_domains(); d++) {
            Val_domain& vk = KK.set_domain(d);
            vk.allocate_conf();
            Index idx(space.get_domain(d)->get_nbr_points());
            do { vk.set(idx) = c.k; } while (idx.inc());
        }
        KK.std_base();
        System_of_eqs syst(space, c.dom, c.dom);
        syst.add_cst("F", F);
        syst.add_cst("RR", RR);
        syst.add_cst("KK", KK);
        syst.add_var("GG", gg);
        syst.add_var("NN", nn);
        Index pos_cf(space.get_domain(c.dom)->get_nbr_coefs());
        syst.add_eq_mode(c.dom, c.bc, "F - GG * exp(KK * NN * log(RR))", pos_cf, 0.0);
        const int nu = syst.get_nbr_unknowns();
        Array<double> r0(syst.sec_member());
        // NN is the last unknown registered: its column
        Array<double> col(syst.do_col_J(nu - 1));
        const double kad = col(0);
        const double h = 1e-6;
        nn = std::sqrt(2.0) + h;
        Array<double> rp(syst.sec_member());
        nn = std::sqrt(2.0) - h;
        Array<double> rm(syst.sec_member());
        nn = std::sqrt(2.0);
        // the 4th-order central difference (h4 = 1e-3): truncation ~ h4^4 and roundoff ~ eps |R| / h4, both far below 1e-8
        const double h4 = 1e-3;
        double r4[4];
        const double off[4] = {-2, -1, 1, 2};
        for (int q = 0; q < 4; q++) {
            nn = std::sqrt(2.0) + off[q] * h4;
            Array<double> rq(syst.sec_member());
            r4[q] = rq(0);
        }
        nn = std::sqrt(2.0);
        const double fd4 = (r4[0] - 8 * r4[1] + 8 * r4[2] - r4[3]) / (12 * h4);
        // sec_member is -F(x) in Kadath's convention (Newton solves J dx = sec_member); compare magnitudes with sign
        const double fd = (rp(0) - rm(0)) / (2 * h);
        const double an = -gg * c.k * std::log(c.r) * std::exp(c.k * std::sqrt(2.0) * std::log(c.r));
        const double rfd = std::fabs(kad - fd) / std::fabs(kad);
        const double rfd4 = std::fabs(kad - fd4) / std::fabs(kad);
        const double ran = std::fabs(kad - an) / std::fabs(kad);
        const bool sgn_ok = kad * an > 0;
        if (rfd4 > 1e-8 || ran > 1e-8)          // the gate: Kadath vs the 4th-order FD (and the analytic value)
            fails++;
        if (rank == 0)
            std::cout << std::setprecision(4) << "     " << std::setw(5) << c.k << " " << std::setw(5) << c.r
                      << std::setprecision(15) << "  " << std::setw(22) << kad << "  " << std::setw(22) << fd
                      << "  " << std::setw(22) << an << std::setprecision(3) << "  " << std::setw(12) << rfd
                      << "  " << std::setw(12) << ran << (sgn_ok ? "" : "  (sign)")
                      << "   4th-order FD " << std::setprecision(15) << fd4 << std::setprecision(3) << " rel " << rfd4 << "\n";
        if (rank == 0)
            std::cout << "        (unknowns " << nu << ", rows " << r0.get_size(0) << ", residual " << r0(0)
                      << "; sign convention: Kadath col / FD of sec_member = " << kad / fd << ")\n";
    }
    if (rank == 0)
        std::cout << "  T1 " << (fails ? "FAIL" : "PASS") << " (" << fails << " of " << cases.size()
                  << " cases beyond 1e-8)\n";
    MPI_Finalize();
    return fails ? 1 : 0;
}

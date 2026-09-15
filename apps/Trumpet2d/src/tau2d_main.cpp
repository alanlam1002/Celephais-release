/*
 * tau2d_main.cpp -- A0: does the TAU machinery work on a log radial domain?
 *
 * Research round 284 ruled this its own rung.  Domain_polar_shell_log inherits
 * export_tau, affecte_tau, nbr_conditions and val_boundary unchanged, on the
 * argument that they are coefficient-space operations and so blind to the
 * radial mapping.  That argument is exactly the shape of the three that have
 * already failed -- der_abs(1) (round 89), val_point (round 90), laplacian
 * (round 94) -- each of which was inherited, unexercised, and wrong.  Three for
 * three is a base rate, not a prior.
 *
 * So: a manufactured System_of_eqs solve with a known answer, on a log layout,
 * isolating "tau works on a log domain" from "the l0_solve port is right".
 *
 *     lap(u) = s   on every domain,
 *     u = u_exact  at the excision,   u -> 0 at infinity,
 *     C0 and C1 matching at every interface.
 *
 * ⚠ THE MANUFACTURED SOLUTION MUST NOT BE REPRESENTABLE, which research asked
 * for explicitly: a field the truncated basis carries exactly would satisfy any
 * tau bookkeeping, and the test would pass with the count wrong.  So
 *
 *     u_exact = A(r) f(theta),  f = 1/(1 - 2q cos2theta + q^2)
 *
 * whose COS_EVEN coefficients are exactly q^k -- geometric, never terminating
 * (round 90) -- times a rational A(r) that no Chebyshev truncation carries
 * exactly either.  The coefficient tail of u_exact on the grid is emitted so
 * that claim is a number rather than an assertion.
 *
 * ⚠ AND THE SOURCE IS ANALYTIC.  s = lap(u_exact) is imported as an add_cst
 * evaluated in closed form, NOT by differentiating an interpolant -- otherwise
 * the source carries the very truncation the test is trying to measure.
 *
 * The verdict is convergence: the solution error must fall spectrally in ntheta
 * on BOTH mappings.  A wrong tau count gives a system that is not square (which
 * throws), or square and wrong (which does not converge).
 *
 * Usage: tau2d <table.dat> [--ntheta N] [--q Q] [--log] [--order N]
 */

#include <mpi.h>

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include "space/space_polar_trumpet.hpp"
#include "manufactured.hpp"
#include "Trumpet1d/src/table_io.hpp"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using Kadath::Dim_array;
using Kadath::Index;
using Kadath::Point;
using Kadath::Scalar;
using Kadath::System_of_eqs;
using Kadath::Val_domain;
using Trumpet::Angular;
using Trumpet::emit;
using Trumpet::Radial;
using TrumpetIO::Table;
using TrumpetIO::read_table;

int main(int argc, char** argv)
{
    // do_newton goes through MPI even at one rank, as l0_solve does.
    MPI_Init(&argc, &argv);
    if (argc < 2) {
        std::cerr << "usage: tau2d <table.dat> [--ntheta N] [--q Q] [--log]\n";
        MPI_Finalize();
        return 2;
    }
    int ntheta = 9;
    double q = 0.4, Lfac = 2.0;
    bool uselog = false;
    int order = 2;                 // the natural tau order of a second-order row
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--ntheta" && i + 1 < argc) ntheta = std::atoi(argv[++i]);
        else if (a == "--q" && i + 1 < argc) q = std::atof(argv[++i]);
        else if (a == "--L" && i + 1 < argc) Lfac = std::atof(argv[++i]);
        else if (a == "--log") uselog = true;
        else if (a == "--order" && i + 1 < argc) order = std::atoi(argv[++i]);
    }

    Table t;
    try {
        t = read_table(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        MPI_Finalize();
        return 1;
    }
    const int ndom = static_cast<int>(t.doms.size());
    const int dlast = ndom - 1;

    std::vector<double> bounds;
    std::vector<Dim_array> res;
    std::vector<bool> logshell;
    for (int d = 0; d < ndom; d++) {
        logshell.push_back(uselog || t.doms[d].kind == "log");
        bounds.push_back(t.doms[d].r_int);
        Dim_array n(2);
        n.set(0) = t.doms[d].nbr;
        n.set(1) = ntheta;
        res.push_back(n);
    }
    Point centre(2);
    centre.set(1) = 0.0;
    centre.set(2) = 0.0;
    Trumpet::Space_polar_trumpet space(CHEB_TYPE, centre, res, bounds, logshell);

    std::cout << "# tau2d  ntheta=" << ntheta << " q=" << q << " ndom=" << ndom
              << (uselog ? "  LOG" : "  LINEAR") << "\n";
    emit("T_ntheta", ntheta);
    emit("T_q", q);
    {   // the mapping ACTUALLY built, read off the objects
        int nlog = 0;
        for (int d = 0; d < dlast; d++)
            if (dynamic_cast<const Kadath::Domain_polar_shell_log*>(space.get_domain(d)))
                nlog++;
        emit("T_log_shells_built", nlog);
        emit("T_shells", dlast);
    }

    const Radial A{Lfac, 0};
    const Angular f{0, q};                      // COS_EVEN, coefficients q^k

    // ---- the manufactured solution, its analytic source, and the BC value ---
    Scalar uex(space), src(space), ubc(space);
    for (int d = 0; d < ndom; d++) {
        const Kadath::Domain* dm = space.get_domain(d);
        Val_domain rad = dm->get_radius();
        Val_domain& vu = uex.set_domain(d);
        Val_domain& vs = src.set_domain(d);
        vu.allocate_conf();
        vs.allocate_conf();
        Index idx(dm->get_nbr_points());
        do {
            const double rr = rad(idx);
            const double th = dm->get_coloc(2)(idx(1));
            if (!std::isfinite(rr)) {           // r = infinity on the compact face
                vu.set(idx) = 0.0;
                vs.set(idx) = 0.0;
                continue;
            }
            const double st = std::sin(th), ct = std::cos(th);
            const double f0 = f.d(0, th), f1 = f.d(1, th), f2 = f.d(2, th);
            const double cotf1 = (idx(1) == 0) ? f2 : (ct / st) * f1;
            vu.set(idx) = A.d(0, rr) * f0;
            // analytic laplacian -- NOT a derivative of the interpolant
            vs.set(idx) = A.d(2, rr) * f0 + 2.0 * A.d(1, rr) * f0 / rr
                          + A.d(0, rr) * (f2 + cotf1) / (rr * rr);
        } while (idx.inc());
    }
    uex.std_base();
    src.std_base();
    ubc = uex;
    ubc.std_base();

    // is the exact solution actually NOT representable?  the test depends on it
    {
        const Kadath::Domain* dm = space.get_domain(0);
        Val_domain v(uex(0));
        v.coef();
        const Dim_array& nc = dm->get_nbr_coefs();
        double top = 0.0, mx = 0.0;
        Index ic(nc);
        do {
            const double c = std::fabs(v.get_coef(ic));
            if (c > mx) mx = c;
            if ((ic(0) >= nc(0) - 2 || ic(1) >= nc(1) - 2) && c > top) top = c;
        } while (ic.inc());
        emit("T_uexact_tail_d0", (mx > 0.0) ? top / mx : 0.0);
    }

    // ---- THE ANGULAR CONDITION COUNT, against round 90's degeneracy --------
    //
    // Round 90 measured that on the DCT-I collocation the fold k -> 2P-k maps
    // the top angular mode onto itself, so its extracted VALUE is contaminated
    // by the whole truncated tail.  That is an accuracy statement, not a rank
    // one: the N even modes cos(2k.theta) are still linearly independent on N
    // points, because the transform is invertible.  The class where a mode is
    // genuinely DEPENDENT is the odd one: with m = 2k+1 the top mode m = 2P+1
    // aliases onto m = 2P-1, which is also retained -- exactly the degeneracy
    // round 90 hit when the odd alias formula broke at k = P.
    //
    // So the prediction is
    //     COS_EVEN (scalars)          ntheta       x (nr - order)
    //     COS_ODD  (beta^theta, Qbar) (ntheta - 1) x (nr - order)
    // and it is checked here on BOTH mappings, since tau bookkeeping is the
    // layout-adjacent part.
    {
        int worst = 0;
        for (int d = 0; d < dlast; d++) {
            const auto* ps =
                dynamic_cast<const Kadath::Domain_polar_shell*>(space.get_domain(d));
            if (!ps)
                continue;
            const int nr = ps->get_nbr_coefs()(0);
            const int nt = ps->get_nbr_coefs()(1);
            Val_domain ve(ps), vo(ps);
            ve.allocate_conf();
            vo.allocate_conf();
            ve = 1.0;
            vo = 1.0;
            ve.std_base();          // COS_EVEN
            vo.std_anti_base();     // COS_ODD
            for (int order = 0; order <= 2; order++) {
                const int ge = ps->nbr_conditions_val_domain(ve, 0, order);
                const int go = ps->nbr_conditions_val_domain(vo, 0, order);
                const int we = nt * (nr - order);
                const int wo = (nt - 1) * (nr - order);
                worst = std::max(worst, std::abs(ge - we));
                worst = std::max(worst, std::abs(go - wo));
                if (d == 0) {
                    emit("T_cnt_even_o" + std::to_string(order), ge);
                    emit("T_cnt_even_pred_o" + std::to_string(order), we);
                    emit("T_cnt_odd_o" + std::to_string(order), go);
                    emit("T_cnt_odd_pred_o" + std::to_string(order), wo);
                }
            }
        }
        emit("T_cnt_worst_mismatch", worst);
    }

    // ------------------------------------------------------------ the system --
    Scalar u(space);
    for (int d = 0; d < ndom; d++)
        u.set_domain(d) = 0.0;
    u.std_base();

    System_of_eqs syst(space, 0, dlast);
    syst.add_var("u", u);
    syst.add_cst("src", src);
    syst.add_cst("ubc", ubc);

    // ⚠ THE CONTROL research asked for: would a wrong tau count SHOW?  A field
    // the truncated basis carried exactly would satisfy any bookkeeping, so
    // --order deliberately mis-states it.  order 2 is the natural one for a
    // second-order row; 1 and 3 must break the solve or the answer, and if they
    // do not the test is vacuous whatever it reports at order 2.
    emit("T_tau_order", order);
    for (int d = 0; d <= dlast; d++) {
        if (order == 2)
            syst.add_eq_inside(d, "lap(u) = src");
        else
            syst.add_eq_order(d, order, "lap(u) = src");
    }
    for (int d = 0; d < dlast; d++) {
        syst.add_eq_matching(d, OUTER_BC, "u");
        syst.add_eq_matching(d, OUTER_BC, "dr(u)");
    }
    syst.add_eq_bc(0, INNER_BC, "u = ubc");
    syst.add_eq_bc(dlast, OUTER_BC, "u = 0");

    emit("T_nbr_unknowns", syst.get_nbr_unknowns());

    // do_newton takes ONE step per call and reports the residual it started
    // from, so a linear problem still needs the loop: the first call returns
    // false having just produced the exact discrete solution.
    double err = 0.0;
    bool ok = false;
    int iters = 0;
    try {
        for (iters = 0; iters < 12 && !ok; iters++)
            ok = syst.do_newton(1e-12, err);
    } catch (const std::exception& e) {
        std::cerr << "FATAL in do_newton: " << e.what() << "\n";
        emit("T_newton_ok", 0);
        MPI_Finalize();
        return 1;
    }
    // nbr_conditions is computed lazily inside do_newton, so it reads -1 until
    // the system has actually been assembled once.
    emit("T_nbr_conditions", syst.get_nbr_conditions());
    emit("T_square",
         syst.get_nbr_unknowns() == syst.get_nbr_conditions() ? 1 : 0);
    emit("T_newton_ok", ok ? 1 : 0);
    emit("T_newton_iters", iters);
    emit("T_newton_residual", err);

    // ------------------------------------------------------ the known answer --
    double e_abs = 0.0, scale = 0.0;
    for (int d = 0; d < ndom; d++) {
        const Kadath::Domain* dm = space.get_domain(d);
        Index idx(dm->get_nbr_points());
        Val_domain rad = dm->get_radius();
        do {
            if (!std::isfinite(rad(idx)))
                continue;
            e_abs = std::max(e_abs, std::fabs(u(d)(idx) - uex(d)(idx)));
            scale = std::max(scale, std::fabs(uex(d)(idx)));
        } while (idx.inc());
    }
    emit("T_sol_err_abs", e_abs);
    emit("T_sol_err_rel", e_abs / std::max(scale, 1e-300));
    std::cout << "# done\n";
    MPI_Finalize();
    return 0;
}

/*
 * backbone2d_main.cpp -- A0: Trumpet1d's Phase-1 battery, in two dimensions.
 *
 * NOTES_global_solve.md section 9 rules that Trumpet2d reproduce Trumpet1d's
 * O(j^2) numbers BEFORE the finite-J equations exist, so that "the 2-D
 * machinery works" is separated from "the equations are right".  This app is
 * the first rung: the SAME backbone table, imported onto the SAME radial
 * layout but on a 2-D polar grid, emitting the SAME RESULT keys.
 *
 * WHY THIS IS A TEST AND NOT A TRANSCRIPTION.  The backbone is spherically
 * symmetric, so on a polar grid the imported field is theta-independent by
 * construction and every Phase-1 oracle must come back UNCHANGED.  Anything
 * that moves is 2-D machinery, not physics:
 *   G1  Domain_polar_shell/compact get_radius() against the same formulas
 *   G2  a2(R) - W^2 pointwise, now over (r, theta)
 *   G3  the same off-grid, through val_point with a 2-D Point
 *   G4  der_abs(1) -- the radial derivative operator, in 2-D
 *   G5  val_boundary(INNER_BC, ., pcf) on a polar shell
 *   G6  the mass-mode tails at infinity through mult_r on Domain_polar_compact
 *   G7  the throat series, pure arithmetic, unchanged
 *
 * ONE THING IS GENUINELY DIFFERENT AND IS MEASURED RATHER THAN ASSUMED.  In
 * 1-D, val_boundary(bound, vd, pcf) with pcf = (0) returns the boundary VALUE.
 * In 2-D it returns the coefficient of the pcf-th ANGULAR basis function there,
 * and whether the theta-constant basis function is normalised to 1 is a
 * property of the basis, not something to assume.  G5 and G6 therefore report
 * the ratio against the 1-D answer, so a normalisation factor shows up as a
 * number rather than as a silent offset.
 *
 * Usage:  backbone2d <table.dat> [--offgrid N] [--ntheta N]
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"

#include "space/space_polar_trumpet.hpp"
#include "Trumpet1d/src/table_io.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using Kadath::Dim_array;
using Kadath::Index;
using Kadath::Point;
using Kadath::Scalar;
using Kadath::Val_domain;
using TrumpetIO::PointRow;
using TrumpetIO::Table;
using TrumpetIO::read_table;

namespace
{

void emit(const std::string& key, double v)
{
    std::cout << "RESULT " << key << " " << std::setprecision(17) << v << "\n";
}

/** Fill one Val_domain from a per-RADIAL-point accessor, constant in theta. */
template <typename F>
void fill2d(Scalar& f, int d, const Table& t, F get)
{
    Val_domain& vd = f.set_domain(d);
    vd.allocate_conf();
    const Dim_array& np = f.get_space().get_domain(d)->get_nbr_points();
    Index idx(np);
    do {
        vd.set(idx) = get(t.pts[d][idx(0)]);
    } while (idx.inc());
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "usage: backbone2d <table.dat> [--offgrid N] [--ntheta N]\n";
        return 2;
    }
    int noff = 10, ntheta = 5;
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--offgrid" && i + 1 < argc)
            noff = std::atoi(argv[++i]);
        else if (a == "--ntheta" && i + 1 < argc)
            ntheta = std::atoi(argv[++i]);
    }

    Table t;
    try {
        t = read_table(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
    if (t.mode == "stock") {
        std::cerr << "FATAL: A0 runs on the EXCISED layout; 'stock' has an r=0 "
                     "nucleus which is what D1 exists to avoid.\n";
        return 1;
    }

    const int ndom = static_cast<int>(t.doms.size());
    std::cout << "# table  " << t.tag << "  mode=" << t.mode << "  ndom=" << ndom
              << "  M=" << t.M << "  ntheta=" << ntheta << "\n";

    // The table's per-domain `kind` selects the radial mapping: "log" builds a
    // Domain_polar_shell_log (round 94).  Taken from the TABLE rather than from
    // a flag, so the collocation radii the table carries and the mapping the
    // space builds cannot disagree -- G1 would catch it, but only after the
    // fact, and a mismatch here is the class of error round 91 produced.
    std::vector<double> bounds;
    std::vector<Dim_array> res;
    std::vector<bool> logshell;
    for (int d = 0; d < ndom; d++) {
        logshell.push_back(t.doms[d].kind == "log");
        bounds.push_back(t.doms[d].r_int);
        Dim_array n(2);
        n.set(0) = t.doms[d].nbr;      // radial: identical to the 1-D layout
        n.set(1) = ntheta;             // angular
        res.push_back(n);
    }
    Point center(2);
    center.set(1) = 0.0;
    center.set(2) = 0.0;

    Trumpet::Space_polar_trumpet space(CHEB_TYPE, center, res, bounds, logshell);
    if (space.get_nbr_domains() != ndom) {
        std::cerr << "FATAL: space has " << space.get_nbr_domains()
                  << " domains, table has " << ndom << "\n";
        return 1;
    }
    emit("A0_ndim", space.get_ndim());
    emit("A0_ndom", space.get_nbr_domains());
    {   // the mapping ACTUALLY built, read off the objects, not off the table
        int nlog = 0, want = 0;
        for (int d = 0; d < ndom - 1; d++) {
            if (dynamic_cast<const Kadath::Domain_polar_shell_log*>(space.get_domain(d)))
                nlog++;
            if (logshell[d])
                want++;
        }
        emit("A0_log_shells_built", nlog);
        emit("A0_log_shells_in_table", want);
        if (nlog != want) {
            std::cerr << "FATAL: built " << nlog << " log shells, table asks for "
                      << want << "\n";
            return 1;
        }
    }

    // -------------------------------------- G1: collocation radii, 2-D grid --
    // Also checks that the radius is theta-INDEPENDENT, which it must be on a
    // polar shell and which nothing else in the battery would notice.
    double g1 = 0.0, g1th = 0.0;
    for (int d = 0; d < ndom; d++) {
        const Kadath::Domain* dom = space.get_domain(d);
        Val_domain rad = dom->get_radius();
        const Dim_array& np = dom->get_nbr_points();
        Index idx(np);
        do {
            const double rk = rad(idx);
            const double rf = t.pts[d][idx(0)].r;
            if (!std::isfinite(rf) || !std::isfinite(rk)) {
                if (std::isfinite(rf) != std::isfinite(rk)) {
                    std::cerr << "FATAL: finiteness mismatch d=" << d
                              << " i=" << idx(0) << "\n";
                    return 1;
                }
                continue;
            }
            g1 = std::max(g1, std::fabs(rk - rf) / std::max(1.0, std::fabs(rf)));
        } while (idx.inc());
        // theta-independence of the radius
        Index i0(np), i1(np);
        for (int i = 0; i < np(0); i++) {
            i0.set(0) = i; i0.set(1) = 0;
            i1.set(0) = i; i1.set(1) = np(1) - 1;
            const double a = rad(i0), b = rad(i1);
            if (std::isfinite(a) && std::isfinite(b))
                g1th = std::max(g1th, std::fabs(a - b) / std::max(1.0, std::fabs(a)));
        }
    }
    emit("G1_radius_max_rel_diff", g1);
    emit("A0_radius_theta_indep", g1th);

    // ------------------------------------------------------- import backbone --
    Scalar Wf(space), Rrf(space), oorf(space);
    for (int d = 0; d < ndom; d++) {
        fill2d(Wf, d, t, [](const PointRow& p) { return p.W; });
        fill2d(Rrf, d, t, [](const PointRow& p) { return p.Rr; });
        fill2d(oorf, d, t, [](const PointRow& p) { return p.oor; });
    }
    Wf.std_base();
    Rrf.std_base();
    oorf.std_base();

    Scalar iR(space);
    for (int d = 0; d < ndom; d++)
        iR.set_domain(d) = oorf(d) / Rrf(d);
    iR.std_base();

    const double M = t.M;
    Scalar a2(space);
    for (int d = 0; d < ndom; d++)
        a2.set_domain(d) = 1.0 - 2.0 * M * iR(d)
                           + (27.0 / 16.0) * std::pow(M, 4) * Kadath::pow(iR(d), 4);
    a2.std_base();

    // ------------------------------------------ G2: a2(R) - W^2 on the grid --
    double g2 = 0.0;
    for (int d = 0; d < ndom; d++) {
        Val_domain rv = a2(d) - Wf(d) * Wf(d);
        Index idx(space.get_domain(d)->get_nbr_points());
        do {
            g2 = std::max(g2, std::fabs(rv(idx)));
        } while (idx.inc());
    }
    emit("G2_a2_minus_W2_max", g2);

    // -------------------------------------- G3: same, off-grid, 2-D val_point --
    // A0 FINDING (theta rung).  Scalar::val_point takes a Point in ABSOLUTE
    // (rho, z) coordinates, not (r, theta): Domain_polar_*::absol_to_num forms
    // air = sqrt(abs(1)^2 + abs(2)^2).  The first version of this loop passed
    // (r, theta) and so sampled radius sqrt(r^2 + theta^2).  It did NOT fail,
    // because a2 - W^2 vanishes at EVERY radius and both sides are evaluated at
    // the same (wrong) point -- G3 is blind to the coordinate convention.  The
    // theta rung, whose manufactured field is not constant, is not.
    double g3 = 0.0;
    for (int d = 0; d < ndom; d++) {
        const double a = t.doms[d].r_int;
        double b = t.doms[d].r_ext;
        if (!std::isfinite(b))
            b = a * 8.0;
        for (int k = 1; k <= noff; k++) {
            const double rr = a + (b - a) * (double(k) - 0.5) / double(noff);
            for (int m = 0; m < 3; m++) {
                const double th = M_PI * (double(m) + 0.5) / 3.0;
                Point pt(2);
                pt.set(1) = rr * std::sin(th);      // rho
                pt.set(2) = rr * std::cos(th);      // z
                const double wv = Wf.val_point(pt);
                const double iv = iR.val_point(pt);
                const double av = 1.0 - 2.0 * M * iv
                                  + (27.0 / 16.0) * std::pow(M, 4) * std::pow(iv, 4);
                g3 = std::max(g3, std::fabs(av - wv * wv));
            }
        }
    }
    emit("G3_a2_minus_W2_offgrid_max", g3);

    // ------------------------------ G4: spectral ODE residual, der_abs(1) in 2-D --
    // A0 FINDING, kept in the code because it is the kind of thing this rung
    // exists to catch: the derivative does NOT transliterate.  In 1-D,
    // Val_domain::der_abs(1) is d/dx and x IS r, so Trumpet1d's G4 reads the
    // radial derivative.  In 2-D polar, der_abs(1) is d/dx CARTESIAN, and using
    // it here gives an ODE residual of 1.0 -- a 100% error that looks like a
    // broken port and is a wrong operator.  The radial derivative is the DOMAIN
    // method Domain::der_r.  Both are computed below so the difference is a
    // measurement rather than a comment.
    double g4 = 0.0, g4abs = 0.0;
    for (int d = 0; d < ndom; d++) {
        const Kadath::Domain* dm = space.get_domain(d);
        {
            Val_domain dW = Rrf(d).der_abs(1);
            Val_domain rw = dW - Rrf(d) * (Wf(d) - 1.0) * oorf(d);
            Index ix(dm->get_nbr_points());
            double nu = 0.0, sc = 0.0;
            do {
                nu = std::max(nu, std::fabs(rw(ix)));
                sc = std::max(sc, std::fabs(dW(ix)));
            } while (ix.inc());
            g4abs = std::max(g4abs, nu / std::max(sc, 1e-300));
        }
        Val_domain dRr = dm->der_r(Rrf(d));
        Val_domain resid = dRr - Rrf(d) * (Wf(d) - 1.0) * oorf(d);
        Index idx(space.get_domain(d)->get_nbr_points());
        double num = 0.0, scale = 0.0;
        do {
            num = std::max(num, std::fabs(resid(idx)));
            scale = std::max(scale, std::fabs(dRr(idx)));
        } while (idx.inc());
        const double rel = num / std::max(scale, 1e-300);
        emit("G4_ode_rel_d" + std::to_string(d), rel);
        g4 = std::max(g4, rel);
    }
    emit("G4_ode_rel_max", g4);
    emit("A0_G4_with_der_abs1", g4abs);

    // ------------------------------------------------ G5: W at the excision --
    {
        const Kadath::Domain* dom = space.get_domain(0);
        Index pcf(dom->get_nbr_coefs());     // all zero: the theta-constant mode
        const double win = dom->val_boundary(INNER_BC, Wf(0), pcf);
        emit("G5_W_at_inner", win);
        emit("G5_W_at_inner_err", std::fabs(win - t.W0));
        // the normalisation question, measured rather than assumed
        emit("A0_G5_ratio_to_1d", win / t.W0);
    }

    // --------------------------------------------- G6: mass-mode tails at inf --
    {
        const int dl = ndom - 1;
        const Kadath::Domain* dom = space.get_domain(dl);
        Val_domain U_M = (1.0 - Wf(dl)) * 0.5;
        Val_domain F_M = -(iR(dl) - (27.0 / 8.0) * std::pow(M, 3) * Kadath::pow(iR(dl), 4))
                         / Wf(dl);
        Val_domain Q_M = 0.0 * Wf(dl);
        Index pcf(dom->get_nbr_coefs());
        const double tU = dom->val_boundary(OUTER_BC, dom->mult_r(U_M), pcf);
        const double tG = dom->val_boundary(OUTER_BC, dom->mult_r(F_M), pcf);
        const double tQ = dom->val_boundary(OUTER_BC, dom->mult_r(Q_M), pcf);
        emit("G6_tU", tU);
        emit("G6_tQ", tQ);
        emit("G6_tG", tG);
        emit("G6_komar_2tU_plus_tG", 2.0 * tU + tG);
        emit("G6_tG_over_tU", tG / tU);
        emit("G6_U_at_inf", dom->val_boundary(OUTER_BC, U_M, pcf));
        emit("G6_W_at_inf", dom->val_boundary(OUTER_BC, Wf(dl), pcf));
        // tG/tU is normalisation-INDEPENDENT, so it is the clean cross-check
        // against the 1-D value even if the angular basis is not unit-normalised.
        emit("A0_G6_ratio_tU_to_half", tU / 0.5);
    }

    // ------------------------------------- G7: throat series R = 3/2 + c_k W^k --
    {
        static const double c[8] = {1.0606601717798212866,  1.0,
                                    0.98700321540622258614, 0.98611111111111111111,
                                    0.98853773533067255907, 0.99151234567901234568,
                                    0.99413447049956926588, 0.99617412551440329218};
        // ⚠ ROUND 94.  This series converges near the THROAT only, and the
        // test was implicitly tied to the layout: on the production W005 layout
        // d0 spans W in [0.05, 0.12] and it holds to 5.9e-9, but on a wide log
        // layout d0 reaches W = 0.78 and the same test returns 0.46.  That is
        // the ORACLE'S domain of validity, not a defect in the field.
        //
        // BOTH are emitted.  The unrestricted one keeps the bit-for-bit match
        // with Trumpet1d that A0a rests on; the restricted one is the diagnostic
        // that stays meaningful when d0 is not a throat shell.  A cut applied
        // silently would have broken the 1-D comparison to fix a layout that is
        // not yet in use.
        const double W_CUT = 0.12;
        double g7 = 0.0, g7r = 0.0, g7wmax = 0.0, wmax = 0.0;
        int g7n = 0;
        for (int i = 0; i < t.doms[0].nbr; i++) {
            const double w = t.pts[0][i].W;
            if (w > wmax)
                wmax = w;
            const double Rv = t.pts[0][i].Rr * t.pts[0][i].r;
            double ser = 1.5 * M, wp = 1.0;
            for (double ck : c) {
                wp *= w;
                ser += ck * wp * M;
            }
            const double e = std::fabs(Rv - ser);
            g7 = std::max(g7, e);
            if (w <= W_CUT) {
                g7n++;
                g7r = std::max(g7r, e);
                if (w > g7wmax)
                    g7wmax = w;
            }
        }
        emit("G7_throat_series_max_abs", g7);
        emit("G7_restricted_max_abs", g7r);
        emit("G7_W_max_in_d0", wmax);
        // the production split sits exactly AT the cut, so compare with a relative
        // tolerance or every production layout reports itself out of range
        emit("G7_valid_range_exceeded", wmax > W_CUT * (1.0 + 1e-9) ? 1 : 0);
        emit("G7_npts_used", g7n);
        emit("G7_W_max_used", g7wmax);
    }

    std::cout << "# done\n";
    return 0;
}

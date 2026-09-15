/*
 * theta2d_main.cpp -- A0, second rung: the ANGULAR direction, on a genuinely
 * theta-dependent manufactured field.
 *
 * WHY THIS EXISTS.  backbone2d reproduces Trumpet1d's Phase-1 battery in 2-D,
 * but the backbone is spherically symmetric, so on a polar grid the imported
 * field is theta-independent BY CONSTRUCTION.  Nothing in that rung exercises
 * a theta-derivative, the angular basis, the parity handling or the axis
 * regularisation.  The der_abs(1) finding is the precedent: a 2-D-specific
 * operator error, invisible in 1-D, caught only by a 2-D-specific test.
 *
 * THE MANUFACTURED FIELD.  F(r,theta) = A(r) f(theta), with f chosen so that
 * its EXACT spectral coefficients are known in closed form:
 *
 *   even class (COS_EVEN, basis cos 2k.theta -- the scalars):
 *       f_e = sum_k c_k cos(2k theta),   c_0 = 1/(1-q^2),  c_k = 2 q^k/(1-q^2)
 *           = 1 / (1 - 2q cos 2theta + q^2)
 *   odd class  (COS_ODD,  basis cos(2k+1)theta -- beta^theta, Qbar):
 *       f_o = sum_k q^k cos((2k+1) theta)
 *           = (1-q) cos theta / (1 - 2q cos 2theta + q^2)
 *
 * Both have coefficients decaying EXACTLY as q^k, so the convergence rate is
 * PREDICTED, not merely observed to be fast.  The series are summed directly
 * to machine precision (KMAX terms, q^KMAX < 1e-20) rather than differentiated
 * by hand, so f, f' and f'' carry no algebra risk.
 *
 * GENERIC WITHIN THE PARITY CONSTRAINTS (round 77's lesson: satisfying the
 * letter of "generic" can still land where the physics cannot be):
 *   - every k is present, with comparable magnitudes -- a field built from
 *     cos 4k.theta alone would sit on a sub-lattice and hide any bug in the
 *     odd-k coefficients;
 *   - not symmetric under theta -> pi/2 - theta, so an axis/equator-asymmetric
 *     bug cannot cancel;
 *   - f_e(0) and f_e(pi/2) are both nonzero, so no accidental vanishing;
 *   - f_o(pi/2) = 0 and f_o(0) != 0, which is the parity the odd class must
 *     have and is checked rather than assumed.
 *
 * THE RADIAL FACTOR A(r) = 1/(1 + (r/L)^2), L = 2M, decays like r^-2 so it is
 * representable in the compact domain's variable.  It is NOT exact in
 * Chebyshev, so its truncation error is the floor of the whole sweep -- which
 * is why the same battery is also run at f == 1 and reported as FLOOR_*.  An
 * algebra slip in A' or A'' would show up there as a non-converging constant.
 *
 * WHAT IS MEASURED, per ntheta:
 *   interp  Scalar::val_point            the polar basis + summation
 *   dt      Domain::dt                   d/dtheta
 *   ddt     Domain::dt twice             d^2/dtheta^2 -- the order the
 *                                        equations actually use
 *   lap     Domain::laplacian(.,0)       d_rr + (2/r)d_r + (1/r^2)(d_tt +
 *                                        cot.theta d_t): the load-bearing
 *                                        operator, and the ONLY one that goes
 *                                        through mult_cos_theta/div_sin_theta,
 *                                        i.e. through the axis regularisation
 *   drho,dz der_abs(1), der_abs(2)       closes the round-89 loop: that round
 *                                        learned what der_abs(1) is NOT
 *   coef    val_boundary(.,.,pcf(1)=k)   the angular coefficient extraction and
 *                                        its normalisation, against exact c_k
 *
 * Plus two parity measurements and one control:
 *   PAR_odd_equator   |f_o| at the equator collocation point      (must be 0)
 *   PAR_even_dt_axis  |dt f_e| at axis and equator                (must be 0)
 *   XPAR_*            the odd function declared with the EVEN base: the
 *                     projection must destroy it, and this says by how much.
 *
 * Usage:  theta2d <table.dat> [--ntheta N] [--q Q]
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"

#include "space/space_polar_trumpet.hpp"
#include "manufactured.hpp"
#include "Trumpet1d/src/table_io.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Kadath::Dim_array;
using Kadath::Index;
using Kadath::Point;
using Kadath::Scalar;
using Kadath::Val_domain;
using TrumpetIO::Table;
using TrumpetIO::read_table;

namespace
{

using Trumpet::Angular;
using Trumpet::Chan;
using Trumpet::Radial;
using Trumpet::emit;

struct Errs {
    Chan interp, dt, ddt, lap, drho, dz;
};

void maxeq(double& a, double b) { if (b > a) a = b; }

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "usage: theta2d <table.dat> [--ntheta N] [--q Q] [--L LfacM]\n";
        return 2;
    }
    int ntheta = 9;
    double q = 0.4, Lfac = 2.0;
    bool uselog = false;
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--ntheta" && i + 1 < argc)
            ntheta = std::atoi(argv[++i]);
        else if (a == "--q" && i + 1 < argc)
            q = std::atof(argv[++i]);
        else if (a == "--log")
            uselog = true;   // Domain_polar_shell_log
        else if (a == "--L" && i + 1 < argc)
            Lfac = std::atof(argv[++i]);   // A's scale, in units of M
    }

    Table t;
    try {
        t = read_table(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
    const int ndom = static_cast<int>(t.doms.size());
    const double M = t.M;
    const Radial A{Lfac * M};

    std::vector<double> bounds;
    std::vector<Dim_array> res;
    for (int d = 0; d < ndom; d++) {
        bounds.push_back(t.doms[d].r_int);
        Dim_array n(2);
        n.set(0) = t.doms[d].nbr;
        n.set(1) = ntheta;
        res.push_back(n);
    }
    Point center(2);
    center.set(1) = 0.0;
    center.set(2) = 0.0;
    Trumpet::Space_polar_trumpet space(CHEB_TYPE, center, res, bounds,
                                      std::vector<bool>(bounds.size(), uselog));

    std::cout << "# theta rung  ntheta=" << ntheta << "  q=" << q << "  ndom=" << ndom
              << "  M=" << M << "\n";
    emit("TH_ntheta", ntheta);
    emit("TH_q", q);
    emit("TH_requested_log", uselog);
    {   // the mapping ACTUALLY built, per shell, read off the objects
        int nlog = 0;
        for (int d = 0; d < ndom - 1; d++)
            if (dynamic_cast<const Kadath::Domain_polar_shell_log*>(space.get_domain(d)))
                nlog++;
        emit("TH_log_shells_built", nlog);
        emit("TH_shells", ndom - 1);
    }
    emit("TH_L_over_M", Lfac);

    // --------------------------------------------------------------- battery --
    // kind 0 even (std_base), 1 odd (std_anti_base), 2 flat (the radial floor),
    // 3 the wrong-parity control: the ODD function declared with the EVEN base.
    static const char* tag[4] = {"EVEN", "ODD", "FLOOR", "XPAR"};
    for (int kind = 0; kind < 4; kind++) {
        const Angular f{(kind == 3) ? 1 : kind, q};

        Scalar F(space);
        for (int d = 0; d < ndom; d++) {
            Val_domain& vd = F.set_domain(d);
            vd.allocate_conf();
            const Kadath::Domain* dom = space.get_domain(d);
            Val_domain rad = dom->get_radius();
            Index idx(dom->get_nbr_points());
            do {
                const double rr = rad(idx);
                const double th = dom->get_coloc(2)(idx(1));
                vd.set(idx) = A.d(0, rr) * f.d(0, th);
            } while (idx.inc());
        }
        if (kind == 1)
            F.std_anti_base();
        else
            F.std_base();

        Errs e;
        for (int d = 0; d < ndom; d++) {
            Chan lapd;
            const Kadath::Domain* dom = space.get_domain(d);
            Val_domain rad = dom->get_radius();
            Val_domain vdt = dom->dt(F(d));
            Val_domain vddt = dom->dt(vdt);
            Val_domain vlap = dom->laplacian(F(d), 0);
            Val_domain vrho = F(d).der_abs(1);
            Val_domain vz = F(d).der_abs(2);

            Index idx(dom->get_nbr_points());
            do {
                const double rr = rad(idx);
                if (!std::isfinite(rr))
                    continue;             // r = infinity on the compact outer face
                const double th = dom->get_coloc(2)(idx(1));
                const double st = std::sin(th), ct = std::cos(th);
                const double f0 = f.d(0, th), f1 = f.d(1, th), f2 = f.d(2, th);
                const double a0 = A.d(0, rr), a1 = A.d(1, rr), a2 = A.d(2, rr);

                e.dt.add(vdt(idx), a0 * f1);
                e.ddt.add(vddt(idx), a0 * f2);

                // cot(theta) f' is 0/0 on the axis; the limit is f''(0).
                const double cotf1 = (idx(1) == 0) ? f2 : (ct / st) * f1;
                const double lex = a2 * f0 + 2.0 * a1 * f0 / rr
                                   + a0 * (f2 + cotf1) / (rr * rr);
                e.lap.add(vlap(idx), lex);
                lapd.add(vlap(idx), lex);

                // d/drho = sin.d_r + (cos/r) d_theta ; d/dz = cos.d_r - (sin/r) d_theta
                e.drho.add(vrho(idx), st * a1 * f0 + ct * a0 * f1 / rr);
                e.dz.add(vz(idx), ct * a1 * f0 - st * a0 * f1 / rr);
            } while (idx.inc());
            // per-domain, so a floor can be attributed to a domain rather than
            // reported as a single opaque number.
            emit(std::string(tag[kind]) + "_lap_d" + std::to_string(d), lapd.rel());
        }

        // ---- val_point, at genuinely off-grid (r, theta), in ABSOLUTE (rho,z) --
        // A0 FINDING: Point for val_point is (rho, z) CARTESIAN, not (r, theta) --
        // absol_to_num takes air = sqrt(abs(1)^2 + abs(2)^2).  backbone2d's G3
        // passed (r, theta) and so sampled radius sqrt(r^2+theta^2); it did not
        // fail because a2 - W^2 is zero at EVERY radius.  Here it would fail.
        for (int d = 0; d < ndom; d++) {
            const double ra = t.doms[d].r_int;
            double rb = t.doms[d].r_ext;
            if (!std::isfinite(rb))
                rb = ra * 4.0;
            for (int k = 1; k <= 6; k++) {
                const double rr = ra + (rb - ra) * (double(k) - 0.5) / 6.0;
                for (int m = 0; m < 5; m++) {
                    const double th = M_PI_2 * (double(m) + 0.37) / 5.0;
                    Point pt(2);
                    pt.set(1) = rr * std::sin(th);
                    pt.set(2) = rr * std::cos(th);
                    e.interp.add(F.val_point(pt), A.d(0, rr) * f.d(0, th));
                }
            }
        }

        const std::string p = std::string(tag[kind]) + "_";
        emit(p + "interp", e.interp.rel());
        emit(p + "dt", e.dt.rel());
        emit(p + "ddt", e.ddt.rel());
        emit(p + "lap", e.lap.rel());
        emit(p + "drho", e.drho.rel());
        emit(p + "dz", e.dz.rel());
        emit(p + "lap_abs", e.lap.err);
        emit(p + "lap_scale", e.lap.scale);

        // ---- the angular coefficients at the excision boundary, against exact --
        {
            const Kadath::Domain* dom = space.get_domain(0);
            const double ain = A.d(0, t.doms[0].r_int);
            double worst = 0.0;
            const int kn = std::min(ntheta, 8);
            for (int k = 0; k < kn; k++) {
                Index pcf(dom->get_nbr_coefs());
                pcf.set(1) = k;
                const double got = dom->val_boundary(INNER_BC, F(0), pcf);
                const double want = ain * f.coef(k);
                emit(p + "coef_ratio_k" + std::to_string(k),
                     (std::fabs(want) > 1e-290) ? got / want : got);
                if (std::fabs(want) > 1e-290)
                    maxeq(worst, std::fabs(got / want - 1.0));
            }
            emit(p + "coef_ratio_worst_dev", worst);
        }

        // ---- parity, at the collocation points that ARE the axis and equator --
        {
            const Kadath::Domain* dom = space.get_domain(0);
            Val_domain vdt = dom->dt(F(0));
            const Dim_array& np = dom->get_nbr_points();
            double axis = 0, eq = 0, dax = 0, deq = 0;
            Index ia(np), ie(np);
            for (int i = 0; i < np(0); i++) {
                ia.set(0) = i; ia.set(1) = 0;
                ie.set(0) = i; ie.set(1) = np(1) - 1;
                maxeq(axis, std::fabs(F(0)(ia)));
                maxeq(eq, std::fabs(F(0)(ie)));
                maxeq(dax, std::fabs(vdt(ia)));
                maxeq(deq, std::fabs(vdt(ie)));
            }
            emit(p + "par_val_axis", axis);
            emit(p + "par_val_equator", eq);
            emit(p + "par_dt_axis", dax);
            emit(p + "par_dt_equator", deq);
        }
    }

    // ---- CONSTANT-FIELD ANGULAR PROBE (round 97) --------------------------
    // A constant has COS_EVEN coefficients (1, 0, 0, ...) EXACTLY -- no
    // aliasing, no truncation, nothing to approximate.  So any nonzero k >= 1
    // coefficient is a defect in the transform or in the coefficient
    // extraction, with no interpretation needed.  Added because l0_solve2d's
    // compact-domain tails returned the SAME value at every angular index at
    // ntheta = 3, and the alias instrument cannot adjudicate that resolution:
    // with P = 2 the two-term alias formula is not valid, so it reports a large
    // deviation whether or not anything is wrong.
    {
        double worst_shell = 0.0, worst_compact = 0.0;
        for (int d = 0; d < ndom; d++) {
            const Kadath::Domain* dom = space.get_domain(d);
            Val_domain v(dom);
            v.allocate_conf();
            v = 1.0;
            v.std_base();
            const int nk = dom->get_nbr_coefs()(1);
            const int bnd = (d == ndom - 1) ? OUTER_BC : INNER_BC;
            for (int k = 0; k < nk; k++) {
                Index pk(dom->get_nbr_coefs());
                pk.set(1) = k;
                const double c = dom->val_boundary(bnd, v, pk);
                const double err = std::fabs(c - (k == 0 ? 1.0 : 0.0));
                double& w = (d == ndom - 1) ? worst_compact : worst_shell;
                w = std::max(w, err);
                if (d == 0 || d == ndom - 1)
                    emit("CONSTPROBE_d" + std::to_string(d) + "_k"
                             + std::to_string(k), c);
            }
        }
        emit("CONSTPROBE_worst_shell", worst_shell);
        emit("CONSTPROBE_worst_compact", worst_compact);
    }

    std::cout << "# done\n";
    return 0;
}

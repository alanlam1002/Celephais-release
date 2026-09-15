/*
 * floor2d_main.cpp -- A0: what accuracy architecture does this discretisation
 * support?
 *
 * Round 90 measured, on the production layouts, that the relative roundoff floor
 * of the Laplacian on the innermost shell RISES with radial resolution
 * (6.86e-10 -> 2.63e-09 -> 9.42e-09 for nr = 21 -> 25 -> 29 on W005) while the
 * field itself stays resolved to 6e-16.  Combined with the hard ceiling of 33
 * points per direction, that makes domain PLACEMENT the only accuracy lever
 * left, and research (round 279) asked the decisive question:
 *
 *     does the Laplacian floor improve by SUBDIVISION at fixed total dof?
 *
 * If it does not, the accuracy target needs revisiting rather than the layout.
 *
 * This app answers it directly, on nothing but the discretisation: it builds
 * shells over an explicit interval with an explicit (ndom, nr) and measures the
 * floor of the manufactured field's value, first radial derivative, second
 * radial derivative and Laplacian.  No backbone table is involved, because the
 * question is not about the backbone -- it is about what a Chebyshev shell
 * stack can represent.
 *
 * WHAT TO EXPECT, AND WHY IT IS WORTH MEASURING RATHER THAN REASONING.  The
 * layout notes record the floor scaling as (N/alpha)^2.  If that is the whole
 * story, subdivision at fixed total dof is EXACTLY NEUTRAL -- halving N and
 * halving alpha leaves N/alpha unchanged -- and the lever is empty.  But round
 * 90 measured the growth at fixed alpha as roughly N^7.7, far steeper than
 * N^2, so the floor cannot be a function of N/alpha alone and the trade is not
 * decidable from the recorded scaling.
 *
 * Usage:
 *   floor2d --rin A --rout B --ndom D --nr N [--ntheta N] [--q Q] [--L LfacM]
 *           [--geometric]
 *
 * --ndom D gives D shells spanning [rin, rout] PLUS the compact r -> infinity
 * domain the space always terminates in, so D+1 domains in all.
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"

#include "space/space_polar_trumpet.hpp"
#include "manufactured.hpp"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using Kadath::Dim_array;
using Kadath::Index;
using Kadath::Point;
using Kadath::Scalar;
using Kadath::Val_domain;
using Trumpet::Angular;
using Trumpet::Chan;
using Trumpet::emit;
using Trumpet::Radial;

int main(int argc, char** argv)
{
    double rin = 0.15519545203273752;    // W005's excision, so the numbers are
    double rout = 3.9388952741521037;    // comparable with the production layout
    double q = 0.4, Lfac = 2.0;
    int ndom = 4, nr = 21, ntheta = 9, kind = 0, field = 0;
    bool geometric = false, uselog = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto nxt = [&]() { return argv[++i]; };
        if (a == "--rin") rin = std::atof(nxt());
        else if (a == "--rout") rout = std::atof(nxt());
        else if (a == "--ndom") ndom = std::atoi(nxt());
        else if (a == "--nr") nr = std::atoi(nxt());
        else if (a == "--ntheta") ntheta = std::atoi(nxt());
        else if (a == "--q") q = std::atof(nxt());
        else if (a == "--L") Lfac = std::atof(nxt());
        else if (a == "--kind") kind = std::atoi(nxt());   // ANGULAR class
        else if (a == "--field") {                         // RADIAL profile
            const std::string f = nxt();
            field = (f == "power") ? 1 : (f == "massmode") ? 2 : 0;
        }
        else if (a == "--geometric") geometric = true;
        else if (a == "--log") uselog = true;   // Domain_polar_shell_log
        else { std::cerr << "unknown flag " << a << "\n"; return 2; }
    }

    // Shell boundaries: uniform in r, or uniform in ln r (--geometric), which is
    // the shape the production layouts actually use.
    std::vector<double> bounds;
    for (int d = 0; d <= ndom; d++) {
        const double t = double(d) / double(ndom);
        bounds.push_back(geometric ? rin * std::pow(rout / rin, t)
                                   : rin + (rout - rin) * t);
    }
    // ROUND 92 CORRECTION.  Space_polar_trumpet ALWAYS makes its last domain a
    // Domain_polar_compact (space_polar_trumpet.hpp:60) -- it is the r -> inf
    // domain and there is no way to ask for a stack of shells alone.  The first
    // version of this app passed bounds.begin()..end()-1 and carried a
    // --compact flag, so "--ndom D" silently built D-1 shells plus the compact
    // domain and the flag did nothing.  Round 91's grid is unaffected in its
    // conclusions -- every configuration had the same structure and alpha_d0
    // was always the first shell -- but the labels were wrong by one domain.
    // Here --ndom D means D SHELLS spanning [rin, rout], plus the compact
    // domain beyond rout, for D+1 in total.
    std::vector<double> inner(bounds);          // all D+1 boundaries
    std::vector<Dim_array> res;
    for (int d = 0; d <= ndom; d++) {
        Dim_array n(2);
        n.set(0) = nr;
        n.set(1) = ntheta;
        res.push_back(n);
    }
    const int nd = static_cast<int>(inner.size());   // D shells + 1 compact

    Point center(2);
    center.set(1) = 0.0;
    center.set(2) = 0.0;
    Trumpet::Space_polar_trumpet space(CHEB_TYPE, center, res, inner,
                                      std::vector<bool>(inner.size(), uselog));

    const Radial A{Lfac, field};
    const Angular f{kind, q};

    std::cout << "# floor2d  rin=" << rin << " rout=" << rout << " ndom=" << ndom
              << " shells + compact  nr=" << nr
              << " ntheta=" << ntheta << " kind=" << kind
              << (geometric ? " geometric" : " uniform") << "\n";
    emit("F_ndom", nd);
    emit("F_requested_log", uselog);
    emit("F_field", field);
    emit("F_nr", nr);
    emit("F_total_radial_dof", nd * nr);

    Scalar F(space);
    for (int d = 0; d < nd; d++) {
        Val_domain& vd = F.set_domain(d);
        vd.allocate_conf();
        const Kadath::Domain* dom = space.get_domain(d);
        Val_domain rad = dom->get_radius();
        Index idx(dom->get_nbr_points());
        do {
            vd.set(idx) = A.d(0, rad(idx)) * f.d(0, dom->get_coloc(2)(idx(1)));
        } while (idx.inc());
    }
    if (kind == 1)
        F.std_anti_base();
    else
        F.std_base();

    Chan gval, gdr, gddr, glap;
    double worst_lap = 0.0;
    int worst_d = -1;
    for (int d = 0; d < nd; d++) {
        const Kadath::Domain* dom = space.get_domain(d);
        Val_domain rad = dom->get_radius();
        Val_domain d1 = dom->der_r(F(d));
        Val_domain d2 = dom->der_r(d1);
        Val_domain lp = dom->laplacian(F(d), 0);
        Chan cdr, cddr, clap;
        Index idx(dom->get_nbr_points());
        do {
            const double rr = rad(idx);
            if (!std::isfinite(rr))
                continue;                       // r = infinity on the compact face
            const double th = dom->get_coloc(2)(idx(1));
            const double st = std::sin(th), ct = std::cos(th);
            const double f0 = f.d(0, th), f1 = f.d(1, th), f2 = f.d(2, th);
            const double a0 = A.d(0, rr), a1 = A.d(1, rr), a2 = A.d(2, rr);
            const double cotf1 = (idx(1) == 0) ? f2 : (ct / st) * f1;
            // NOT a test: F was filled from the exact values at these very
            // points.  The resolution control is the off-grid val_point below.
            cdr.add(d1(idx), a1 * f0);
            cddr.add(d2(idx), a2 * f0);
            clap.add(lp(idx), A.lap(rr) * f0 + a0 * (f2 + cotf1) / (rr * rr));
            // massmode's exact laplacian is identically zero, so Chan::rel()
            // would fall back to the ABSOLUTE error.  logfloor scales it on
            // max|f''| instead, and the two instruments must use the same
            // normalisation or they differ by 2/r^3 ~ 535 and it reads as a
            // broken domain.
            if (field == 2 && std::fabs(A.d(2, rr)) > clap.scale)
                clap.scale = std::fabs(A.d(2, rr));
        } while (idx.inc());
        // alpha is the domain half-width, the quantity the floor scaling is
        // usually written in terms of; reported so (nr/alpha) can be formed.
        const double a = (d + 1 < static_cast<int>(bounds.size()))
                             ? 0.5 * (bounds[d + 1] - bounds[d]) : 0.0;
        emit("F_alpha_d" + std::to_string(d), a);
        // the mapping ACTUALLY built, read off the object, not off the request
        emit("F_is_log_d" + std::to_string(d),
             dynamic_cast<const Kadath::Domain_polar_shell_log*>(dom) != nullptr);

        emit("F_dr_d" + std::to_string(d), cdr.rel());
        emit("F_ddr_d" + std::to_string(d), cddr.rel());
        emit("F_lap_d" + std::to_string(d), clap.rel());
        // Is the field actually RESOLVED?  If this is at machine precision the
        // floors above are roundoff amplification, not truncation -- the whole
        // reading of the experiment turns on the distinction.
        Chan cint;
        const int nsamp = (d + 1 < static_cast<int>(bounds.size())) ? 8 : 0;
        for (int k = 1; k <= nsamp; k++) {
            const double rr = bounds[d] + (bounds[d + 1] - bounds[d])
                                              * (double(k) - 0.5) / 8.0;
            for (int mth = 0; mth < 3; mth++) {
                const double th = M_PI_2 * (double(mth) + 0.37) / 3.0;
                Point pt(2);
                pt.set(1) = rr * std::sin(th);
                pt.set(2) = rr * std::cos(th);
                cint.add(F.val_point(pt), A.d(0, rr) * f.d(0, th));
            }
        }
        emit("F_val_d" + std::to_string(d), cint.rel());
        gval.err = std::max(gval.err, cint.rel());
        if (clap.rel() > worst_lap) { worst_lap = clap.rel(); worst_d = d; }
        gdr.err = std::max(gdr.err, cdr.rel());
        gddr.err = std::max(gddr.err, cddr.rel());
        glap.err = std::max(glap.err, clap.rel());
    }
    emit("F_val", gval.err);
    emit("F_dr", gdr.err);
    emit("F_ddr", gddr.err);
    emit("F_lap", glap.err);
    emit("F_lap_worst_domain", worst_d);
    std::cout << "# done\n";
    return 0;
}

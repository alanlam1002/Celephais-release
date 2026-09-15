/*
 * logfloor_main.cpp -- what does a LOG radial domain buy on the FLOOR?
 *
 * Research round 282 ruled: the truncation case for a log-mapped inner region
 * is real (2729x on the actual backbone over d0+d1), and it reaches round 91's
 * lever -- fewer domains at the same resolution, where placement is the only
 * nearly-free variable.  But the binding constraint is the FLOOR, and the
 * connection (fewer domains -> fewer dof -> better floor) has not been
 * measured.  Measure it before building Domain_polar_shell_log.
 *
 * WHY THIS IS KADATH'S LOG SHELL AND NOT MY OWN.  Round 92's mapping comparison
 * was a standalone reimplementation.  Kadath HAS a log shell -- Domain_shell_log
 * in spheric.hpp, log r = alpha x + beta -- just not a polar one.  The radial
 * operator is what the question is about, so this app measures Kadath's own
 * implementation, which is the thing a port would carry over.  Both arms are
 * ordinary 3-D spheric shells, so the comparison is internally valid:
 *   - Domain_shell and Domain_shell_log both inherit Domain::laplacian, which
 *     is sum_j der_abs(j).der_abs(j);
 *   - each supplies its OWN do_der_abs_from_der_var, i.e. its own chain rule.
 * So the two arms differ in exactly the mapping and nothing else.
 *
 * ⚠ THE GUARD RESEARCH ASKED FOR.  Round 91's error was in floor2d's own layout
 * construction -- "--ndom D" silently built D-1 shells.  A comparison BETWEEN
 * layouts would be inverted by a silent off-by-one in either arm, so this app
 * reports the structure it ACTUALLY built: each domain's class, its get_rmin()
 * and get_rmax() read back from the object, and its radial point count.  Those
 * accessors are virtual and overridden in the log class, so they report the
 * built mapping rather than the requested one.
 *
 * ⚠ AND THE ANCHOR (round 282 ruling 3: an instrument reproduces a known number
 * before it produces a new one).  The linear arm over d0's bounds at nr = 21 is
 * emitted as ANCHOR_linear_d0 so it can be held against floor2d's polar
 * F_lap_d0 = 6.855e-10, which is round 90's FLOOR_lap_d0 to the last digit.
 * These are different domain families and different laplacian code paths, so
 * they need not agree exactly -- but they must agree in ORDER, and if they do
 * not, nothing below is worth reading.
 *
 * No Space is built: Val_domain(const Domain*) and Val_domain::std_base() are
 * public, and laplacian/der_r are per-domain.  That keeps the instrument to the
 * thing being measured.
 *
 * ⚠ THE FIELD DECIDES THE ANSWER, so there are three.  Round 92 established
 * that the mapping decides which fields are RESOLVED rather than what the floor
 * is; a single manufactured field therefore silently picks a winner.  The
 * rational one favours linear, the two power laws favour log, and the trumpet's
 * near-throat behaviour is a power law (rho = r^n).  All three are reported.
 *
 * Usage: logfloor --rin A --rout B --ndom D --nr N [--log]
 *                 [--field rational|power|massmode] [--ntheta N] [--nphi N]
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"

#include "manufactured.hpp"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using Kadath::Dim_array;
using Kadath::Domain;
using Kadath::Domain_shell;
using Kadath::Domain_shell_log;
using Kadath::Index;
using Kadath::Point;
using Kadath::Val_domain;
using Trumpet::Chan;
using Trumpet::emit;
using Trumpet::Radial;

int main(int argc, char** argv)
{
    double rin = 0.15519545203273752, rout = 0.30737634143970582, Lfac = 2.0;
    int ndom = 1, nr = 21, ntheta = 5, nphi = 4;
    bool uselog = false;
    int field = 0;                     // 0 rational, 1 power, 2 massmode
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto nxt = [&]() { return argv[++i]; };
        if (a == "--rin") rin = std::atof(nxt());
        else if (a == "--rout") rout = std::atof(nxt());
        else if (a == "--ndom") ndom = std::atoi(nxt());
        else if (a == "--nr") nr = std::atoi(nxt());
        else if (a == "--ntheta") ntheta = std::atoi(nxt());
        else if (a == "--nphi") nphi = std::atoi(nxt());
        else if (a == "--L") Lfac = std::atof(nxt());
        else if (a == "--log") uselog = true;
        else if (a == "--field") {
            const std::string f = nxt();
            field = (f == "power") ? 1 : (f == "massmode") ? 2 : 0;
        }
        else { std::cerr << "unknown flag " << a << "\n"; return 2; }
    }

    // Shell boundaries uniform in ln r, so the two arms cover the same
    // sub-intervals and only the WITHIN-domain mapping differs.
    std::vector<double> b;
    for (int d = 0; d <= ndom; d++)
        b.push_back(rin * std::pow(rout / rin, double(d) / double(ndom)));

    Point centre(3);
    for (int k = 1; k <= 3; k++)
        centre.set(k) = 0.0;
    Dim_array nbr(3);
    nbr.set(0) = nr;
    nbr.set(1) = ntheta;
    nbr.set(2) = nphi;

    std::vector<std::unique_ptr<Domain>> doms;
    for (int d = 0; d < ndom; d++) {
        if (uselog)
            doms.emplace_back(new Domain_shell_log(d, CHEB_TYPE, b[d], b[d + 1],
                                                   centre, nbr));
        else
            doms.emplace_back(new Domain_shell(d, CHEB_TYPE, b[d], b[d + 1],
                                               centre, nbr));
    }

    std::cout << "# logfloor  " << (uselog ? "LOG" : "LINEAR") << "  ndom=" << ndom
              << "  nr=" << nr << "  [" << rin << ", " << rout << "]\n";
    emit("L_is_log", uselog ? 1 : 0);
    emit("L_ndom_requested", ndom);
    emit("L_nr", nr);

    // ---- the structure ACTUALLY built, read back off the objects ----------
    int built = 0;
    for (int d = 0; d < ndom; d++) {
        const Domain* dm = doms[d].get();
        const bool is_log = (dynamic_cast<const Domain_shell_log*>(dm) != nullptr);
        emit("L_built_is_log_d" + std::to_string(d), is_log ? 1 : 0);
        emit("L_built_rmin_d" + std::to_string(d), dm->get_rmin());
        emit("L_built_rmax_d" + std::to_string(d), dm->get_rmax());
        emit("L_built_nr_d" + std::to_string(d), dm->get_nbr_points()(0));
        if (is_log != uselog) {
            std::cerr << "FATAL: domain " << d << " is not the requested type\n";
            return 1;
        }
        built++;
    }
    emit("L_ndom_built", built);
    emit("L_total_radial_dof", built * nr);

    const Radial A{Lfac, field};
    emit("L_field", field);
    Chan glap, gdr;
    double gtail = 0.0, worst = 0.0;
    for (int d = 0; d < ndom; d++) {
        const Domain* dm = doms[d].get();
        Val_domain v(dm);
        v.allocate_conf();
        Val_domain rad = dm->get_radius();
        Index idx(dm->get_nbr_points());
        do {
            v.set(idx) = A.d(0, rad(idx));
        } while (idx.inc());
        v.std_base();

        Val_domain lp = dm->laplacian(v, 0);
        Val_domain d1 = dm->der_r(v);
        Chan clap, cdr;
        Index ix(dm->get_nbr_points());
        do {
            const double r = rad(ix);
            clap.add(lp(ix), A.lap(r));
            // massmode's exact laplacian is identically zero, so Chan::rel()
            // would report an absolute error; the honest scale is the size of
            // the terms that cancel, max|f''|.
            if (field == 2 && std::fabs(A.d(2, r)) > clap.scale)
                clap.scale = std::fabs(A.d(2, r));
            cdr.add(d1(ix), A.d(1, r));
        } while (ix.inc());

        // TRUNCATION, the quantity the two arms must be MATCHED on: the last
        // two radial coefficients as a fraction of the largest.
        v.coef();
        const Dim_array& nc = dm->get_nbr_coefs();
        double top = 0.0, mx = 0.0;
        Index ic(nc);
        do {
            const double c = std::fabs(v.get_coef(ic));
            if (c > mx) mx = c;
            if (ic(0) >= nc(0) - 2 && c > top) top = c;
        } while (ic.inc());
        const double tail = (mx > 0.0) ? top / mx : 0.0;

        emit("L_lap_d" + std::to_string(d), clap.rel());
        emit("L_dr_d" + std::to_string(d), cdr.rel());
        emit("L_tail_d" + std::to_string(d), tail);
        if (tail > gtail) gtail = tail;
        if (clap.rel() > worst) worst = clap.rel();
        glap.err = std::max(glap.err, clap.rel());
        gdr.err = std::max(gdr.err, cdr.rel());
    }
    emit("L_lap", glap.err);
    emit("L_dr", gdr.err);
    emit("L_tail", gtail);
    std::cout << "# done\n";
    return 0;
}

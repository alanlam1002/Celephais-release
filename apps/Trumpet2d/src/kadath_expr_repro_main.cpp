/*
 * kadath_expr_repro_main.cpp -- a MINIMAL reproducer, built by hand.
 *
 * Round 105 claimed two defects in Kadath's expression evaluation.  That is a
 * strong claim and "I found a bug in the library" is what a tired investigator
 * concludes one step early, so it has to be discharged rather than carried.
 *
 * This file depends on NOTHING of ours.  Stock Kadath only: Space_polar, two
 * add_cst Scalars with simple analytic content, and expressions typed out by
 * hand.  No generated text, no emitter, no Trumpet space, no backbone table.
 *
 * ⚠ The interesting outcome is the NEGATIVE one.  If a hand-built two-field
 * case does NOT show the defect, the fault is in something the emitter does and
 * not in the library -- which is a more useful result than a confirmation, and
 * is why every pair below is a spelling of ONE expression with the two
 * spellings printed side by side rather than a single number to eyeball.
 */

#include <mpi.h>

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace Kadath;

static int g_fail = 0;

/** Register one expression and return its value at a fixed point. */
static double val(System_of_eqs& syst, const Space& sp, const std::string& def,
                  int dom, const Index& ix)
{
    syst.add_def(def.c_str());
    const std::string nm = def.substr(0, def.find(' '));
    return syst.give_val_def_scalar_domain(nm.c_str(), dom)(ix);
}

/** Two spellings of ONE expression.  They must agree. */
static void check_pair(System_of_eqs& syst, const Space& sp, int dom, const Index& ix,
                 const char* what, const std::string& a, const std::string& b)
{
    const double va = val(syst, sp, a, dom, ix);
    const double vb = val(syst, sp, b, dom, ix);
    const double sc = std::max(std::max(std::fabs(va), std::fabs(vb)), 1e-30);
    const double rel = std::fabs(va - vb) / sc;
    const bool bad = rel > 1e-10;
    if (bad)
        g_fail++;
    std::cout << "  " << std::left << std::setw(34) << what
              << std::setprecision(17) << std::setw(26) << va
              << std::setw(26) << vb << std::setprecision(3) << rel
              << (bad ? "   <-- DIFFER" : "") << "\n";
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    const int nr = 13, nt = 9;
    Dim_array res(2);
    res.set(0) = nr;
    res.set(1) = nt;
    Point cr(2);
    cr.set(1) = 0.0;
    cr.set(2) = 0.0;
    Array<double> bounds(2);
    bounds.set(0) = 1.0;
    bounds.set(1) = 2.0;
    Space_polar space(CHEB_TYPE, cr, res, bounds);
    const int dom = 1;                       // the shell; domain 0 is the nucleus

    // Two add_cst fields, both representable: polynomial in r, COS_EVEN in theta.
    Scalar F(space), G(space);
    for (int d = 0; d < space.get_nbr_domains(); d++) {
        const Domain* dm = space.get_domain(d);
        Val_domain& vf = F.set_domain(d);
        Val_domain& vg = G.set_domain(d);
        vf.allocate_conf();
        vg.allocate_conf();
        Index idx(dm->get_nbr_points());
        do {
            const double rr = dm->get_radius()(idx);
            const double th = dm->get_coloc(2)(idx(1));
            vf.set(idx) = 2.0 + 0.5 * rr * rr + 0.3 * rr * std::cos(2.0 * th);
            vg.set(idx) = 1.0 + 0.7 * rr + 0.4 * rr * rr * std::cos(2.0 * th);
        } while (idx.inc());
    }
    F.std_base();
    G.std_base();

    System_of_eqs syst(space, dom, dom);
    syst.add_cst("F", F);
    syst.add_cst("G", G);

    syst.add_def("FG = F + G");

    Index ix(space.get_domain(dom)->get_nbr_points());
    for (int k = 0; k < 3 + 2 * nr; k++)
        ix.inc();                            // an interior, off-axis point

    std::cout << "MINIMAL REPRODUCER -- stock Space_polar, two add_cst fields,\n"
                 "every expression typed by hand.  Each row is ONE expression in\n"
                 "two spellings; they must agree.\n\n";
    std::cout << "  " << std::left << std::setw(34) << "what"
              << std::setw(26) << "spelling A" << std::setw(26) << "spelling B"
              << "rel\n";

    // --- DEFECT 1, narrowed.
    //
    // ⚠ Every row must compare two GENUINELY DIFFERENT spellings.  My first
    // draft had two rows comparing a string with itself -- vacuous checks, in
    // the file whose whole job is rigour.  The variable isolated here is
    // whether an operator applied to an INLINE sum equals the same operator
    // applied to that sum through a NAME.
    val(syst, space, "S  = dt(G) + divr(F)", dom, ix);        // the sum, NAMED
    val(syst, space, "A1 = multr(dr(F))", dom, ix);
    val(syst, space, "A2 = multr(S)", dom, ix);               // op(named sum)
    val(syst, space, "A3 = multr(divr(F))", dom, ix);

    check_pair(syst, space, dom, ix, "op(inline sum) vs op(named sum)",
         "B1 = multr(dt(G) + divr(F))", "B2 = A2");
    check_pair(syst, space, dom, ix, "  same for divr",
         "B3 = divr(dt(G) + divr(F))", "B4 = divr(S)");
    check_pair(syst, space, dom, ix, "  same for dr",
         "B5 = dr(dt(G) + divr(F))", "B6 = dr(S)");
    check_pair(syst, space, dom, ix, "  same for dt",
         "B7 = dt(dt(G) + divr(F))", "B8 = dt(S)");
    check_pair(syst, space, dom, ix, "  argument a sum of PLAIN fields",
         "B9 = multr(F + G)", "B10 = multr(FG)");
    check_pair(syst, space, dom, ix, "  argument NOT a sum (control)",
         "B11 = multr(divr(F))", "B12 = A3");

    // and the round-105 form, which is the above inside an outer sum
    check_pair(syst, space, dom, ix, "op(sum) + op(sum), both inline",
         "B13 = multr(dr(F)) + multr(dt(G) + divr(F))", "B14 = A1 + A2");
    check_pair(syst, space, dom, ix, "  field + op(inline sum)",
         "B15 = F + multr(dt(G) + divr(F))", "B16 = F + A2");
    check_pair(syst, space, dom, ix, "  op + op, NEITHER arg a sum",
         "B17 = multr(dr(F)) + multr(divr(F))", "B18 = A1 + A3");
    check_pair(syst, space, dom, ix, "  op + op(NAMED sum)",
         "B19 = multr(dr(F)) + multr(S)", "B20 = A1 + A2");
    check_pair(syst, space, dom, ix, "  NAMED op + op(inline sum)",
         "B21 = A1 + multr(dt(G) + divr(F))", "B22 = A1 + A2");
    check_pair(syst, space, dom, ix, "  op(inline sum) + field",
         "B23 = multr(dt(G) + divr(F)) + F", "B24 = A2 + F");
    check_pair(syst, space, dom, ix, "  op + op, one arg a PLAIN-field sum",
         "B25 = multr(dr(F)) + multr(F + G)", "B26 = A1 + multr(FG)");

    // --- round 105's DEFECT 2: a named pair vs the same expression inline.
    // The interesting outcome here is the NEGATIVE one.
    val(syst, space, "A4 = dr(log(F))", dom, ix);
    val(syst, space, "A5 = 4 * dr(log(G))", dom, ix);
    val(syst, space, "A6 = dr(F)", dom, ix);
    val(syst, space, "A7 = 4 * dr(G)", dom, ix);
    check_pair(syst, space, dom, ix, "defect 2: named vs inline, with log",
         "C1 = A4 - A5", "C2 = dr(log(F)) - 4 * dr(log(G))");
    check_pair(syst, space, dom, ix, "  same, no log",
         "C3 = A6 - A7", "C4 = dr(F) - 4 * dr(G)");
    check_pair(syst, space, dom, ix, "  named difference, plain fields",
         "C5 = F - G", "C6 = FG - G - G");

    std::cout << "\n  pairs that DIFFER: " << g_fail << "\n";
    std::cout << "RESULT REPRO_pairs_differing " << g_fail << "\n";
    MPI_Finalize();
    return 0;
}

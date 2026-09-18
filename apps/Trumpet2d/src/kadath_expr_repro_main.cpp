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
    Scalar F(space), G(space), CA(space), CB(space), CC(space), CD(space), CE(space);
    for (int d = 0; d < space.get_nbr_domains(); d++) {
        const Domain* dm = space.get_domain(d);
        Val_domain& vf = F.set_domain(d);
        Val_domain& vg = G.set_domain(d);
        Val_domain& va = CA.set_domain(d); Val_domain& vb = CB.set_domain(d);
        Val_domain& vc = CC.set_domain(d); Val_domain& vd = CD.set_domain(d);
        Val_domain& ve = CE.set_domain(d);
        vf.allocate_conf();
        vg.allocate_conf();
        for (Val_domain* v : {&va, &vb, &vc, &vd, &ve}) v->allocate_conf();
        Index idx(dm->get_nbr_points());
        do {
            const double rr = dm->get_radius()(idx);
            const double th = dm->get_coloc(2)(idx(1));
            vf.set(idx) = 2.0 + 0.5 * rr * rr + 0.3 * rr * std::cos(2.0 * th);
            vg.set(idx) = 1.0 + 0.7 * rr + 0.4 * rr * rr * std::cos(2.0 * th);
            // coefficient fields, the shape l0_setup.hpp's 39 csts have
            va.set(idx) = 1.3 + 0.2 * rr;
            vb.set(idx) = -0.7 + 0.5 * rr * rr;
            vc.set(idx) = 0.9 - 0.3 * rr;
            vd.set(idx) = 1.1 + 0.6 * rr * std::cos(2.0 * th);
            ve.set(idx) = -1.4 + 0.8 * rr;
        } while (idx.inc());
    }
    for (Scalar* z : {&CA, &CB, &CC, &CD, &CE}) z->std_base();
    F.std_base();
    G.std_base();

    System_of_eqs syst(space, dom, dom);
    syst.add_cst("F", F);
    syst.add_cst("G", G);
    syst.add_cst("CA", CA); syst.add_cst("CB", CB); syst.add_cst("CC", CC);
    syst.add_cst("CD", CD); syst.add_cst("CE", CE);

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

    // --- ITEM 1: does l0_setup.hpp's ROW SHAPE present the trigger? ---------
    // A row there is  c1*G + c2*dr(G) + c3*Q + c4*dr(Q) + ... + c8*jsrc :
    // a SUM OF PRODUCTS whose first summand is a product of two plain fields.
    // That is NOT the shape the trigger was found on, so it is measured rather
    // than read off the source.
    val(syst, space, "R1 = CA * F", dom, ix);
    val(syst, space, "R2 = CB * dr(F)", dom, ix);
    val(syst, space, "R3 = CC * ddr(F)", dom, ix);
    val(syst, space, "R4 = CD * G", dom, ix);
    val(syst, space, "R5 = CE * dr(G)", dom, ix);
    check_pair(syst, space, dom, ix, "l0 row shape, 5 terms",
         "R6 = CA * F + CB * dr(F) + CC * ddr(F) + CD * G + CE * dr(G)",
         "R7 = R1 + R2 + R3 + R4 + R5");
    check_pair(syst, space, dom, ix, "  same, DERIVATIVE term first",
         "R8 = CB * dr(F) + CA * F + CC * ddr(F) + CD * G + CE * dr(G)",
         "R9 = R2 + R1 + R3 + R4 + R5");
    check_pair(syst, space, dom, ix, "  bare op first, no coefficient",
         "R10 = dr(F) + CA * F + CC * ddr(F)",
         "R11 = dr(F) + R1 + R3");
    check_pair(syst, space, dom, ix, "  product of two plain fields first",
         "R12 = CA * F + dr(F) + CC * ddr(F)",
         "R13 = R1 + dr(F) + R3");

    // --- DEFECT 2, and the variable round 106 never varied -----------------
    //
    // ⚠ Every named def above was created through val(), which READS it back
    // immediately -- and reading a def indexes its Val_domain, which forces it
    // into configuration space.  Our emitter does not do that: it registers 309
    // sub-defs and reads nothing until the end.  So the whole table above was
    // measured in ONE state, and it is not the state the emitter runs in.
    //
    // The variable here is therefore not the spelling but whether an operand
    // has been READ before the sum that uses it is registered.  Three
    // spellings of one expression; the inline form is the reference.
    {
        auto reg = [&](const char* d) { syst.add_def(d); };
        auto rd = [&](const char* nm) {
            return syst.give_val_def_scalar_domain(nm, dom)(ix);
        };
        std::cout << "\n  DEFECT 2 -- the operand-read variable\n";
        std::cout << "  " << std::left << std::setw(32) << "what"
                  << std::setw(20) << "a+b (truth)" << std::setw(20) << "unread"
                  << std::setw(20) << "re-read" << std::setw(20) << "read first"
                  << std::setw(20) << "inline" << "\n";
        struct Case { const char* what; const char* t1; const char* t2; };
        const std::vector<Case> cs = {
            {"dt(G) + divr(F)", "dt(G)", "divr(F)"},
            {"the emitter's own failing pair",
             "-1 * (-1 * (multr(dr(F))))", "-1 * (multr(dt(G) + divr(F)))"},
            {"both operands plain", "F", "G"},
        };
        int k = 0;
        for (const auto& c : cs) {
            const std::string n = std::to_string(k++);
            auto R = [&](const std::string& nm) { return rd(nm.c_str()); };
            // (i) the sum registered while its operands have NEVER been read
            reg((std::string("U") + n + "a = " + c.t1).c_str());
            reg((std::string("U") + n + "b = " + c.t2).c_str());
            reg((std::string("U") + n + "s = U" + n + "a + U" + n + "b").c_str());
            const double unread = R("U" + n + "s");
            // ⚠ the TRUTH for a pointwise sum is the two operands' own values,
            // read back individually.  Without it the columns only compare
            // Kadath with Kadath and none of them is a reference.
            const double a = R("U" + n + "a"), b = R("U" + n + "b");
            const double reread = R("U" + n + "s");   // same def, read again
            // (ii) the same three defs, operands read BEFORE the sum is parsed
            reg((std::string("V") + n + "a = " + c.t1).c_str());
            reg((std::string("V") + n + "b = " + c.t2).c_str());
            R("V" + n + "a");
            R("V" + n + "b");
            reg((std::string("V") + n + "s = V" + n + "a + V" + n + "b").c_str());
            const double readfirst = R("V" + n + "s");
            // (iii) the inline spelling
            reg((std::string("W") + n + "s = (" + c.t1 + ") + (" + c.t2 + ")").c_str());
            const double inl = R("W" + n + "s");
            const double truth = a + b;
            const double sc = std::max(std::fabs(truth), 1e-30);
            auto off = [&](double v) { return std::fabs(v - truth) / sc; };
            const bool bad = off(unread) > 1e-10 || off(readfirst) > 1e-10
                             || off(inl) > 1e-10 || off(reread) > 1e-10;
            if (bad)
                g_fail++;
            std::cout << "  " << std::left << std::setw(32) << c.what
                      << std::setprecision(12)
                      << std::setw(20) << truth << std::setw(20) << unread
                      << std::setw(20) << reread << std::setw(20) << readfirst
                      << std::setw(20) << inl << (bad ? "  <-- DIFFER" : "")
                      << "\n";
        }
        // ⚠ A PREDICTION, tested rather than asserted.  If a sum is evaluated in
        // the space of its FIRST operand, then reading only the first should
        // fix it and reading only the second should not -- which is also what
        // round 106 saw and mis-attributed to NAMING the first operand.
        {
            const char* t1 = "-1 * (-1 * (multr(dr(F))))";
            const char* t2 = "-1 * (multr(dt(G) + divr(F)))";
            struct P { const char* what; bool r1, r2; };
            for (const auto& pz : std::vector<P>{{"read neither", false, false},
                                                 {"read FIRST only", true, false},
                                                 {"read SECOND only", false, true},
                                                 {"read both", true, true}}) {
                static int t = 0;
                const std::string n = "P" + std::to_string(t++);
                reg((n + "a = " + t1).c_str());
                reg((n + "b = " + t2).c_str());
                if (pz.r1) rd((n + "a").c_str());
                if (pz.r2) rd((n + "b").c_str());
                reg((n + "s = " + n + "a + " + n + "b").c_str());
                const double v = rd((n + "s").c_str());
                const double a = rd((n + "a").c_str()), b = rd((n + "b").c_str());
                const bool bad = std::fabs(v - (a + b)) / std::max(std::fabs(a + b), 1e-30) > 1e-10;
                if (bad)
                    g_fail++;
                std::cout << "  " << std::left << std::setw(32) << pz.what
                          << std::setprecision(12) << std::setw(20) << (a + b)
                          << std::setw(20) << v << (bad ? "  <-- DIFFER" : "")
                          << "\n";
            }
        }
        std::cout << "  (columns: a+b read back | sum parsed with operands unread |\n"
                     "   the SAME sum re-read after the operands were read |\n"
                     "   sum parsed after the operands were read | inline)\n";
    }

    std::cout << "\n  pairs that DIFFER: " << g_fail << "\n";
    std::cout << "RESULT REPRO_pairs_differing " << g_fail << "\n";
    MPI_Finalize();
    return 0;
}

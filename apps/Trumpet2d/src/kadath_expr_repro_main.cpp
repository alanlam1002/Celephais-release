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
    // ZZ is identically zero and RO depends on r alone: between them they are the
    // J = 0 seed's beta~^theta and beta~^r, which is the shape D0140 sits on.
    Scalar ZZ(space), RO(space);
    for (int d = 0; d < space.get_nbr_domains(); d++) {
        const Domain* dm = space.get_domain(d);
        Val_domain& vf = F.set_domain(d);
        Val_domain& vg = G.set_domain(d);
        Val_domain& va = CA.set_domain(d); Val_domain& vb = CB.set_domain(d);
        Val_domain& vc = CC.set_domain(d); Val_domain& vd = CD.set_domain(d);
        Val_domain& ve = CE.set_domain(d);
        Val_domain& vz = ZZ.set_domain(d);
        Val_domain& vo = RO.set_domain(d);
        vf.allocate_conf();
        vg.allocate_conf();
        for (Val_domain* v : {&va, &vb, &vc, &vd, &ve, &vz, &vo}) v->allocate_conf();
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
            vz.set(idx) = 0.0;
            vo.set(idx) = 1.7 + 0.9 * rr + 0.25 * rr * rr;
        } while (idx.inc());
    }
    for (Scalar* z : {&CA, &CB, &CC, &CD, &CE, &ZZ, &RO}) z->std_base();
    F.std_base();
    G.std_base();

    System_of_eqs syst(space, dom, dom);
    syst.add_cst("F", F);
    syst.add_cst("G", G);
    syst.add_cst("CA", CA); syst.add_cst("CB", CB); syst.add_cst("CC", CC);
    syst.add_cst("CD", CD); syst.add_cst("CE", CE);
    syst.add_cst("ZZ", ZZ); syst.add_cst("RO", RO);

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

    // --- DEFECT 3: dt() of a NAMED def whose own values read CORRECT --------
    //
    // Round 108 localised D0140 = dt(D0055), with D0055 = dt(BT) + divr(BR).
    // On the J = 0 seed BT is identically zero and BR depends on r alone, so
    // D0055 is theta-independent and its dt must vanish -- and D0055 reads back
    // correct while dt(D0055) does not.  Forcing a def into configuration space
    // fixes its VALUES and evidently not what a derivative of it gives.
    //
    // ⚠ THE EVALUATOR STATE IS DECLARED, NOT ASSUMED.  Round 106's reproducer
    // missed a defect because every named def in it was read the instant it was
    // registered, so the whole table sat in one state.  The emitter runs under
    // the round-109 read contract: each def is registered and then read on every
    // domain before the next is registered -- the "read first" column.  The
    // "unread" column is the state where nothing is read until the end.  Both
    // are measured, because if the fault survives the read it is not round
    // 108's defect.
    //
    // ⚠ Every truth below is analytic, from the closed forms the add_cst fields
    // were filled with, and the first draft of this table had two of them wrong
    // (d_th(F/r) written as d_th F).  They are computed in one place now and the
    // point's r and theta are printed so the arithmetic can be checked.
    {
        auto reg = [&](const std::string& d) { syst.add_def(d.c_str()); };
        auto rd = [&](const std::string& nm) {
            return syst.give_val_def_scalar_domain(nm.c_str(), dom)(ix);
        };
        const double rr = space.get_domain(dom)->get_radius()(ix);
        const double tt = space.get_domain(dom)->get_coloc(2)(ix(1));
        const double c2 = std::cos(2.0 * tt), s2 = std::sin(2.0 * tt);
        const double vF = 2.0 + 0.5 * rr * rr + 0.3 * rr * c2;
        const double vRO = 1.7 + 0.9 * rr + 0.25 * rr * rr;
        const double dtF = -0.6 * rr * s2, dtG = -0.8 * rr * rr * s2;
        const double ddtF = -1.2 * rr * c2, ddtG = -1.6 * rr * rr * c2;
        const double dtdrG = -1.6 * rr * s2;          // dr(G) = 0.7 + 0.8 r c2

        std::cout << "\n  DEFECT 3 -- dt() of a named def.  r = " << rr
                  << ", theta = " << tt << "\n";
        std::cout << "  " << std::left << std::setw(42) << "expression"
                  << std::setw(22) << "truth (analytic)" << std::setw(22)
                  << "unread" << std::setw(22) << "read first" << "\n";
        struct Row { const char* what; const char* inner; bool dt; double truth; };
        const std::vector<Row> rows = {
            {"dt( dt(ZZ) + divr(RO) )   D0140's shape", "dt(ZZ) + divr(RO)", true, 0.0},
            {"dt( divr(RO) + dt(ZZ) )   swapped", "divr(RO) + dt(ZZ)", true, 0.0},
            {"  control dt( divr(RO) )", "divr(RO)", true, 0.0},
            {"  control dt( dt(ZZ) )", "dt(ZZ)", true, 0.0},
            {"  the inner def itself, no dt", "dt(ZZ) + divr(RO)", false, vRO / rr},
            {"dt( dt(G) + divr(F) )", "dt(G) + divr(F)", true, ddtG + dtF / rr},
            {"dt( divr(F) + dt(G) )     swapped", "divr(F) + dt(G)", true, ddtG + dtF / rr},
            {"  that inner def itself, no dt", "dt(G) + divr(F)", false, dtG + vF / rr},
            {"dt( dt(G) + F )", "dt(G) + F", true, ddtG + dtF},
            {"dt( F + dt(G) )           swapped", "F + dt(G)", true, ddtG + dtF},
            {"dt( dt(G) + dt(G) )       both dt", "dt(G) + dt(G)", true, 2 * ddtG},
            {"dt( divr(F) + divr(F) )   neither dt", "divr(F) + divr(F)", true,
             2 * dtF / rr},
            {"dt( dr(G) + divr(F) )     dr first", "dr(G) + divr(F)", true,
             dtdrG + dtF / rr},
            {"  control dt( dt(G) )", "dt(G)", true, ddtG},
            {"  control dt( divr(F) )", "divr(F)", true, dtF / rr},
            {"  control dt( F )", "F", true, dtF},
        };
        int t = 0;
        for (const auto& row : rows) {
            const std::string n = "W" + std::to_string(t++);
            const std::string outer = row.dt ? "dt(%s)" : "%s";
            // (i) outer def registered while the inner has never been read
            reg(n + "ai = " + row.inner);
            reg(n + "au = " + (row.dt ? "dt(" + n + "ai)" : n + "ai"));
            const double unread = rd(n + "au");
            // (ii) the same, with the inner read first: the contract's state
            reg(n + "bi = " + row.inner);
            rd(n + "bi");
            reg(n + "br = " + (row.dt ? "dt(" + n + "bi)" : n + "bi"));
            const double readfirst = rd(n + "br");
            const double sc = std::max(std::fabs(row.truth), 1.0);
            const bool bad = std::fabs(unread - row.truth) / sc > 1e-10
                             || std::fabs(readfirst - row.truth) / sc > 1e-10;
            if (bad)
                g_fail++;
            std::cout << "  " << std::left << std::setw(42) << row.what
                      << std::setprecision(12) << std::setw(22) << row.truth
                      << std::setw(22) << unread << std::setw(22) << readfirst
                      << (bad ? "  <-- DIFFER" : "") << "\n";
            (void)outer;
        }
        // ⚠ THE HYPOTHESIS, MEASURED.  The values are right in every row and
        // only the derivative is wrong, which points at the recorded angular
        // BASIS rather than at the arithmetic.  Read it off instead of
        // inferring it: dt(G) of a COS_EVEN field is SIN_EVEN, divr(F) and F
        // stay COS_EVEN, and the rows that agree are exactly the like-with-like
        // sums.  basis_name() prints the theta basis of each def's Val_domain
        // on the shell.
        {
            auto bname = [](int b) {
                switch (b) {
                    case COS_EVEN: return "COS_EVEN";
                    case COS_ODD: return "COS_ODD";
                    case SIN_EVEN: return "SIN_EVEN";
                    case SIN_ODD: return "SIN_ODD";
                    case COSSIN_EVEN: return "COSSIN_EVEN";
                    case COSSIN_ODD: return "COSSIN_ODD";
                    case COS: return "COS";
                    case COSSIN: return "COSSIN";
                    default: return "other";
                }
            };
            auto tbase = [&](const char* nm) {
                const Val_domain& v = syst.give_val_def_scalar_domain(nm, dom);
                Index iq(space.get_domain(dom)->get_nbr_points());
                (void)v(iq);                       // force it to have one
                const Array<int>* b1 = v.get_base().get_base_1d(1);
                return b1 ? bname((*b1)(0)) : "none";
            };
            std::cout << "\n  theta basis recorded for each piece\n";
            struct BR { const char* nm; const char* txt; };
            const std::vector<BR> br = {
                {"BA1", "dt(G)"}, {"BA2", "divr(F)"}, {"BA3", "F"},
                {"BA4", "dr(G)"}, {"BA5", "dt(ZZ)"}, {"BA6", "divr(RO)"},
                {"BA7", "dt(G) + divr(F)"}, {"BA8", "divr(F) + dt(G)"},
                {"BA9", "dt(ZZ) + divr(RO)"}, {"BB1", "divr(RO) + dt(ZZ)"},
                {"BB2", "dt(G) + dt(G)"}, {"BB3", "divr(F) + divr(F)"},
                {"BB4", "dr(G) + divr(F)"},
                // ⚠ and the operands of ROUND 108's pair, to find out whether
                // that defect is a mixed-basis sum too.  Measured, not assumed:
                // the two were found on different shapes and the claim that
                // they are one mechanism has been withheld twice.
                {"BB5", "-1 * (-1 * (multr(dr(F))))"},
                {"BB6", "-1 * (multr(dt(G) + divr(F)))"},
                {"BB7", "dt(G) + divr(F)"},
            };
            for (const auto& x : br) {
                reg(std::string(x.nm) + " = " + x.txt);
                std::cout << "    " << std::left << std::setw(26) << x.txt
                          << tbase(x.nm) << "\n";
            }
        }

        // ⚠ THE OPERATOR x BASIS TABLE, measured before anything is built on it.
        // The emitter needs to know each node's theta basis in order to order a
        // sum, and a table of what each operator does to a basis is the input to
        // that.  Reading it off get_base() costs one run; assuming it is how the
        // last three rounds each went wrong.  CE (COS_EVEN) and dt(CE)
        // (SIN_EVEN) are the two sources.
        {
            auto bname = [](int b) {
                switch (b) {
                    case COS_EVEN: return "COS_EVEN";
                    case COS_ODD: return "COS_ODD";
                    case SIN_EVEN: return "SIN_EVEN";
                    case SIN_ODD: return "SIN_ODD";
                    case COSSIN_EVEN: return "COSSIN_EVEN";
                    case COSSIN_ODD: return "COSSIN_ODD";
                    case COS: return "COS";
                    case SIN: return "SIN";
                    case COSSIN: return "COSSIN";
                    default: return "other";
                }
            };
            auto tbase = [&](const std::string& nm) {
                const Val_domain& v =
                    syst.give_val_def_scalar_domain(nm.c_str(), dom);
                Index iq(space.get_domain(dom)->get_nbr_points());
                (void)v(iq);
                const Array<int>* b1 = v.get_base().get_base_1d(1);
                return std::string(b1 ? bname((*b1)(0)) : "none");
            };
            reg("SRCE = F");                      // COS_EVEN
            reg("SRCO = dt(G)");                  // SIN_EVEN
            reg("SRCP = multsint(F)");            // SIN_ODD
            reg("SRCQ = dt(multsint(F))");        // COS_ODD, if the table holds
            std::cout << "\n  operator x basis, measured\n";
            std::cout << "    " << std::left << std::setw(14) << "operator"
                      << std::setw(14) << "on " + tbase("SRCE")
                      << std::setw(14) << "on " + tbase("SRCO")
                      << std::setw(14) << "on " + tbase("SRCP")
                      << std::setw(14) << "on " + tbase("SRCQ") << "\n";
            int t = 0;
            for (const char* op : {"dr", "dt", "ddr", "multr", "divr", "multsint",
                                   "divsint", "multrsint", "divrsint", "lap",
                                   "lap2"}) {
                const std::string n = "BT" + std::to_string(t++);
                reg(n + "e = " + op + "(SRCE)");
                reg(n + "o = " + op + "(SRCO)");
                reg(n + "p = " + op + "(SRCP)");
                reg(n + "q = " + op + "(SRCQ)");
                std::cout << "    " << std::left << std::setw(14) << op
                          << std::setw(14) << tbase(n + "e")
                          << std::setw(14) << tbase(n + "o")
                          << std::setw(14) << tbase(n + "p")
                          << std::setw(14) << tbase(n + "q") << "\n";
            }
            // the scalar functions, which the emitter only ever applies to a
            // COS_EVEN operand, and the products
            for (const char* op : {"exp", "log", "sqrt"}) {
                const std::string n = "BU" + std::to_string(t++);
                reg(n + "e = " + op + "(SRCE)");
                std::cout << "    " << std::left << std::setw(14) << op
                          << std::setw(14) << tbase(n + "e") << "\n";
            }
            // ⚠ which std_base_* gives which theta basis, measured.  The
            // lattice says the emission is homogeneous only for particular
            // declarations; this says which call produces each of them.
            {
                Scalar P(space);
                struct Nb { const char* what; void (Scalar::*fn)(); };
                const std::vector<Nb> nb = {
                    {"std_base", &Scalar::std_base},
                    {"std_anti_base", &Scalar::std_anti_base},
                    // std_base_*_spher throw "Cheb base r not implemented"
                    // on the polar nucleus, so COS_EVEN and COS_ODD are the
                    // only theta bases a std_base_* call reaches here.
                };
                std::cout << "\n  std_base_* -> theta basis, measured\n";
                for (const auto& x : nb) {
                    P = F;
                    (P.*(x.fn))();
                    const Array<int>* b1 =
                        P(dom).get_base().get_base_1d(1);
                    std::cout << "    " << std::left << std::setw(20) << x.what
                              << (b1 ? bname((*b1)(0)) : "none") << "\n";
                }
            }
            // ⚠ DOES THE r FACTOR MOVE THE CLASS?  Round 314's reading of the
            // ERRATA conflict was that round 285's (beta^theta, Qbar) label
            // refers to beta^hat^theta = r beta~^theta rather than to
            // beta~^theta.  That only reconciles the two records if multr
            // changes the theta class.  Measured on a field registered SIN_EVEN
            // directly, rather than on one produced by dt().
            {
                Scalar SE(space);
                SE = F;
                SE.std_anti_base(1);                   // SIN_EVEN
                syst.add_cst("SE", SE);
                reg("BR1 = SE");
                reg("BR2 = multr(SE)");
                reg("BR3 = multr(multr(SE))");
                reg("BR4 = divr(SE)");
                std::cout << "\n  does the r factor move the theta class?\n";
                for (const auto& pr :
                     std::vector<std::pair<const char*, const char*>>{
                         {"a field registered SIN_EVEN", "BR1"},
                         {"multr of it", "BR2"},
                         {"multr twice", "BR3"},
                         {"divr of it", "BR4"}})
                    std::cout << "    " << std::left << std::setw(30) << pr.first
                              << tbase(pr.second) << "\n";
            }
            reg("BP1 = SRCE * SRCE");
            reg("BP2 = SRCE * SRCO");
            reg("BP3 = SRCO * SRCE");
            reg("BP4 = SRCO * SRCO");
            reg("BP5 = 1 / (SRCE)");
            reg("BP6 = 3 * SRCO");
            reg("BP7 = SRCE * SRCP");
            reg("BP8 = SRCO * SRCP");
            reg("BP9 = SRCP * SRCP");
            reg("BQ1 = SRCP * SRCQ");
            reg("BQ2 = SRCQ * SRCQ");
            reg("BQ3 = SRCE * SRCQ");
            for (const auto& pr : std::vector<std::pair<const char*, const char*>>{
                     {"E * E", "BP1"}, {"E * O", "BP2"}, {"O * E", "BP3"},
                     {"O * O", "BP4"}, {"1 / E", "BP5"}, {"3 * O", "BP6"},
                     {"E * P", "BP7"}, {"O * P", "BP8"}, {"P * P", "BP9"},
                     {"P * Q", "BQ1"}, {"Q * Q", "BQ2"}, {"E * Q", "BQ3"}})
                std::cout << "    " << std::left << std::setw(14) << pr.first
                          << std::setw(14) << tbase(pr.second) << "\n";
        }

        // and the same expression written INLINE, which has no name to be in a
        // state at all
        reg("WI1 = dt(dt(G) + divr(F))");
        reg("WI2 = dt(dt(ZZ) + divr(RO))");
        std::cout << "  " << std::left << std::setw(42) << "INLINE dt(dt(G) + divr(F))"
                  << std::setprecision(12) << std::setw(22) << (ddtG + dtF / rr)
                  << std::setw(22) << rd("WI1") << "\n";
        std::cout << "  " << std::left << std::setw(42) << "INLINE dt(dt(ZZ) + divr(RO))"
                  << std::setprecision(12) << std::setw(22) << 0.0
                  << std::setw(22) << rd("WI2") << "\n";
    }

    // ---------------------------------------------------------------------
    // ROUND 126: is divsint on a NON-VANISHING operand a library defect, or us?
    //
    // A1's bisection localised the one structural disagreement between the
    // emitted text and its sympy twin to D0105 = (1/4) divsint^4(divr^4(1/psi^8)),
    // the single site the static axis check still flags.  Every other
    // disagreement converges spectrally.  So the question is whether divsint
    // misbehaves, or whether we asked it for something outside its contract.
    {
        std::cout << "\n  divsint against the pointwise quotient X/sin(theta)\n";
        std::cout << "    " << std::left << std::setw(34) << "operand"
                  << std::setw(22) << "X/sin at a point"
                  << std::setw(22) << "divsint(X)" << "rel\n";
        Scalar VAN(space), NOV(space);
        VAN = F; NOV = F;
        {   // VAN vanishes on the axis (SIN_EVEN); NOV does not (COS_EVEN)
            for (int d = 0; d < space.get_nbr_domains(); d++) {
                Val_domain& a = VAN.set_domain(d);
                Val_domain& b = NOV.set_domain(d);
                a.allocate_conf(); b.allocate_conf();
                Index ix(space.get_domain(d)->get_nbr_points());
                do {
                    const double tt = space.get_domain(d)->get_coloc(2)(ix(1));
                    a.set(ix) = std::sin(2.0 * tt);
                    b.set(ix) = 1.0 + std::cos(2.0 * tt) / 3.0;
                } while (ix.inc());
            }
            VAN.std_anti_base(1);
            NOV.std_base();
            syst.add_cst("VAN", VAN);
            syst.add_cst("NOV", NOV);
        }
        auto reg = [&](const std::string& d) { syst.add_def(d.c_str()); };
        auto rd = [&](const std::string& nm) {
            return syst.give_val_def_scalar_domain(nm.c_str(), dom)(ix);
        };
        reg("DV1 = divsint(VAN)");
        reg("DV2 = divsint(NOV)");
        const double tt = space.get_domain(dom)->get_coloc(2)(ix(1));
        const double want_v = std::sin(2.0 * tt) / std::sin(tt);
        const double want_n = (1.0 + std::cos(2.0 * tt) / 3.0) / std::sin(tt);
        for (const auto& row : std::vector<std::tuple<const char*, double, const char*>>{
                 {"VANISHES on the axis  sin2t", want_v, "DV1"},
                 {"does NOT vanish  1 + cos2t/3", want_n, "DV2"}}) {
            const double got = rd(std::get<2>(row));
            const double w = std::get<1>(row);
            const double rel = std::fabs(w) > 0 ? std::fabs(got - w) / std::fabs(w) : 0.0;
            std::cout << "    " << std::left << std::setw(34) << std::get<0>(row)
                      << std::setprecision(12) << std::setw(22) << w
                      << std::setw(22) << got << rel << "\n";
            std::cout << "RESULT REPRO_divsint_" << std::get<2>(row) << "_rel "
                      << rel << "\n";
        }
    }

    std::cout << "\n  pairs that DIFFER: " << g_fail << "\n";
    std::cout << "RESULT REPRO_pairs_differing " << g_fail << "\n";
    MPI_Finalize();
    return 0;
}

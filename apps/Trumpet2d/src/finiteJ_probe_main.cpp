/*
 * finiteJ_probe_main.cpp -- A1 rung 0: do the SECTION-4 equations mean, inside
 * Kadath, what they mean in sympy?
 *
 * Before any manufactured solution, the chain that carries it has to be checked
 * against an answer already known: the parser accepting 1.2k-3.6k character
 * strings, dr and dt acting on a Domain_polar_shell, add_cst fields behaving as
 * fields under differentiation, and the emitted text meaning what the sympy
 * expression meant.  Four inherited-and-unexercised pieces, and this project's
 * base rate on those is four wrong in six.
 *
 * THE KNOWN ANSWER is the J = 0 Schwarzschild maximal trumpet, which round 102
 * showed annihilates all six equations symbolically:
 *
 *     PS = R/r,  PH = W R/r,  QF = 0,  BR = C r/R^3,  BT = 0,  QB = 0,  JJ = 0
 *
 * with C = 3 sqrt3 M^2 / 4.  So every def must evaluate to zero here, down to
 * the discretisation floor -- and the same defs on a PERTURBED seed must not,
 * which is the control that stops this from being a test that cannot fail.
 */

#include <algorithm>

#include <mpi.h>

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include "space/space_polar_trumpet.hpp"
#include "src/finiteJ_eqs.hpp"
#include "Trumpet1d/src/table_io.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Kadath::Dim_array;
using Kadath::Index;
using Kadath::Point;
using Kadath::Scalar;
using Kadath::System_of_eqs;
using Kadath::Val_domain;

static void emit(const std::string& k, double v)
{
    std::cout << "RESULT " << k << " " << std::setprecision(17) << v << "\n";
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (argc < 2) {
        if (rank == 0)
            std::cerr << "usage: finiteJ_probe <backbone.dat> [--ntheta N] "
                         "[--perturb X]\n";
        MPI_Finalize();
        return 2;
    }
    int ntheta = 5;
    double perturb = 0.0;
    bool monolithic = false;
    // --clean: register ONLY the emitted sub-defs and equations.  The smoke,
    // vocabulary and parser-length batteries above them are diagnostics, not
    // part of the object under test, and they put ~45 extra defs -- five of
    // them 100 to 120 operations deep, at a stack limit known not to be
    // reproducible -- into the same System_of_eqs before the acceptance test
    // reads it.  Holding them out is the first thing a bisection on
    // registration order has to do, or "order" and "what else is registered"
    // are varied together.
    bool clean = false;
    // --extra-def "NAME = TEXT", repeatable: registered AFTER everything else
    // and reported at the probe point.  A re-spelling of a sub-def that is
    // already wrong, registered at a different position in the same system,
    // separates "this text evaluates wrong" from "this text evaluates wrong
    // HERE" -- which is the question a bisection on registration order asks.
    std::vector<std::string> extra;
    // --maxdefs K: register only the first K sub-defs (and no equations).  If a
    // sub-def is right at K = its own index and wrong at K = 309, then a LATER
    // registration changed it, and bisecting K names the one that did.
    int maxdefs = -1;
    // --extra-early: register the extras BEFORE anything is read back, instead
    // of after.  The difference between the two isolates whether reading a
    // def's operands is what makes a later sum of them come out right.
    bool extra_early = false;
    // --touch: read every sub-def back on every domain immediately after
    // registering it.  Reading a def is supposed to be an observation; if it
    // changes what a LATER def computes from it, it is not.
    bool touch = false;
    // --read-before NAME, repeatable: read exactly these defs, on the probe
    // domain only, immediately before the --extra-early registrations.  Naming
    // the reads one at a time is how the trigger gets isolated.
    std::vector<std::string> readbefore;
    bool readall = false;
    // --break-contract: register the emission WITHOUT the reads the header
    // requires.  It exists so finiteJ_check() can be shown to catch the breach;
    // a control that has never been seen to fail is not a control.
    bool breakcontract = false;
    for (int i = 2; i < argc; i++) {
        const std::string k = argv[i];
        if (k == "--ntheta") ntheta = std::stoi(argv[++i]);
        else if (k == "--perturb") perturb = std::stod(argv[++i]);
        else if (k == "--try-monolithic") monolithic = true;
        else if (k == "--clean") clean = true;
        else if (k == "--extra-def") extra.push_back(argv[++i]);
        else if (k == "--maxdefs") maxdefs = std::stoi(argv[++i]);
        else if (k == "--extra-early") extra_early = true;
        else if (k == "--touch") touch = true;
        else if (k == "--read-before") readbefore.push_back(argv[++i]);
        else if (k == "--read-all") readall = true;
        else if (k == "--break-contract") breakcontract = true;
    }

    TrumpetIO::Table t;
    try {
        t = TrumpetIO::read_table(argv[1]);
    } catch (const std::exception& e) {
        if (rank == 0) std::cerr << "FATAL: " << e.what() << "\n";
        MPI_Finalize();
        return 1;
    }
    const int ndom = static_cast<int>(t.doms.size());
    const int dlast = ndom - 1;

    std::vector<double> bounds;
    std::vector<Dim_array> res;
    std::vector<bool> logshell;
    for (int d = 0; d < ndom; d++) {
        logshell.push_back(t.doms[d].kind == "log");
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

    // The compact domain carries r = infinity, where the seed's R/r and C r/R^3
    // are 0/0 in floating point.  The probe therefore runs on the SHELLS only
    // and says so, rather than reporting a NaN as a small number.
    const int dtop = dlast - 1;
    std::cout << "# finiteJ_probe  ntheta=" << ntheta << "  shells 0.." << dtop
              << " of " << ndom << " domains  perturb=" << perturb << "\n";
    emit("FJP_ntheta", ntheta);
    emit("FJP_domains_probed", dtop + 1);
    emit("FJP_perturb", perturb);

    const double M = t.M;
    const double C = 3.0 * std::sqrt(3.0) * M * M / 4.0;

    Scalar PS(space), PH(space), QF(space), BR(space), BT(space), QB(space);
    Scalar RR(space), ST(space), CT(space), C2(space), H2(space), L2(space),
           CX(space), SQ(space), T7(space), ONE(space);
    // Every domain must be allocated even though the system covers only the
    // shells: a Scalar with an unallocated domain segfaults inside add_cst.
    for (int d = 0; d < ndom; d++) {
        const Kadath::Domain* dm = space.get_domain(d);
        Val_domain& vps = PS.set_domain(d);
        Val_domain& vph = PH.set_domain(d);
        Val_domain& vqf = QF.set_domain(d);
        Val_domain& vbr = BR.set_domain(d);
        Val_domain& vbt = BT.set_domain(d);
        Val_domain& vqb = QB.set_domain(d);
        Val_domain& vrr = RR.set_domain(d);
        Val_domain& vst = ST.set_domain(d);
        Val_domain& vct = CT.set_domain(d);
        Val_domain& vc2 = C2.set_domain(d);
        Val_domain& vh2 = H2.set_domain(d);
        Val_domain& vl2 = L2.set_domain(d);
        Val_domain& vcx = CX.set_domain(d);
        Val_domain& vsq = SQ.set_domain(d);
        Val_domain& vt7 = T7.set_domain(d);
        Val_domain& von = ONE.set_domain(d);
        for (Val_domain* v : {&vps, &vph, &vqf, &vbr, &vbt, &vqb, &vrr, &vst, &vct, &vc2, &vh2, &vl2, &vcx, &vsq, &vt7, &von})
            v->allocate_conf();
        Index idx(dm->get_nbr_points());
        do {
            const int i = idx(0);
            if (d > dtop) {                      // the compact domain: r = infinity
                for (Val_domain* v : {&vps, &vph, &vqf, &vbr, &vbt, &vqb,
                                      &vrr, &vst, &vct, &vc2, &vh2, &vl2, &vcx, &vsq, &vt7, &von})
                    v->set(idx) = 0.0;
                continue;
            }
            const double rr = t.pts[d][i].r;
            const double W = t.pts[d][i].W;
            const double R = t.pts[d][i].Rr * rr;        // Rr is R/r
            const double th = dm->get_coloc(2)(idx(1));
            // the J = 0 seed, exactly as round 102 verified it symbolically
            vps.set(idx) = R / rr;
            vph.set(idx) = W * R / rr;
            vqf.set(idx) = 0.0;
            vbr.set(idx) = C * rr / (R * R * R);
            vbt.set(idx) = 0.0;
            vqb.set(idx) = 0.0;
            vrr.set(idx) = rr;
            vst.set(idx) = std::sin(th);
            vct.set(idx) = std::cos(th);
            vc2.set(idx) = std::cos(2.0 * th);
            // lap2(r^2 cos2th) = (n^2 - m^2) r^{n-2} cos = 0 exactly
            vh2.set(idx) = rr * rr * std::cos(2.0 * th);
            // lap(r^2 P_2(cos th)) = 0 exactly;  P_2 = (1 + 3cos2th)/4
            vl2.set(idx) = rr * rr * (1.0 + 3.0 * std::cos(2.0 * th)) / 4.0;
            // the TARGET of the cot-theta construction, built pointwise purely
            // as a comparison value -- it is never differentiated
            vcx.set(idx) = (std::sin(th) == 0.0) ? 0.0
                           : (std::cos(th) / std::sin(th)) * std::cos(2.0 * th);
            // A field that VANISHES on the axis, so cot(theta)*X is finite
            // there: X = sin^2 = (1 - cos2th)/2, and cot*X = sin(2th)/2.
            vsq.set(idx) = (1.0 - std::cos(2.0 * th)) / 2.0;
            vt7.set(idx) = std::sin(2.0 * th) / 2.0;
            von.set(idx) = 1.0;
        } while (idx.inc());
    }
    // the control: a perturbation of the seed must NOT annihilate the equations
    if (perturb != 0.0) {
        for (int d = 0; d <= dtop; d++)
            PH.set_domain(d) = PH(d) * (1.0 + perturb);
    }
    for (Scalar* s : {&PS, &PH, &QF, &BR, &BT, &QB, &RR, &ST, &CT, &C2, &H2, &L2, &CX, &SQ, &T7, &ONE})
        s->std_base();

    // is the seed the one that was verified?  R(throat) must be 3M/2.
    {
        double rmin = 1e300, Rmin = 0.0;
        for (std::size_t i = 0; i < t.pts[0].size(); i++)
            if (t.pts[0][i].r < rmin) { rmin = t.pts[0][i].r; Rmin = t.pts[0][i].Rr * rmin; }
        emit("FJP_R_at_inner", Rmin);
        emit("FJP_R_over_1p5M", Rmin / (1.5 * M));
    }

    std::cout << "# building system" << std::endl;
    System_of_eqs syst(space, 0, dtop);
    std::cout << "# system built" << std::endl;
    syst.add_cst("PS", PS);
    syst.add_cst("PH", PH);
    syst.add_cst("QF", QF);
    syst.add_cst("BR", BR);
    syst.add_cst("BT", BT);
    syst.add_cst("QB", QB);
    syst.add_cst("RR", RR);
    syst.add_cst("ST", ST);
    syst.add_cst("CT", CT);
    syst.add_cst("C2", C2);
    syst.add_cst("H2", H2);
    syst.add_cst("L2", L2);
    syst.add_cst("CX", CX);
    syst.add_cst("SQ", SQ);
    syst.add_cst("T7", T7);
    // `ones` is a field the apps register themselves (BH2d/NS2d do the same);
    // the emitted monomials materialise against it.
    syst.add_cst("ones", ONE);
    syst.add_cst("JJ", 0.0);
    std::cout << "# csts registered" << std::endl;

    // ---- SMOKE: do the operators work at all, and where is the size limit? --
    // Two separate questions, and a 2254-character def failing answers neither
    // on its own.  Small defs with EXACT known answers first.
    if (!clean) {
        struct Sm { const char* def; const char* what; };
        const std::vector<Sm> sm = {
            {"S1 = dr(RR)", "dr(r) = 1"},
            {"S2 = (ST)^2 + (CT)^2", "sin^2 + cos^2 = 1"},
            {"S3 = dt(CT) + ST", "dt(cos) + sin = 0"},
            {"S4 = dt(ST) - CT", "dt(sin) - cos = 0"},
            {"S5 = dt(dt(CT)) + CT", "ddt(cos) + cos = 0"},
            {"S6 = dt(dt(C2)) + 4 * C2", "ddt(cos2th) + 4cos2th = 0  <- IS representable"},
            {"S7 = dt(C2)", "dt(cos2th) = -2 sin2th, max |.| = 2"},
            // ---- THE VOCABULARY BATTERY: which constructs are usable? ----
            {"V1 = divsint(multsint(C2)) - C2", "divsint . multsint = id"},
            {"V2 = divr(multr(C2)) - C2",       "divr . multr = id"},
            {"V3 = lap2(H2)",                   "lap2(r^2 cos2th) = 0 EXACT"},
            {"V4 = lap(L2)",                    "lap(r^2 P_2) = 0 EXACT"},
            {"V5 = multr(divr(C2)) - C2",       "multr . divr = id"},
            {"V6 = lap2(C2) * (RR)^2 + 4 * C2", "lap2(cos2th) = -4cos2th/r^2"},
            // ---- cot(theta) WITHOUT a cos(theta) field.  There is no
            // Ope_mult_cost, but d_th(sin X) = cos X + sin d_th X gives
            //     cot(theta) X = divsint(dt(multsint(X))) - dt(X)
            // in operators that have just been verified individually.
            {"V7 = divsint(dt(multsint(SQ))) - dt(SQ) - T7", "cot(th)*sin^2 = sin2th/2, built"},
            {"V8 = divrsint(multrsint(C2)) - C2", "divrsint . multrsint = id"},
            // ---- research round 297 ruling (3): measure the add_cst exposure
            // rather than argue it.  PS is theta-INDEPENDENT in the seed, the
            // same shape as L0ModelT's 39 coefficient fields.
            {"S8 = dt(PS)", "dt of a theta-independent add_cst field = 0"},
            {"S9 = dt(dt(PS))", "ddt of the same = 0"},
            // ---- T8 bisection: with BT = 0 and BR theta-independent, only
            // these survive.  Each is compared with sympy at the probe point.
            {"P1 = dr(dr(BR))",                   "P1"},
            {"P2 = divr(divr(multr(dr(BR))))",    "P2 = dr(BR)/r"},
            {"P3 = divr(divr(multr(divr(BR))))",  "P3 = BR/r^2"},
            {"P4 = divr(dr(BR))",                 "P4 = dr(BR)/r"},
            {"P5 = divr(divr(BR))",               "P5 = BR/r^2"},
            {"P6 = multr(divr(BR)) - BR",         "P6 = 0"},
            {"P7 = divr(multr(BR)) - BR",         "P7 = 0"},
            // the T8 terms that MUST vanish on this seed (BT = 0, BR theta-indep)
            {"Q1 = dt(dt(BR) + (-1 * (multr(BT))))",   "Q1 = 0"},
            {"Q2 = multr(dt(BT) + divr(BR)) - BR",     "Q2 = 0"},
            {"Q3 = dr(BT) + divr(divr(dt(BR) + (-1 * (multr(BT)))))", "Q3 = Lrt = 0"},
            {"Q4 = divsint(BT)",                       "Q4 = divsint of a zero field"},
            {"Q5 = divsint(dt(multsint(BT))) - dt(BT)", "Q5 = cot(BT) = 0"},
            {"Q6 = dt(BR)",                            "Q6 = 0"},
            {"Q7 = multr(BT)",                         "Q7 = 0"},
            // A1's inner terms, one at a time
            {"Z1 = (-1 * (-1 * (multr(dr(BR))))) - multr(dr(BR))", "Z1 = 0?"},
            {"Z2 = multr(dr(BR))",                     "Z2 = r dr(BR)"},
            {"Z3 = multr(dt(BT) + divr(BR))",          "Z3 = BR"},
            {"Z4 = dt(dt(BR) + (-1 * (multr(BT)))) + ((-1 * (-1 * (multr(dr(BR))))) + (-1 * (multr(dt(BT) + divr(BR)))))", "Z4 = A1 inner"},
            {"Z6 = (-1 * (-1 * (multr(dr(BR))))) + (-1 * (multr(dt(BT) + divr(BR))))", "Z6 = Z2 - Z3"},
            {"Z7 = dt(BR) + BR - BR",   "Z7 = dt(BR) = 0, added to a COS_EVEN field"},
            {"Z8 = dt(BR) + multr(dr(BR)) - multr(dr(BR))", "Z8 = 0, mixing dt into a sum"},
            {"Z9 = dt(PS) + PS - PS",   "Z9 = 0, same with a nonzero smooth field"},
            {"Y1 = (-1 * (multr(dt(BT) + divr(BR)))) + multr(dt(BT) + divr(BR))", "Y1: -1*(op(a+b))"},
            {"Y2 = (0 - (multr(dt(BT) + divr(BR)))) + multr(dt(BT) + divr(BR))", "Y2: 0-(op(a+b))"},
            {"Y3 = ((-1) * (multr(dt(BT) + divr(BR)))) + multr(dt(BT) + divr(BR))", "Y3: (-1)*(op(a+b))"},
            {"Y4 = (-1 * (multr(divr(BR)))) + multr(divr(BR))", "Y4: -1*(op(simple))"},
            {"Y5 = (-1 * (dt(BT) + divr(BR))) + (dt(BT) + divr(BR))", "Y5: -1*(a+b) no op"},
            {"Z10 = Z2 - Z3",  "Z10: the SAME sum via NAMED sub-defs"},
            // ⚠ log inside an expression: each operand right, difference wrong
            {"L1 = dr(log(PS)) - dr(log(PS))",   "L1 = 0?"},
            {"L2 = log(PS) - log(PS)",           "L2 = 0?"},
            {"L3 = dr(log(PH)) - dr(log(PH))",   "L3 = 0?"},
            {"L4 = dr(log(PH)) - 4 * dr(log(PS))", "L4 = the T8 combination inline"},
            {"L5 = log(PS) * PS - log(PS) * PS", "L5 = 0?"},
            {"Z11 = (-1 * (-1 * (multr(dr(BR))))) + (-1 * Z3)", "Z11: one named"},
            {"Z5 = divr(divr(dt(dt(BR) + (-1 * (multr(BT)))) + ((-1 * (-1 * (multr(dr(BR))))) + (-1 * (multr(dt(BT) + divr(BR)))))))", "Z5 = A1 second term"},
            // ---- which EXPRESSION SHAPES does the parser accept? ----
        };
        for (const auto& x : sm) {
            try {
                syst.add_def(x.def);
            } catch (const std::exception& ex) {
                std::cout << "#   " << x.what << " : add_def THREW " << ex.what()
                          << std::endl;
                continue;
            }
            const char* nm = std::string(x.def).substr(0, 2).c_str();
            char id[3] = {x.def[0], x.def[1], 0};
            double mx = 0.0;
            for (int d = 0; d <= dtop; d++) {
                const Val_domain& v = syst.give_val_def_scalar_domain(id, d);
                Index ix(space.get_domain(d)->get_nbr_points());
                do { mx = std::max(mx, std::fabs(v(ix))); } while (ix.inc());
            }
            double want = 0.0;
            if (std::string(id) == "S1" || std::string(id) == "S2") want = 1.0;
            if (std::string(id) == "S7") want = 2.0;
            emit(std::string("FJP_smoke_") + id, std::fabs(mx - want));
            std::cout << "#   " << x.what << "  ->  |value - expected| = "
                      << std::setprecision(3) << std::fabs(mx - want) << std::endl;
            (void)nm;
        }
    }
    // ---- how long a def can the parser take?  Built from ONE repeated term,
    // so only the length varies.
    if (!clean) {
        std::string acc = "PS";
        for (int k = 1; k <= 120; k++) {   // ~145 is where it dies, and NOT reproducibly
            acc = "(" + acc + " + PS)";
            if (k < 100 || k % 5) continue;
            const std::string d = "L" + std::to_string(k) + " = " + acc;
            std::cout << "#   parser length probe: " << acc.size()
                      << " chars, depth " << k << std::endl << std::flush;
            syst.add_def(d.c_str());
        }
    }

    (void)monolithic;
    // ---- THE ACCEPTANCE TEST.  finiteJ_register() owns the read-state
    // contract -- it registers each def and reads it into configuration space,
    // which is what the evaluator needs and what the probe must not be trusted
    // to remember.  The emission is NOT registered by hand here.
    //
    // ⚠ finiteJ_detail::subdefs() is reached only by the bisection flags below,
    // which break the contract on purpose.
    const auto& SUB = Trumpet::finiteJ_detail::subdefs();
    const int nsub = (maxdefs < 0) ? int(SUB.size())
                                   : std::min<int>(maxdefs, int(SUB.size()));
    emit("FJP_axis_violations",
         static_cast<double>(Trumpet::finiteJ_axis_violations().size()));
    // ⚠ the list holds every mixed sum the ordering pass CONSIDERED, and says
    // of each whether it was swapped or left; counting the list as "reordered"
    // would claim more than it shows.
    {
        int nswap = 0;
        for (const char* o : Trumpet::finiteJ_ordered_sums()) {
            if (std::string(o).find("swapped") != std::string::npos)
                nswap++;
            if (rank == 0)
                std::cout << "#   sum ordering (seed-specific): " << o << "\n";
        }
        emit("FJP_ordered_sums_considered",
             static_cast<double>(Trumpet::finiteJ_ordered_sums().size()));
        emit("FJP_ordered_sums_swapped", static_cast<double>(nswap));
    }
    if (rank == 0)
        for (const char* v : Trumpet::finiteJ_axis_violations())
            std::cout << "#   axis violation carried by the emission: " << v
                      << "\n";
    try {
        if (maxdefs < 0) {
            Trumpet::finiteJ_register(syst, space, 0, dtop, !breakcontract);
        } else {
            // --maxdefs truncates the emission, so the header's registration
            // cannot be used; this path is a diagnostic and says so.
            for (int si = 0; si < nsub; si++) {
                syst.add_def((std::string(SUB[si].name) + " = " + SUB[si].def)
                                 .c_str());
                if (touch)
                    Trumpet::finiteJ_detail::read_def(syst, space, SUB[si].name,
                                                      0, dtop);
            }
        }
    } catch (const std::exception& ex) {
        if (rank == 0)
            std::cerr << "FATAL: registration threw: " << ex.what() << "\n";
        MPI_Finalize();
        return 3;
    }
    emit("FJP_subdefs", static_cast<double>(nsub));

    // ---- THE CONTROL.  Every sum re-expressed with its operands read first.
    if (maxdefs < 0) {
        double readchk = 0.0;
        try {
            readchk = Trumpet::finiteJ_check(syst, space, 0, dtop);
            if (rank == 0)
                std::cout << "# read-state contract kept; worst re-expression "
                             "difference " << readchk << "\n";
        } catch (const std::exception& ex) {
            readchk = 1.0;
            if (rank == 0)
                std::cout << "# read-state contract BROKEN: " << ex.what()
                          << "\n";
        }
        emit("FJP_readcheck", readchk);
    }

    if (rank == 0)
        std::cout << "# all six defs registered\n";

    std::vector<std::string> extra_names;
    // ⚠ The read has to INDEX the Val_domain, not just fetch the reference:
    // operator()(Index) is what forces configuration space.  A fetch alone
    // leaves the def in whatever space it was built in, which is why the first
    // version of this probe found no trigger.
    {
        const int dR = (dtop >= 1) ? 1 : 0;
        auto peek = [&](const char* nm) {
            const Val_domain& v = syst.give_val_def_scalar_domain(nm, dR);
            Index iq(space.get_domain(dR)->get_nbr_points());
            (void)v(iq);
        };
        if (readall)
            for (int si = 0; si < nsub; si++) peek(SUB[si].name);
        for (const auto& nm : readbefore) peek(nm.c_str());
    }
    if (extra_early)
        for (const auto& x : extra) {
            const std::string nm = x.substr(0, x.find(' '));
            try {
                syst.add_def(x.c_str());
                extra_names.push_back(nm);
            } catch (const std::exception& ex) {
                std::cout << "#   extra " << nm << " THREW " << ex.what() << "\n";
            }
        }

    // ---- SUB-DEF LOCALISATION (research round 301 item 1) -----------------
    // Equation-level residuals are the wrong granularity.  Every sub-def is
    // reported AT ONE POINT so the same quantity can be evaluated through sympy
    // on the same seed and the FIRST disagreement found.
    {
        const int dP = (dtop >= 1) ? 1 : 0, iP = 3, jP = 2;
        const Kadath::Domain* dm = space.get_domain(dP);
        Index ix(dm->get_nbr_points());
        for (int k = 0; k < iP + jP * dm->get_nbr_points()(0); k++) ix.inc();
        emit("FJP_pt_dom", dP);
        emit("FJP_pt_r", t.pts[dP][iP].r);
        emit("FJP_pt_R", t.pts[dP][iP].Rr * t.pts[dP][iP].r);
        emit("FJP_pt_W", t.pts[dP][iP].W);
        emit("FJP_pt_th", dm->get_coloc(2)(jP));
        if (!clean)
        for (const char* nm : {"P1", "P2", "P3", "P4", "P5", "P6", "P7",
                               "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7",
                               "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7",
                               "Z8", "Z9", "Y1", "Y2", "Y3", "Y4", "Y5",
                               "Z10", "Z11", "L1", "L2", "L3", "L4", "L5"})
            emit(std::string("FJP_val_") + nm,
                 syst.give_val_def_scalar_domain(nm, dP)(ix));
        for (int si = 0; si < nsub; si++)
            emit(std::string("FJP_val_") + SUB[si].name,
                 syst.give_val_def_scalar_domain(SUB[si].name, dP)(ix));
        if (maxdefs < 0)
            for (const char* nm : Trumpet::finiteJ_eq_names())
                emit(std::string("FJP_val_") + nm,
                     syst.give_val_def_scalar_domain(nm, dP)(ix));
        for (const auto& nm : extra_names)
            emit(std::string("FJP_extra_") + nm,
                 syst.give_val_def_scalar_domain(nm.c_str(), dP)(ix));
        if (!extra_early) {
            for (const auto& x : extra) {
                const std::string nm = x.substr(0, x.find(' '));
                try {
                    syst.add_def(x.c_str());
                } catch (const std::exception& ex) {
                    std::cout << "#   extra " << nm << " THREW " << ex.what()
                              << "\n";
                    continue;
                }
                emit(std::string("FJP_extra_") + nm,
                     syst.give_val_def_scalar_domain(nm.c_str(), dP)(ix));
            }
        }
    }

    // ⚠ Split AXIS from INTERIOR.  The cot(theta) construction is only valid
    // on an operand that vanishes on the axis; if the failures sit at
    // idx(1) == 0 that is the cause, and if they are spread it is not.
    double worst = 0.0, worst_int = 0.0;
    std::cout << "#   equation      max|E| axis     max|E| interior\n";
    for (const char* ename : Trumpet::finiteJ_eq_names()) {
        if (maxdefs >= 0) break;
        double mx = 0.0, mxi = 0.0;
        for (int d = 0; d <= dtop; d++) {
            const Val_domain& v = syst.give_val_def_scalar_domain(ename, d);
            const int nth = space.get_domain(d)->get_nbr_points()(1);
            Index idx(space.get_domain(d)->get_nbr_points());
            do {
                const double x = v(idx);
                const double a = std::isfinite(x) ? std::fabs(x) : 1e300;
                mx = std::max(mx, a);
                if (idx(1) != 0 && idx(1) != nth - 1)
                    mxi = std::max(mxi, a);
            } while (idx.inc());
        }
        std::cout << "#   " << std::left << std::setw(13) << ename
                  << std::setprecision(4) << std::setw(16) << mx << mxi << "\n";
        emit(std::string("FJP_") + ename, mx);
        emit(std::string("FJP_int_") + ename, mxi);
        worst = std::max(worst, mx);
        worst_int = std::max(worst_int, mxi);
    }
    emit("FJP_worst_interior", worst_int);
    emit("FJP_worst", worst);

    MPI_Finalize();
    return 0;
}

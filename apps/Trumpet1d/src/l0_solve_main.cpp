/*
 * l0_solve_main.cpp -- Phase 2 / T2.2 of the spinning-trumpet program.
 *
 * Closes the l=0 BVP and solves it.  The operator is imported unchanged from
 * src/l0_setup.hpp, i.e. it is exactly the five-row operator T2.1 gated; this
 * file only adds the six conditions that make the system square, and Newton.
 *
 * THE CONDITION COUNT IS NOT A CHOICE.  Kadath counts, per field per domain,
 * nbr_coefs unknowns against nbr_coefs - order conditions from add_eq_inside
 * (src/Domain/Oned/domain_oned_qcq_nbr_{unknowns,conditions}.cpp), and one
 * condition per boundary or matching equation.  With three fields, three
 * second-order rows in every domain and C0+C1 matching at each interface the
 * deficit is 6*ndom - 6*(ndom-1) = 6 exactly, for any ndom.  SPEC_T22_innerBC's
 * "2 pins + 2 outer + 2 inner = 6" therefore fits with no slack and no spare
 * row -- and if it did not, do_newton would report [DOFDUMP] and throw rather
 * than silently solve something else.
 *
 * THE SIX CONDITIONS (SPEC_T22_innerBC.md, status READY):
 *   inner, at r_in = r(W0), on domain 0's INNER_BC
 *     dU/dr = j2 [dU/dr]_P ,  dQ/dr = j2 [dQ/dr]_P              (--inner dU,dQ)
 *     NEUMANN-WITH-SOURCE because the throat-regular homogeneous space is
 *     2-dimensional and consists of CONSTANTS only -- V1 = (1,-4,0),
 *     V2 = (0,-1,1) in (U,Q,G) -- which every pure-derivative row annihilates.
 *   constraint pins, same boundary
 *     EH = 0 ,  EM = 0
 *     T4.1's fix: the constraints obey their own first-order system with
 *     admissible decaying-nonzero solutions, so decay does NOT propagate them.
 *     Pinning both at one point does (identity propagation).
 *   outer, at r = infinity, on the compactified domain's OUTER_BC
 *     two of  U = 0 , Q = 0 , G = 0                             (--outer U,G)
 *     Two, not three: they only have to kill span{V1,V2}, which is 2-dim.  Any
 *     pair does it; {U,G} has the identity coefficient matrix.  The unimposed
 *     third value is then a free null test.
 *
 * SCALING.  The inner rows are UNSCALED field equations, as the spec gives
 * them.  The pins reference the scaled rows EH/EM, which is harmless: scaling
 * by a nonvanishing function does not move a zero set.
 *
 * Usage:
 *   l0_solve <backbone.dat> <coefs.dat> <bc.dat> [options]
 *     --j2 <v>        source amplitude (default 1)
 *     --inner a,b     inner pair from {dU,dQ,dG,vU,vQ,vG,combo} (default dU,dQ)
 *     --outer a,b     outer pair from {U,Q,G}                   (default U,G)
 *     --seed <amp>    nonzero initial guess; for the uniqueness test, where a
 *                     zero start would make the test vacuous (do_newton
 *                     returns at iteration 0 before assembling anything)
 *     --prec <p>      Newton precision (default 1e-11)
 *     --profile <f>   write the solved profile + spectral coefficients
 */

#include "src/l0_setup.hpp"

#include "Apps/Startup/solver_startup.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using Kadath::Index;
using Kadath::Scalar;
using Kadath::System_of_eqs;
using Kadath::Val_domain;
using TrumpetIO::BcTable;

namespace
{

std::vector<std::string> split2(const std::string& s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty())
            out.push_back(tok);
    return out;
}

/**
 * The left-hand side of an inner BC row as a Kadath expression string.
 *
 * SPEC v2 rows are general 6-vectors on the r-jet (U, dr(U), Q, dr(Q), G,
 * dr(G)); scripts/l0_bc_rows.py has already folded 1/(dW/dr) into the
 * derivative slots (the rows are quoted in W-derivatives) and normalised the
 * row by its own max|entry|.  Coefficients are emitted into the string with
 * full precision -- these are boundary conditions, not diagnostics.
 */
std::string bc_lhs(const TrumpetIO::BcRow& r, const std::string& prefix,
                   Kadath::System_of_eqs& syst)
{
    if (!r.general)
        return (r.kind == "der1") ? "dr(" + r.field + ")"
             : (r.field == "4U+Q+G" ? "4 * U + Q + G" : r.field);
    static const char* JETSTR[6] = {"U", "dr(U)", "Q", "dr(Q)", "G", "dr(G)"};
    static const char* SUF[6] = {"cU", "cUp", "cQ", "cQp", "cG", "cGp"};
    std::string out;
    for (int k = 0; k < 6; k++) {
        if (r.coef[k] == 0.0)
            continue;                       // a genuinely absent slot (Q_W is 0)
        const std::string cn = prefix + SUF[k];
        syst.add_cst(cn.c_str(), r.coef[k]);
        if (!out.empty())
            out += " + ";
        out += cn + " * " + JETSTR[k];
    }
    if (out.empty())
        throw std::runtime_error("bc row " + prefix + " is identically zero");
    return out;
}

/** Boundary value of a named add_def, read through the Chebyshev coefficients. */
double def_at_boundary(System_of_eqs& syst, const Kadath::Space& space,
                       const char* name, int dom, int bound)
{
    Index pcf(space.get_domain(dom)->get_nbr_coefs());
    return space.get_domain(dom)->val_boundary(
        bound, syst.give_val_def_scalar_domain(name, dom), pcf);
}

} // namespace

/** true when --horizon-order asks for the first-order reading. */
static bool horizonorder_is_o1(const std::string& h) { return h.rfind("o1@", 0) == 0; }

extern "C" {
// Least squares by SVD.  MKL LAPACK is already linked and Kadath declares
// dgesdd_ this way itself (src/Newton/do_newton_jfnk_schur.cpp).  dgelsd
// returns the singular values of the stack as a by-product, which is exactly
// the P3 diagnostic.
void dgelsd_(int* m, int* n, int* nrhs, double* a, int* lda, double* b, int* ldb,
             double* s, double* rcond, int* rank, double* work, int* lwork,
             int* iwork, int* info);
}

int main(int argc, char** argv)
{
    const int rank = KadathApps::init_mpi(argc, argv);

    if (argc < 4) {
        if (rank == 0)
            std::cerr << "usage: l0_solve <backbone.dat> <coefs.dat> <bc.dat> "
                         "[--j2 v] [--inner a,b] [--outer a,b] [--seed a] "
                         "[--prec p] [--profile f]\n";
        MPI_Finalize();
        return 2;
    }

    double j2 = 1.0, seed = 0.0, prec = 1e-11;
    std::string inner = "dU,dQ", outer = "U,G", pins = "EH,EM", profile;
    bool rownorm = false, manufactured = false, dumpresid = false;
    std::string jacdump, pinat = "inner", horizonfix, hside = "left", horder;
    double residfloor = 1e-12;
    bool additive = false;
    double addweight = 1.0;
    std::string addequil = "appended";   // research's prescription
    std::string addrows = "all";
    std::string logenrich;          // e.g. "1:UQG" or "1:U,2:U"
    // ROUND-9 DROP-ONE-PIN.  The constraint-proportionality theorem
    // (E_Mr = lambda E_H + gK E_K + gTT E_chi_tt, exact on every slot -- checked
    // independently in scripts/bianchi_closure.py --proportionality) says there
    // is ONE independent constraint, so ONE pin suffices in the continuum and
    // the sixth budget slot is freed.  Whose it becomes is research's ruling, so
    // this flag does NOT reassign it: it permits the physics total to fall short
    // by `conddef` and reports the resulting counts.
    int conddef = 0;
    for (int i = 4; i < argc; i++) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("missing value after " + k);
            return argv[++i];
        };
        if (k == "--j2") j2 = std::stod(next());
        else if (k == "--seed") seed = std::stod(next());
        else if (k == "--prec") prec = std::stod(next());
        else if (k == "--inner") inner = next();
        else if (k == "--outer") outer = next();
        else if (k == "--pins") pins = next();
        // T4.1 argues a pin at ONE point zeroes the constraints globally by
        // identity propagation.  Measurement says otherwise on this grid, so
        // where the pin sits is a testable choice, not a detail.
        else if (k == "--pin-at") pinat = next();
        else if (k == "--cond-deficit") conddef = std::stoi(next());
        else if (k == "--profile") profile = next();
        else if (k == "--rownorm") rownorm = true;
        // Per-POINT row residuals.  The gate needs the TAU-projected residual
        // (add_eq_inside drops the top two Chebyshev coefficients of each
        // equation), and the pointwise maximum below is a different number
        // entirely: it can sit at O(1) while the tau residual is at 1e-10 if
        // the solution is under-resolved on that domain.  Emitting the
        // samples lets scripts/l0_diag.py do the projection and report both.
        else if (k == "--dump-resid") dumpresid = true;
        // --horizon-fix f1,f2  BUDGET-NEUTRAL form of the horizon trial:
        // at the interface where the rows degenerate, DROP the C1 matching
        // of the two named fields and impose the two lost pointwise
        // conditions (EK = 0, EXT = 0) instead.  Two out, two in, so all six
        // physics conditions -- both inner rows, both constraint pins and
        // both outer decay rows -- survive untouched.  Rationale: at a
        // degenerate endpoint the operator itself relates the one-sided
        // derivatives, so full C1 matching there is not independent data.
        else if (k == "--horizon-fix") horizonfix = next();
        // where the two recovered pointwise rows are imposed.  "left" puts
        // both on the inner domain's outer face; "split" gives each adjacent
        // domain one, which is what a lost C1 link on BOTH sides needs.
        else if (k == "--horizon-side") hside = next();
        // --horizon-order d1|d2|split   THE FIX (research round-2 Q1).
        // add_eq_order lowers E_K and E_chi_tt by one tau order on the
        // horizon-adjacent domain(s), freeing exactly two conditions, and they
        // are spent on the two pointwise rows the degeneracy leaves unimposed
        // at r(2M):  E_K = 0  and  (E_chi_tt - kappa*E_K) = 0.  Matching and
        // all six physics conditions are untouched.  Which adjacent domain
        // gives up the conditions is a measured choice, not a derived one.
        else if (k == "--horizon-order") horder = next();
        // Relative floor below which a collocation point is too small to carry
        // a meaningful RELATIVE residual -- see the residual block below.
        else if (k == "--resid-floor") residfloor = std::stod(next());
        // --additive  ROUND-4 FORMULATION.  Leave the square 339x339 baseline
        // untouched -- no add_eq_order, all six physics conditions, matching
        // intact -- and APPEND three rows that are exact properties of the
        // continuum solution, then solve the consistent 342x339 least-squares
        // problem.  Round 4's theorem: the horizon costs the continuum budget
        // nothing, so the two bits the tau projection loses must be handed back
        // WITHOUT taking anything away.
        else if (k == "--additive") additive = true;
        // relative weight on the appended rows, 1 = equilibrated to the median
        // core-row norm.  A sensitivity knob, reported, never a tuning knob.
        else if (k == "--add-weight") addweight = std::stod(next());
        // rows = every row to unit norm (default); appended = research's
        // literal prescription, only the appended rows scaled, core untouched;
        // none = no scaling at all.  Row scaling is not neutral for a least
        // squares problem, so which one is used is measured, not assumed.
        else if (k == "--add-equil") addequil = next();
        // which appended rows to use: all | compat (rows 1-2 only) | g (row 3
        // only).  Lets the core-row inconsistency be attributed to a row
        // rather than to "the formulation".
        else if (k == "--add-rows") addrows = next();
        // --log-enrich <k>:<fields>[,<k>:<fields>]  far-field enrichment.
        // "1:UQG" adds ln(r)/r in all three field directions, three
        // amplitudes, so the DIRECTION is recovered rather than assumed;
        // "1:U" adds one, which keeps the appended rows overdetermining the
        // system so the P2 residual split stays a real test.
        else if (k == "--log-enrich") logenrich = next();
        else if (k == "--dump-jacobian") jacdump = next();
        else if (k == "--manufactured") {
            // MANUFACTURED-SOLUTION BVP (playbook rule 2).  The exact mass mode
            // U = (1-W)/2, Q = 0, G = F_M - 2 annihilates all five rows at
            // j2 = 0, so it is a solution of THIS BVP provided the six
            // conditions are the ones it satisfies:
            //   inner  dU/dr = -(dW/dr)/2   and   dQ/dr = 0
            //   outer  U(inf) = 0  and  Q(inf) = 0     (G_M(inf) = -2, so G
            //          cannot be an outer condition here)
            // Recovering it is a full-BVP oracle with a known answer -- the
            // T2.2 analogue of T2.1's operator oracle.
            manufactured = true;
            j2 = 0.0;
            inner = "mU,mQ";
            outer = "U=0,Q=0,G=-2";
            pins = "EH";
        }
        else {
            if (rank == 0)
                std::cerr << "FATAL: unknown option " << k << "\n";
            MPI_Finalize();
            return 2;
        }
    }

    std::unique_ptr<Trumpet::L0Model> mp;
    BcTable bc;
    try {
        mp = std::make_unique<Trumpet::L0Model>(argv[1], argv[2], rownorm);
        bc = TrumpetIO::read_bc(argv[3]);
        // The manufactured BVP's inner data comes from the backbone itself:
        // U_M = (1-W)/2  =>  dU_M/dr = -(dW/dr)/2, and Q_M = 0 identically.
        if (manufactured) {
            TrumpetIO::BcRow mU, mQ;
            mU.kind = "der1"; mU.field = "U"; mU.rhs = -0.5 * bc.dWdr;
            mQ.kind = "der1"; mQ.field = "Q"; mQ.rhs = 0.0;
            bc.row["mU"] = mU;
            bc.row["mQ"] = mQ;
        }
    } catch (const std::exception& e) {
        if (rank == 0)
            std::cerr << "FATAL: " << e.what() << "\n";
        MPI_Finalize();
        return 1;
    }
    Trumpet::L0Model& m = *mp;
    const int ndom = m.nb_domains();
    const int dlast = ndom - 1;

    const auto inn = split2(inner);
    const auto out = split2(outer);
    const auto pin = split2(pins);
    // Kadath's deficit for three second-order rows on any number of 1-D domains
    // with C0+C1 matching is exactly SIX (see the header note).  How they are
    // distributed between inner rows, constraint pins and outer decay is the
    // open question SPEC_T22_innerBC and T4.1 answer differently, so the split
    // is a runtime choice and only the TOTAL is enforced.
    // How many PHYSICS conditions the budget wants.  Six with every equation at
    // its natural tau order; two fewer if the degenerate rows are declared
    // first-order somewhere (they then supply two more themselves).  The
    // order-3 reading also imposes two pointwise rows of its own, which is why
    // it leaves the physics total at six.
    int want_conditions = 6;
    if (horizonorder_is_o1(horder))
        want_conditions = 4;
    // Round 9: permit a deliberate shortfall, additive only.  Without the
    // appended rows a short budget leaves do_newton a singular system, which
    // would fail for a reason unrelated to the pin question.
    if (conddef != 0 && !additive) {
        if (rank == 0)
            std::cerr << "FATAL: --cond-deficit requires --additive\n";
        MPI_Finalize();
        return 2;
    }
    want_conditions -= conddef;
    if (inn.size() + out.size() + pin.size() != (std::size_t)want_conditions) {
        if (rank == 0)
            std::cerr << "FATAL: --inner (" << inn.size() << ") + --pins ("
                      << pin.size() << ") + --outer (" << out.size()
                      << ") must total exactly " << want_conditions << "\n";
        MPI_Finalize();
        return 2;
    }

    if (rank == 0) {
        m.print_banner();
        std::cout << "# T2.2 solve  j2=" << j2 << "  inner=" << inner
                  << "  outer=" << outer << "  seed=" << seed
                  << "  prec=" << prec << "\n";
        std::cout << "# bc tag=" << bc.tag << "  W0=" << bc.W0
                  << "  r_in=" << std::setprecision(17) << bc.r_in
                  << "  dWdr=" << bc.dWdr << "\n";
    }

    // ------------------------------------------------------ initial guess ----
    // Zero for a production solve.  For the uniqueness test a NONZERO start is
    // mandatory: do_newton computes the residual first and returns immediately
    // if it is already below precision, so a zero guess on a homogeneous
    // problem "passes" without ever assembling the Jacobian.
    m.set_fields_zero();
    if (seed != 0.0) {
        m.fill(m.U, [&](int, int i) { return seed * (1.0 + 0.1 * (i % 3)); });
        m.fill(m.Q, [&](int, int i) { return -seed * (1.0 + 0.2 * (i % 2)); });
        m.fill(m.G, [&](int d, int) { return seed * (0.5 + 0.25 * d); });
    }

    // ----------------------------------------------------------- system -----
    System_of_eqs syst(m.space, 0, dlast);
    const auto defs = m.register_rows(syst, j2);
    if (rank == 0)
        for (const auto& d : defs)
            std::cout << "# add_def  " << d << "\n";

    // the imported backbone, so dr(Wbb) at the excision can be checked against
    // the research session's own dW/dr -- a manufactured test of the whole
    // boundary-derivative path with an exactly known answer
    syst.add_cst("Wbb", m.Wf);
    syst.add_def("DWCHK = dr(Wbb)");

    // per-term definitions, used ONLY to build the row normalisation (largest
    // single term) for the constraint gate.  They must be compound expressions:
    // a bare der_abs() result read pointwise lives in COEFFICIENT space, which
    // is what made T2.1's first attempt report a scale of garbage.
    std::vector<std::vector<std::string>> termdef(5);
    for (int n = 0; n < 5; n++) {
        int t = 0;
        for (const auto& jet : m.coefs().jets.at(Trumpet::rows()[n])) {
            std::ostringstream nm;
            nm << "T" << Trumpet::row_codes()[n] << t++;
            const std::string d = nm.str() + " = c" + Trumpet::row_codes()[n] + jet
                                  + " * " + Trumpet::apply_jet(jet, Trumpet::field_of(jet));
            syst.add_def(d.c_str());
            termdef[n].push_back(nm.str());
        }
    }

    // Locate the interface at which the rows degenerate: the domain boundary
    // where E_K's only second-derivative coefficient vanishes against its own
    // grid maximum.  Found, never hardcoded -- a table without this structure
    // makes the horizon options fail loudly instead of acting on the wrong point.
    int dh = -1;
    {
        const auto& cv = m.coefs().v;
        const auto& kUpp = cv.at(std::make_pair(std::string("E_K"),
                                                std::string("Upp")));
        double mx = 0.0;
        for (int d = 0; d <= dlast; d++)
            for (int i = 0; i < m.nbr(d); i++)
                mx = std::max(mx, std::fabs(kUpp[d][i]));
        for (int d = 0; d < dlast && dh < 0; d++)
            if (std::fabs(kUpp[d][m.nbr(d) - 1]) < 1e-12 * mx)
                dh = d;
    }

    // --- which domains give up a tau order, per --horizon-order.  dh is the
    // domain whose OUTER face is r(2M); dh+1 is the one whose INNER face is.
    // TWO READINGS OF "lower the order at the degenerate edge", and they point
    // opposite ways in the tau bookkeeping:
    //   order 3  -- the equation supplies one FEWER condition on that domain, so
    //              two extra rows must be imposed (research Q1's literal form);
    //   order 1  -- the equation supplies one MORE, which is what "this equation
    //              is first-order here, so it needs one fewer boundary condition"
    //              means for a tau scheme; then two PHYSICS conditions come out
    //              and no extra rows go in.
    // Both are implemented so the choice is measured, not argued.
    bool lowerEK[64] = {false}, lowerEXT[64] = {false};
    // Where each pointwise row is imposed.  Defaults reproduce the earlier
    // behaviour (both rows on the inner domain's outer face) so the old
    // placements keep their recorded numbers.
    int rowEK_dom = -1, rowEK_bound = OUTER_BC;
    int rowCOMPAT_dom = -1, rowCOMPAT_bound = OUTER_BC;
    int taurder = 3;
    if (horder.rfind("o1@", 0) == 0) { taurder = 1; horder = horder.substr(3); }
    else if (horder.rfind("o3@", 0) == 0) { horder = horder.substr(3); }
    if (!horder.empty()) {
        if (dh < 0) {
            if (rank == 0)
                std::cerr << "FATAL: --horizon-order: no interface found where "
                             "E_K's Upp vanishes\n";
            MPI_Finalize();
            return 2;
        }
        // WHICH domain gives up the two tau orders is a MEASURED choice.  The
        // spend is fixed (both rows at the r(2M) interface, per research Q1);
        // the free is a placement, and a bare integer names the donor domain so
        // every candidate can be scanned rather than assumed.
        // ROUND-3 PRESCRIPTION.  Free on the degenerate pairs themselves and
        // give each donor its condition back AT ITS OWN degenerate face:
        //   r3   add_eq_order(3) on (E_K, d1) and (E_chi_tt, d2);
        //        E_K row at d1/OUTER_BC, (E_chi_tt - kappa E_K) row at d2/INNER_BC
        //   r3t  the transpose, rows swapped to follow their donors
        // Last session's "split" freed the same pair but spent BOTH rows at
        // d1/OUTER_BC, leaving d1 over-supplied and d2 short -- and d2 is exactly
        // what broke.  The pairing is the completion of that experiment.
        // Compact pairing spec so the whole matrix can be scanned rather than
        // enumerated by name:  p:<ek_donor><ext_donor>:<ek_row><compat_row>
        // Each digit is 1 (the domain whose OUTER face is r(2M)) or 2 (the one
        // whose INNER face is); a row placed on 1 goes to d1/OUTER_BC, on 2 to
        // d2/INNER_BC.  r3 is p:12:12 and r3t is p:21:21.
        if (horder.rfind("p:", 0) == 0 && horder.size() == 7 && horder[4] == ':') {
            auto dom = [&](char c) { return c == '1' ? dh : dh + 1; };
            auto bnd = [&](char c) { return c == '1' ? OUTER_BC : INNER_BC; };
            for (char c : {horder[2], horder[3], horder[5], horder[6]})
                if (c != '1' && c != '2') {
                    if (rank == 0)
                        std::cerr << "FATAL: --horizon-order p: digits must be 1 or 2\n";
                    MPI_Finalize();
                    return 2;
                }
            lowerEK[dom(horder[2])] = true;
            lowerEXT[dom(horder[3])] = true;
            rowEK_dom = dom(horder[5]);         rowEK_bound = bnd(horder[5]);
            rowCOMPAT_dom = dom(horder[6]);     rowCOMPAT_bound = bnd(horder[6]);
        } else if (horder == "r3") {
            lowerEK[dh] = true;      rowEK_dom = dh;      rowEK_bound = OUTER_BC;
            lowerEXT[dh + 1] = true; rowCOMPAT_dom = dh + 1; rowCOMPAT_bound = INNER_BC;
        } else if (horder == "r3t") {
            lowerEXT[dh] = true;     rowCOMPAT_dom = dh;  rowCOMPAT_bound = OUTER_BC;
            lowerEK[dh + 1] = true;  rowEK_dom = dh + 1;  rowEK_bound = INNER_BC;
        } else if (horder == "d1") {
            lowerEK[dh] = lowerEXT[dh] = true;
        } else if (horder == "d2") {
            lowerEK[dh + 1] = lowerEXT[dh + 1] = true;
        } else if (horder == "split") {
            lowerEK[dh] = true;
            lowerEXT[dh + 1] = true;
        } else if (horder.find_first_not_of("0123456789") == std::string::npos) {
            const int dd = std::stoi(horder);
            if (dd < 0 || dd > dlast) {
                if (rank == 0)
                    std::cerr << "FATAL: --horizon-order domain " << dd
                              << " out of range 0.." << dlast << "\n";
                MPI_Finalize();
                return 2;
            }
            lowerEK[dd] = lowerEXT[dd] = true;
        } else {
            if (rank == 0)
                std::cerr << "FATAL: --horizon-order must be r3, r3t, d1, d2, "
                             "split, o1@<dom> or a domain index\n";
            MPI_Finalize();
            return 2;
        }
        if (rowEK_dom < 0)     { rowEK_dom = dh;     rowEK_bound = OUTER_BC; }
        if (rowCOMPAT_dom < 0) { rowCOMPAT_dom = dh; rowCOMPAT_bound = OUTER_BC; }
    }

    // ---------------------------------------------- log enrichment ----------
    // Parse "<k>:<fields>[,<k>:<fields>]" into (power, field) amplitude slots,
    // register the analytic profiles and one double unknown per slot, and build
    // the per-row log columns E<row>L<k><field>.
    //
    // WHERE THE LOG ACTUALLY BITES.  The profiles are zero on every domain but
    // the last, so the amplitude columns of the inner rows, the constraint pins,
    // the compatibility rows and the outer decay rows are IDENTICALLY zero (at
    // r = infinity L, L' and L'' are all 0 too).  Only the bulk rows on the last
    // domain and the matching at the last interface see a nonzero column.  Those
    // two are written compound; the rest are left alone because adding a term
    // that is exactly zero would change nothing but the string.  LOGCOL_zero
    // below asserts that claim rather than trusting it.
    struct LogSlot { int k; std::string field, amp; std::array<std::string,3> L; };
    std::vector<LogSlot> logslots;
    std::vector<double> logamp;
    logamp.reserve(16);
    if (!logenrich.empty()) {
        for (const auto& part : split2(logenrich)) {
            const auto colon = part.find(':');
            if (colon == std::string::npos) {
                if (rank == 0)
                    std::cerr << "FATAL: --log-enrich expects <k>:<fields>\n";
                MPI_Finalize();
                return 2;
            }
            const int k = std::stoi(part.substr(0, colon));
            const std::string tagn = "Lg" + std::to_string(k);
            const auto L = m.enrich_log(syst, k, tagn);
            for (char c : part.substr(colon + 1)) {
                const std::string f(1, c);
                if (f != "U" && f != "Q" && f != "G") {
                    if (rank == 0)
                        std::cerr << "FATAL: --log-enrich field must be U, Q or G\n";
                    MPI_Finalize();
                    return 2;
                }
                logamp.push_back(0.0);
                LogSlot sl{k, f, "alg" + std::to_string(k) + f, L};
                syst.add_var(sl.amp.c_str(), logamp.back());
                logslots.push_back(sl);
            }
        }
        for (int n = 0; n < 5; n++)
            for (const auto& sl : logslots) {
                const std::string body = m.log_row(n, sl.field, sl.L);
                const std::string nm = std::string(Trumpet::row_defs()[n]) + "L"
                                       + std::to_string(sl.k) + sl.field;
                syst.add_def((nm + " = " + (body.empty() ? std::string("0 * ") + sl.L[0]
                                                         : body)).c_str());
            }
        if (rank == 0) {
            std::cout << "# logenrich " << logslots.size() << " amplitude(s):";
            for (const auto& sl : logslots)
                std::cout << " " << sl.amp << "=ln(r)/r^" << sl.k << " in " << sl.field;
            std::cout << "  (supported on d" << dlast << " only)\n";
        }
    }
    // row n's equation string, with the log terms appended when enriched
    auto eqstr = [&](int n) {
        std::string e = Trumpet::row_defs()[n];
        for (const auto& sl : logslots)
            e += " + " + sl.amp + " * " + Trumpet::row_defs()[n] + "L"
                 + std::to_string(sl.k) + sl.field;
        return e + " = 0";
    };
    // a field, plus its log content -- for matching, where the two sides of the
    // last interface see different profile values
    auto aug = [&](const std::string& f, int order) {
        std::string e = (order == 0) ? f : "dr(" + f + ")";
        for (const auto& sl : logslots)
            if (sl.field == f)
                e += " + " + sl.amp + " * " + sl.L[order];
        return e;
    };

    // --- bulk: the three evolution rows, in every domain.  E_XR is never
    // lowered: it keeps all three second-derivative slots at r(2M) and is the
    // one row that does not degenerate there.
    int freed = 0;
    for (int d = 0; d <= dlast; d++) {
        if (lowerEK[d])  { syst.add_eq_order(d, taurder, eqstr(0).c_str());  freed++; }
        else               syst.add_eq_inside(d, eqstr(0).c_str());
        syst.add_eq_inside(d, eqstr(1).c_str());
        if (lowerEXT[d]) { syst.add_eq_order(d, taurder, eqstr(2).c_str()); freed++; }
        else               syst.add_eq_inside(d, eqstr(2).c_str());
    }
    const auto hfix = horizonfix.empty() ? std::vector<std::string>()
                                         : split2(horizonfix);
    if (!hfix.empty()) {
        if (dh < 0) {
            if (rank == 0)
                std::cerr << "FATAL: --horizon-fix: no interface found where E_K's "
                             "Upp vanishes\n";
            MPI_Finalize();
            return 2;
        }
        if (hfix.size() != 2) {
            if (rank == 0)
                std::cerr << "FATAL: --horizon-fix takes exactly two fields (two C1 "
                             "matchings out, two pointwise rows in)\n";
            MPI_Finalize();
            return 2;
        }
        for (const auto& f : hfix)
            if (f != "U" && f != "Q" && f != "G") {
                if (rank == 0)
                    std::cerr << "FATAL: --horizon-fix fields must be U, Q or G\n";
                MPI_Finalize();
                return 2;
            }
    }
    // --- C0 + C1 matching of all three fields at every interface
    for (int d = 0; d < dlast; d++)
        for (const char* f : {"U", "Q", "G"}) {
            syst.add_eq_matching(d, OUTER_BC, aug(f, 0).c_str());
            if (d == dh
                && std::find(hfix.begin(), hfix.end(), std::string(f)) != hfix.end()) {
                if (rank == 0)
                    std::cout << "# hfix   C1 matching of " << f << " at d" << d
                              << "/OUTER_BC DROPPED\n";
                continue;
            }
            syst.add_eq_matching(d, OUTER_BC, aug(f, 1).c_str());
        }
    // --- THE FIX: spend the freed conditions on the two pointwise rows the
    // degeneracy leaves unimposed at r(2M).  Research round-2 Q1 names them as
    // E_K = 0 and (E_chi_tt - 2 E_K) = 0; the DIFFERENCE form matters
    // numerically, not just formally -- at that point it has exactly two
    // nonzero slots (Qp and j2) while bare E_chi_tt has six of order 1e4..1e5
    // that must cancel.  kappa is MEASURED one point inside (it is 0/0 at the
    // interface itself) and asserted, so a table without this structure fails
    // loudly rather than imposing a wrong row.
    if (!horder.empty() && taurder == 3) {
        const auto& cv = m.coefs().v;
        auto C = [&](const char* row, const char* jet, int d, int i) {
            return cv.at(std::make_pair(std::string(row), std::string(jet)))[d][i];
        };
        const int ih = m.nbr(dh) - 1;
        const double k1 = C("E_chi_tt", "Upp", dh, ih - 1) / C("E_K", "Upp", dh, ih - 1);
        const double k2 = C("E_chi_tt", "Upp", dh, ih - 2) / C("E_K", "Upp", dh, ih - 2);
        if (std::fabs(k1 - k2) > 1e-10 * std::fabs(k1)) {
            if (rank == 0)
                std::cerr << "FATAL: --horizon-order: E_chi_tt.Upp / E_K.Upp is not "
                             "constant near the interface (" << k1 << " vs " << k2
                          << ")\n";
            MPI_Finalize();
            return 2;
        }
        double rowmax = 0.0;
        for (const char* jt : {"G", "Gp", "Q", "Qp", "U", "Up", "Upp"})
            rowmax = std::max(rowmax, std::fabs(C("E_chi_tt", jt, dh, ih)));
        for (const char* jt : {"G", "Gp", "Q", "U", "Up", "Upp"}) {
            const double v = C("E_chi_tt", jt, dh, ih) - k1 * C("E_K", jt, dh, ih);
            if (std::fabs(v) > 1e-12 * rowmax) {
                if (rank == 0)
                    std::cerr << "FATAL: --horizon-order: slot " << jt
                              << " of E_chi_tt - " << k1 << "*E_K is " << v
                              << ", not zero\n";
                MPI_Finalize();
                return 2;
            }
        }
        const double cQp = C("E_chi_tt", "Qp", dh, ih) - k1 * C("E_K", "Qp", dh, ih);
        const double cj2 = -k1 * C("E_K", "j2", dh, ih);
        // ROW NORMALISATION CHANGES THE CONSTANT.  With --rownorm every
        // coefficient of row n at (d,i) is divided by that row's OWN
        // max_j|c_{n,j}|, so the registered defs are E_K/N_K and E_chi_tt/N_T
        // with N_T != N_K.  The string "EXT - kappa*EK" would then mean
        // E_chi_tt/N_T - kappa*E_K/N_K, which is NOT proportional to
        // E_chi_tt - kappa*E_K and destroys the bit-zero cancellation that is
        // the whole point of the difference form.  (Measured here: N_T = 2*N_K
        // at the interface, so the naive string imposes E_chi_tt - 4*E_K.)
        // Rescale by the normaliser ratio; without --rownorm both are 1.
        // kappa_eff is evaluated AT THE FACE WHERE THE ROW IS IMPOSED, not
        // reused from d1's last point.  The two sides of the interface carry
        // identical coefficients here (measured), so the two are equal in
        // practice -- but the code must not assume what it can compute.
        const int ic = (rowCOMPAT_bound == OUTER_BC) ? m.nbr(rowCOMPAT_dom) - 1 : 0;
        const double kap_eff = k1 * m.row_norm(0, rowCOMPAT_dom, ic)
                                  / m.row_norm(2, rowCOMPAT_dom, ic);
        syst.add_cst("hkap", kap_eff);
        syst.add_def("ECOMPAT = EXT - hkap * EK");
        syst.add_eq_bc(rowEK_dom, rowEK_bound, "EK = 0");
        syst.add_eq_bc(rowCOMPAT_dom, rowCOMPAT_bound, "ECOMPAT = 0");
        if (rank == 0) {
            std::cout << "# horder placement=" << horder << "  interface d" << dh
                      << "|d" << dh + 1 << "  kappa=" << k1
                      << "  kappa_eff(after rownorm)=" << kap_eff << "\n";
            std::cout << "# horder add_eq_order(.,3,.) on:";
            for (int d = 0; d <= dlast; d++) {
                if (lowerEK[d])  std::cout << " EK@d" << d;
                if (lowerEXT[d]) std::cout << " EXT@d" << d;
            }
            std::cout << "   conditions freed = " << freed << "\n";
            auto bnd = [](int b) { return b == OUTER_BC ? "OUTER_BC" : "INNER_BC"; };
            std::cout << "# horder add_eq_bc(" << rowEK_dom << ", " << bnd(rowEK_bound)
                      << ", \"EK = 0\")  and  add_eq_bc(" << rowCOMPAT_dom << ", "
                      << bnd(rowCOMPAT_bound) << ", \"ECOMPAT = 0\")   -> spent = 2\n";
            std::cout << "# horder implied dr(Q) at r(2M) = " << std::setprecision(17)
                      << -cj2 / cQp << " * j2\n";
            Trumpet::emit("HORDER_freed", freed);
            Trumpet::emit("HORDER_kappa", k1);
            Trumpet::emit("HORDER_compat_per_j2", -cj2 / cQp);
        }
    }

    if (!hfix.empty()) {
        if (hside == "left") {
            syst.add_eq_bc(dh, OUTER_BC, "EK = 0");
            syst.add_eq_bc(dh, OUTER_BC, "EXT = 0");
            if (rank == 0)
                std::cout << "# hfix   add_eq_bc(" << dh << ", OUTER_BC, \"EK = 0\")"
                             "  and  \"EXT = 0\"\n";
        } else if (hside == "right") {
            syst.add_eq_bc(dh + 1, INNER_BC, "EK = 0");
            syst.add_eq_bc(dh + 1, INNER_BC, "EXT = 0");
            if (rank == 0)
                std::cout << "# hfix   add_eq_bc(" << dh + 1 << ", INNER_BC, "
                             "\"EK = 0\")  and  \"EXT = 0\"\n";
        } else if (hside == "split") {
            syst.add_eq_bc(dh, OUTER_BC, "EK = 0");
            syst.add_eq_bc(dh + 1, INNER_BC, "EXT = 0");
            if (rank == 0)
                std::cout << "# hfix   add_eq_bc(" << dh << ", OUTER_BC, \"EK = 0\")"
                             "  and  add_eq_bc(" << dh + 1
                          << ", INNER_BC, \"EXT = 0\")\n";
        } else if (hside == "split2") {
            syst.add_eq_bc(dh, OUTER_BC, "EXT = 0");
            syst.add_eq_bc(dh + 1, INNER_BC, "EK = 0");
            if (rank == 0)
                std::cout << "# hfix   add_eq_bc(" << dh << ", OUTER_BC, \"EXT = 0\")"
                             "  and  add_eq_bc(" << dh + 1
                          << ", INNER_BC, \"EK = 0\")\n";
        } else {
            if (rank == 0)
                std::cerr << "FATAL: --horizon-side must be left, right, split or "
                             "split2\n";
            MPI_Finalize();
            return 2;
        }
    }

    // --- inner: two rows at the excision surface
    std::vector<std::string> inner_eq;
    for (const auto& nm : inn) {
        auto it = bc.row.find(nm);
        if (it == bc.row.end()) {
            if (rank == 0)
                std::cerr << "FATAL: no bc row '" << nm << "' in " << argv[3] << "\n";
            MPI_Finalize();
            return 2;
        }
        // RHS folded to j2 * value in C++ so the string carries one constant
        const std::string cnm = "bc" + nm;
        // The BC data is quoted per unit j2, so the imposed value is j2*rhs --
        // except the manufactured rows, which are absolute (they come from the
        // backbone, not from the j2 ladder).
        const double rhs = manufactured ? it->second.rhs : j2 * it->second.rhs;
        syst.add_cst(cnm.c_str(), rhs);
        const std::string lhs = bc_lhs(it->second, "r" + nm, syst);
        inner_eq.push_back(lhs + " = " + cnm);
        syst.add_eq_bc(0, INNER_BC, inner_eq.back().c_str());
    }
    // --- HORIZON COMPATIBILITY ROW ("compat"), a DIAGNOSTIC allocation.
    //
    // At R = 2M the second-derivative coefficients of E_K (Upp) and E_chi_tt
    // (Upp and Qpp) all vanish exactly, and E_chi_tt's Upp is exactly twice
    // E_K's at every point.  So at that ONE point the combination
    // E_chi_tt - 2*E_K has every slot bit-zero except Qp and j2, and the
    // system forces an algebraic COMPATIBILITY CONDITION on any true solution:
    //
    //     dr(Q)|_{R=2M}  =  -c_j2 / c_Qp * j2
    //
    // The tau method never imposes a pointwise condition, so nothing makes the
    // discrete solution satisfy it -- measured violation is ~1200x.  This block
    // imposes it explicitly, as ONE of the six conditions (so it must displace
    // another; the existing total-6 check enforces that).  It is a diagnostic
    // for the research session's adjudication of the degeneracy, never a gate.
    //
    // The interface is LOCATED, not hardcoded: it is the domain boundary where
    // E_K's only second-derivative coefficient vanishes against its own grid
    // maximum.  Everything else is asserted, so a table that does not have this
    // structure makes the run fail loudly rather than impose a wrong number.
    double compat = 0.0;
    const bool want_compat =
        std::find(pin.begin(), pin.end(), std::string("compat")) != pin.end();
    const bool want_hEK =
        std::find(pin.begin(), pin.end(), std::string("hEK")) != pin.end();
    const bool want_hEXT =
        std::find(pin.begin(), pin.end(), std::string("hEXT")) != pin.end();
    if (want_compat || want_hEK || want_hEXT) {
        const auto& cv = m.coefs().v;
        auto C = [&](const char* row, const char* jet, int d, int i) {
            return cv.at(std::make_pair(std::string(row), std::string(jet)))[d][i];
        };
        if (dh < 0) {
            if (rank == 0)
                std::cerr << "FATAL: --pins compat: no interface found where "
                             "E_K's Upp vanishes\n";
            MPI_Finalize();
            return 2;
        }
        const int ih = m.nbr(dh) - 1;
        // kappa = E_chi_tt.Upp / E_K.Upp, taken one point INSIDE (it is 0/0 at
        // the interface itself) and required to be the same 2 at two points.
        const double k1 = C("E_chi_tt", "Upp", dh, ih - 1) / C("E_K", "Upp", dh, ih - 1);
        const double k2 = C("E_chi_tt", "Upp", dh, ih - 2) / C("E_K", "Upp", dh, ih - 2);
        if (std::fabs(k1 - k2) > 1e-10 * std::fabs(k1)) {
            if (rank == 0)
                std::cerr << "FATAL: --pins compat: E_chi_tt.Upp / E_K.Upp is not "
                             "constant near the interface (" << k1 << " vs " << k2
                          << ")\n";
            MPI_Finalize();
            return 2;
        }
        // every slot of E_chi_tt - kappa*E_K must be bit-zero at the interface
        // except Qp and j2 -- that IS the content of the condition
        static const char* SLOT[] = {"G", "Gp", "Q", "U", "Up", "Upp"};
        double rowmax = 0.0;
        for (const char* jt : {"G", "Gp", "Q", "Qp", "U", "Up", "Upp"})
            rowmax = std::max(rowmax, std::fabs(C("E_chi_tt", jt, dh, ih)));
        for (const char* jt : SLOT) {
            const double v = C("E_chi_tt", jt, dh, ih) - k1 * C("E_K", jt, dh, ih);
            if (std::fabs(v) > 1e-12 * rowmax) {
                if (rank == 0)
                    std::cerr << "FATAL: --pins compat: slot " << jt << " of "
                                 "E_chi_tt - " << k1 << "*E_K is " << v
                              << ", not zero\n";
                MPI_Finalize();
                return 2;
            }
        }
        const double cQp = C("E_chi_tt", "Qp", dh, ih) - k1 * C("E_K", "Qp", dh, ih);
        // E_chi_tt carries no j2 column, so the j2 part is -kappa * E_K.j2
        const double cj2 = -k1 * C("E_K", "j2", dh, ih);
        compat = -cj2 / cQp * j2;
        if (rank == 0) {
            std::cout << "# compat  interface d" << dh << "/OUTER_BC  kappa=" << k1
                      << "  cQp=" << std::setprecision(17) << cQp
                      << "  cj2=" << cj2 << "\n";
            Trumpet::emit("COMPAT_target", compat);
            Trumpet::emit("COMPAT_per_j2", j2 != 0.0 ? compat / j2 : 0.0);
        }
        if (want_compat) {
            syst.add_cst("compatv", compat);
            syst.add_eq_bc(dh, OUTER_BC, "dr(Q) = compatv");
            if (rank == 0)
                std::cout << "# compat  add_eq_bc(" << dh << ", OUTER_BC, "
                             "\"dr(Q) = " << compat << "\")\n";
        }
        // hEK / hEXT: impose the degenerate rows POINTWISE at the interface.
        // add_eq_inside imposes them in the tau sense, which drops the top two
        // Chebyshev coefficients -- the right count for a second-order equation
        // but not for one that has lost its second-order content at this point.
        // Two of the three are degenerate here, so there are exactly TWO
        // independent conditions to recover, and "compat" is their difference.
        if (want_hEK) {
            syst.add_eq_bc(dh, OUTER_BC, "EK = 0");
            if (rank == 0)
                std::cout << "# compat  add_eq_bc(" << dh
                          << ", OUTER_BC, \"EK = 0\")\n";
        }
        if (want_hEXT) {
            syst.add_eq_bc(dh, OUTER_BC, "EXT = 0");
            if (rank == 0)
                std::cout << "# compat  add_eq_bc(" << dh
                          << ", OUTER_BC, \"EXT = 0\")\n";
        }
    }

    // --- constraint pins, same boundary
    for (const auto& p : pin) {
        if (p == "compat" || p == "hEK" || p == "hEXT")
            continue;                       // handled above, at its own interface
        if (p != "EH" && p != "EM") {
            if (rank == 0)
                std::cerr << "FATAL: --pins entries must be EH, EM, compat, hEK "
                             "or hEXT, got " << p << "\n";
            MPI_Finalize();
            return 2;
        }
        if (pinat == "inner")
            syst.add_eq_bc(0, INNER_BC, (p + " = 0").c_str());
        else if (pinat == "outer")
            syst.add_eq_bc(dlast, OUTER_BC, (p + " = 0").c_str());
        else {
            if (rank == 0)
                std::cerr << "FATAL: --pin-at must be inner or outer\n";
            MPI_Finalize();
            return 2;
        }
    }
    // --- outer: decay conditions at r = infinity.  An entry is "F" (meaning
    // F = 0) or "F=value" -- the manufactured BVP needs G(inf) = -2.
    std::vector<std::pair<std::string, double>> outset;
    for (const auto& f : out) {
        const auto eq = f.find('=');
        const std::string fld = (eq == std::string::npos) ? f : f.substr(0, eq);
        const double tgt = (eq == std::string::npos) ? 0.0 : std::stod(f.substr(eq + 1));
        outset.emplace_back(fld, tgt);
        const std::string cnm = "out" + fld;
        syst.add_cst(cnm.c_str(), tgt);
        syst.add_eq_bc(dlast, OUTER_BC, (fld + " = " + cnm).c_str());
    }

    // --- ROUND-4 APPENDED ROWS.  Registered last, so they are the final three
    // eq_index values and are identifiable in the row metadata.  Every one is an
    // exact property of the continuum solution:
    //   1  E_K pointwise at r(2M)                     -- theorem: implied by
    //   2  (E_chi_tt - kappa E_K) pointwise at r(2M)     evolution + the pins
    //   3  outer G decay                              -- (0,0,1) is a non-solution
    // so the rectangular system is CONSISTENT at truncation level and the
    // least-squares solution is the square solution plus the two bits the tau
    // projection drops.  Row 3 also retires the G-constant near-null.
    int n_appended = 0;
    if (additive) {
        if (dh < 0) {
            if (rank == 0)
                std::cerr << "FATAL: --additive: no interface found where E_K's "
                             "Upp vanishes\n";
            MPI_Finalize();
            return 2;
        }
        const auto& cv = m.coefs().v;
        auto C = [&](const char* row, const char* jet, int d, int i) {
            return cv.at(std::make_pair(std::string(row), std::string(jet)))[d][i];
        };
        const int ih = m.nbr(dh) - 1;
        const double k1 = C("E_chi_tt", "Upp", dh, ih - 1) / C("E_K", "Upp", dh, ih - 1);
        const double k2 = C("E_chi_tt", "Upp", dh, ih - 2) / C("E_K", "Upp", dh, ih - 2);
        if (std::fabs(k1 - k2) > 1e-10 * std::fabs(k1)) {
            if (rank == 0)
                std::cerr << "FATAL: --additive: E_chi_tt.Upp / E_K.Upp is not "
                             "constant near the interface\n";
            MPI_Finalize();
            return 2;
        }
        double rowmax = 0.0;
        for (const char* jt : {"G", "Gp", "Q", "Qp", "U", "Up", "Upp"})
            rowmax = std::max(rowmax, std::fabs(C("E_chi_tt", jt, dh, ih)));
        for (const char* jt : {"G", "Gp", "Q", "U", "Up", "Upp"}) {
            const double v = C("E_chi_tt", jt, dh, ih) - k1 * C("E_K", jt, dh, ih);
            if (std::fabs(v) > 1e-12 * rowmax) {
                if (rank == 0)
                    std::cerr << "FATAL: --additive: slot " << jt << " of E_chi_tt - "
                              << k1 << "*E_K is " << v << ", not zero\n";
                MPI_Finalize();
                return 2;
            }
        }
        const double kap_eff = k1 * m.row_norm(0, dh, ih) / m.row_norm(2, dh, ih);
        syst.add_cst("hkap", kap_eff);
        syst.add_def("ECOMPAT = EXT - hkap * EK");
        if (addrows == "all" || addrows == "compat") {
            syst.add_eq_bc(dh, OUTER_BC, "EK = 0");
            syst.add_eq_bc(dh, OUTER_BC, "ECOMPAT = 0");
            n_appended += 2;
        }
        // ROW 3 AND ITS TARGET.  Round 4 says the mass mode satisfies every
        // appended row exactly at j2 = 0; that holds for rows 1 and 2 but NOT for
        // this one -- the manufactured mass mode has G(inf) = -2, not 0 (see the
        // --manufactured note above, and OUTVAL_G = -1.9999999999999996 in the
        // control).  Targeting 0 there would fail P1 for a reason that has nothing
        // to do with the formulation, so the target follows the same convention
        // --outer already uses.
        const double gtarget = manufactured ? -2.0 : 0.0;
        if (addrows == "all" || addrows == "g") {
            syst.add_cst("outGadd", gtarget);
            syst.add_eq_bc(dlast, OUTER_BC, "G = outGadd");
            n_appended += 1;
        }
        if (n_appended == 0) {
            if (rank == 0)
                std::cerr << "FATAL: --add-rows must be all, compat or g\n";
            MPI_Finalize();
            return 2;
        }
        if (rank == 0) {
            std::cout << "# additive kappa=" << k1 << "  kappa_eff=" << kap_eff
                      << "  outer-G target=" << gtarget << "\n";
            // --add-rows selects which of these are actually registered, so
            // the roster must be built from addrows, not printed as a fixed
            // string (it read "all three" under --add-rows compat).
            std::cout << "# additive appended:";
            if (addrows == "all" || addrows == "compat")
                std::cout << " EK@d" << dh << "/OUTER_BC, ECOMPAT@d" << dh
                          << "/OUTER_BC";
            if (addrows == "all" || addrows == "g")
                std::cout << (addrows == "all" ? ", " : " ") << "G@d" << dlast
                          << "/OUTER_BC";
            std::cout << "  (n_appended = " << n_appended << ")\n";
            Trumpet::emit("ADD_kappa_eff", kap_eff);
            Trumpet::emit("ADD_G_target", gtarget);
        }
    }

    if (rank == 0) {
        for (const auto& e : inner_eq)
            std::cout << "# inner  add_eq_bc(0, INNER_BC, \"" << e << "\")\n";
        for (const auto& p : pin)
            if (p != "compat" && p != "hEK" && p != "hEXT")
                std::cout << "# pin    add_eq_bc("
                          << (pinat == "inner" ? 0 : dlast) << ", "
                          << (pinat == "inner" ? "INNER_BC" : "OUTER_BC")
                          << ", \"" << p << " = 0\")\n";
        for (const auto& o : outset)
            std::cout << "# outer  add_eq_bc(" << dlast << ", OUTER_BC, \"" << o.first
                      << " = " << o.second << "\")\n";
        std::cout << "# conditions " << syst.get_nbr_conditions() << "  unknowns "
                  << syst.get_nbr_unknowns() << "\n";
    }
    if (rank == 0) {
        std::cout << "# rownorm " << (rownorm ? "ON" : "off") << "  row normaliser"
                  << " max/min:";
        for (int n = 0; n < 5; n++)
            std::cout << " " << Trumpet::rows()[n] << "=" << m.norm_spread(n);
        std::cout << "\n";
    }

    // ------------------------------------------------------------- solve ----
    // Linear system: one Newton step is exact, so a second iteration only
    // measures the residual of the first.  Iterating to `prec` and reporting
    // the count keeps that visible instead of assuming it.
    // do_newton's `error` is an ABSOLUTE max residual, and this problem's
    // solution scales linearly with j2, so its residual floor does too: the
    // same solve reads 3.4e-13 at j2 = 1e-6 and 3.4e-7 at j2 = 1.  An absolute
    // threshold is therefore meaningless here.  Convergence is judged on
    // err/err0 instead, and err0 is the residual of the zero (or seeded) guess.
    double err = 0.0, err0 = -1.0;
    int iter = 0;
    bool ok = false;
    if (additive) {
        // ---------------------------------------- additive least squares ----
        // The system is LINEAR, so one step from a zero guess is exact and the
        // Jacobian is the operator itself.  sec_member() and do_col_J() carry no
        // squareness assertion, so the 342x339 stack is extracted from Kadath
        // rather than hand-assembled: the core 339 rows are literally the
        // baseline system's rows, in the same order.
        m.set_fields_zero();
        Kadath::Array<double> bb(syst.sec_member());
        const int nrow = syst.get_nbr_conditions();
        const int ncol = syst.get_nbr_unknowns();
        const int ncore = nrow - n_appended;
        std::vector<double> A(std::size_t(nrow) * ncol, 0.0), rhs(nrow, 0.0);
        for (int r = 0; r < nrow; r++)
            rhs[r] = bb(r);
        for (int c = 0; c < ncol; c++) {
            Kadath::Array<double> col(syst.do_col_J(c));
            for (int r = 0; r < nrow; r++)
                A[std::size_t(c) * nrow + r] = col(r);       // column-major for LAPACK
        }
        // EQUILIBRATION.  The core rows of this system span many orders (matching
        // rows, tau rows and BC rows are not commensurate), and for a LEAST
        // SQUARES problem row scaling is not neutral -- it decides what is being
        // minimised.  So every row is scaled to unit norm first, which makes
        // "consistent" mean the same thing for all of them, and the appended rows
        // then carry `addweight` on top (1 = the same weight as a core row, which
        // is research's "equilibrate to the median core-row norm" once the core is
        // itself equilibrated).  addweight is a sensitivity knob, reported.
        std::vector<double> rnorm(nrow, 0.0), rscale(nrow, 1.0);
        for (int r = 0; r < nrow; r++) {
            double t = 0.0;
            for (int c = 0; c < ncol; c++)
                t += A[std::size_t(c) * nrow + r] * A[std::size_t(c) * nrow + r];
            rnorm[r] = std::sqrt(t);
        }
        double medcore;
        {
            std::vector<double> cc(rnorm.begin(), rnorm.begin() + ncore);
            std::nth_element(cc.begin(), cc.begin() + ncore / 2, cc.end());
            medcore = cc[ncore / 2];
        }
        for (int r = 0; r < nrow; r++) {
            double t = 0.0;
            for (int c = 0; c < ncol; c++)
                t += A[std::size_t(c) * nrow + r] * A[std::size_t(c) * nrow + r];
            rnorm[r] = std::sqrt(t);
            if (rnorm[r] <= 0.0)
                continue;
            if (addequil == "rows")
                rscale[r] = (r >= ncore ? addweight : 1.0) / rnorm[r];
            else if (addequil == "appended")
                rscale[r] = (r >= ncore) ? addweight * medcore / rnorm[r] : 1.0;
            else if (addequil == "none")
                rscale[r] = 1.0;
        }
        std::vector<double> core(rnorm.begin(), rnorm.begin() + ncore);
        std::nth_element(core.begin(), core.begin() + ncore / 2, core.end());
        const double med = core[ncore / 2];
        std::vector<double> Awork(std::size_t(nrow) * ncol, 0.0);
        std::vector<double> bwork(std::max(nrow, ncol), 0.0);
        for (int r = 0; r < nrow; r++) {
            for (int c = 0; c < ncol; c++)
                Awork[std::size_t(c) * nrow + r] = A[std::size_t(c) * nrow + r] * rscale[r];
            bwork[r] = rhs[r] * rscale[r];
        }
        std::vector<double> sv(std::min(nrow, ncol), 0.0);
        int mm = nrow, nn = ncol, nrhs = 1, lda = nrow, ldb = std::max(nrow, ncol);
        int rank_out = 0, info = 0, lwork = -1;
        // rcond NEGATIVE: machine precision, so dgelsd truncates nothing that is
        // genuinely there.  A too-large rcond silently returns the minimum-norm
        // solution of a TRUNCATED problem -- measured: rcond 1e-13 against
        // sv_max 2.1e6 cut at 2.1e-7, above the true sv_min 2.6e-8, and P1 read
        // 0.55 for that reason alone.  ADD_rank is reported so it is never
        // invisible.
        double rcond = -1.0, wkopt = 0.0;
        std::vector<int> iwork(std::size_t(11) * ncol + 3 * ncol * 32 + 256, 0);
        std::vector<double> Atmp(Awork), btmp(bwork);
        dgelsd_(&mm, &nn, &nrhs, Atmp.data(), &lda, btmp.data(), &ldb, sv.data(),
                &rcond, &rank_out, &wkopt, &lwork, iwork.data(), &info);
        lwork = int(wkopt);
        std::vector<double> work(std::max(lwork, 1), 0.0);
        const double scale_used = medcore;

        // ITERATIVE REFINEMENT.  One step is exact in exact arithmetic -- the
        // problem is linear -- but this operator's condition number is ~1e11 and a
        // single dense SVD solve lands three to four orders short of what MUMPS
        // reaches on the same square system (measured: 3.5e-7 against 6.5e-10 on
        // the manufactured control with the appended rows switched off).  The gap
        // is refinement, which Kadath's Newton loop gets for free by iterating.
        // Recomputing sec_member() from the FIELDS each pass makes the residual
        // honest rather than a running linear-algebra estimate; the matrix is
        // unchanged, so only the right-hand side is rebuilt.
        double post_core = 0.0, post_add = 0.0, rescore = 0.0, resadd = 0.0;
        int refine = 0;
        for (refine = 0; refine < 8; refine++) {
            Kadath::Array<double> res(refine == 0 ? bb : syst.sec_member());
            for (int r = 0; r < nrow; r++)
                btmp[r] = res(r) * rscale[r];
            for (int r = nrow; r < ldb; r++)
                btmp[r] = 0.0;
            Atmp = Awork;
            dgelsd_(&mm, &nn, &nrhs, Atmp.data(), &lda, btmp.data(), &ldb, sv.data(),
                    &rcond, &rank_out, &work[0], &lwork, iwork.data(), &info);
            if (info != 0) {
                if (rank == 0)
                    std::cerr << "FATAL: dgelsd returned info = " << info << "\n";
                MPI_Finalize();
                return 1;
            }
            Kadath::Array<double> xx(ncol);
            for (int cc2 = 0; cc2 < ncol; cc2++)
                xx.set(cc2) = btmp[cc2];
            int conte = 0;
            syst.xx_to_vars_delta(xx, conte);
            // residual FROM THE FIELDS, split core / appended -- P2
            Kadath::Array<double> b2(syst.sec_member());
            double pc = 0.0, pa = 0.0;
            for (int r = 0; r < nrow; r++) {
                double& acc = (r >= ncore ? pa : pc);
                acc = std::max(acc, std::fabs(b2(r)));
            }
            if (rank == 0)
                std::cout << "# additive refine " << refine + 1 << "  core " << pc
                          << "  appended " << pa << "\n";
            const bool stalled = (refine > 0 && pc > 0.5 * post_core);
            post_core = pc;
            post_add = pa;
            if (stalled)
                break;
        }
        rescore = post_core;
        resadd = post_add;
        // LOGCOL_zero: the claim that only the last domain's bulk rows and the
        // last matching see a nonzero amplitude column.  Measured, not assumed:
        // the largest |column| entry outside those rows must be exactly 0.
        if (!logslots.empty() && rank == 0) {
            double outside = 0.0;
            const int ncol_poly = ncol - int(logslots.size());
            for (int c = ncol_poly; c < ncol; c++)
                for (int r = 0; r < nrow; r++)
                    outside = std::max(outside, std::fabs(A[std::size_t(c) * nrow + r]));
            Trumpet::emit("LOGCOL_max", outside);
        }
        // WHERE the core residual lives.  If the least-squares has to sacrifice
        // core rows to satisfy the appended ones, which rows it sacrifices is the
        // whole diagnosis: tau rows on the horizon domains would mean the appended
        // rows are exposing the tau blind spot, anything else means they are
        // over-constraining.
        if (rank == 0) {
            std::vector<Kadath::System_of_eqs::RowMetadata> rmeta;
            syst.classify_equation_row_metadata(rmeta);
            Kadath::Array<double> b3(syst.sec_member());
            std::vector<std::pair<double, int>> ord;
            for (int r = 0; r < ncore; r++)
                ord.emplace_back(std::fabs(b3(r)), r);
            std::sort(ord.rbegin(), ord.rend());
            auto taxname = [](Kadath::RowTaxonomy t) -> const char* {
                switch (t) {
                    case Kadath::RowTaxonomy::Vol: return "Vol";
                    case Kadath::RowTaxonomy::TauBc: return "TauBc";
                    case Kadath::RowTaxonomy::TauMatch: return "TauMatch";
                    default: return "other";
                }
            };
            std::cout << "# additive worst core rows (|res|, taxonomy, dom, eq, mode):\n";
            for (int k = 0; k < 8 && k < int(ord.size()); k++) {
                const auto& rm = rmeta[ord[k].second];
                std::cout << "#   " << ord[k].first << "  " << taxname(rm.taxonomy)
                          << " d" << rm.dom << " eq" << rm.eq_index
                          << " mode" << rm.basis_mode << "\n";
            }
            std::map<std::string, double> bytax;
            std::map<int, double> bydom;
            for (int r = 0; r < ncore; r++) {
                bytax[taxname(rmeta[r].taxonomy)] =
                    std::max(bytax[taxname(rmeta[r].taxonomy)], std::fabs(b3(r)));
                bydom[rmeta[r].dom] = std::max(bydom[rmeta[r].dom], std::fabs(b3(r)));
            }
            std::cout << "# additive core residual by taxonomy:";
            for (const auto& kv : bytax)
                std::cout << " " << kv.first << "=" << kv.second;
            std::cout << "\n# additive core residual by domain:";
            for (const auto& kv : bydom)
                std::cout << " d" << kv.first << "=" << kv.second;
            std::cout << "\n";
            // as RESULT lines too, so a gate or a sweep table can read them
            for (const auto& kv : bydom)
                Trumpet::emit("ADD_res_core_d" + std::to_string(kv.first), kv.second);
            for (const auto& kv : bytax)
                Trumpet::emit("ADD_res_core_" + kv.first, kv.second);
        }
        err = post_core;
        err0 = 0.0;
        for (int r = 0; r < nrow; r++)
            err0 = std::max(err0, std::fabs(bb(r)));
        ok = true;
        iter = 1;
        if (rank == 0) {
            std::cout << "# additive  " << nrow << " x " << ncol << "  core " << ncore
                      << " + appended " << n_appended << "  median core-row norm "
                      << med << "  appended scale " << scale_used << "\n";
            Trumpet::emit("ADD_rows", nrow);
            Trumpet::emit("ADD_cols", ncol);
            Trumpet::emit("ADD_rank", rank_out);
            Trumpet::emit("ADD_sv_min", sv[std::min(nrow, ncol) - 1]);
            Trumpet::emit("ADD_sv_next", sv[std::min(nrow, ncol) - 2]);
            Trumpet::emit("ADD_sv_max", sv[0]);
            Trumpet::emit("ADD_res_core", post_core);
            Trumpet::emit("ADD_res_appended", post_add);
            Trumpet::emit("ADD_lsq_res_core", rescore);
            Trumpet::emit("ADD_lsq_res_appended", resadd);
            Trumpet::emit("ADD_scale", scale_used);
            Trumpet::emit("ADD_weight", addweight);
            for (std::size_t j = 0; j < logslots.size(); j++)
                Trumpet::emit("LOGAMP_" + std::to_string(logslots[j].k) + "_"
                              + logslots[j].field, logamp[j]);
        }
    }
    try {
        while (!additive && iter < 12) {
            iter++;
            ok = syst.do_newton(prec, err);
            if (err0 < 0.0)
                err0 = err;
            if (rank == 0)
                std::cout << "# newton iter " << iter << "  error " << err << "\n";
            if (ok || (err0 > 0.0 && err / err0 < 1e-12))
                break;
        }
    } catch (const std::exception& e) {
        if (rank == 0)
            std::cerr << "FATAL: do_newton threw: " << e.what() << "\n";
        MPI_Finalize();
        return 1;
    }
    // nbr_conditions is filled in by do_newton, not by add_eq_*, so this read
    // is only meaningful here -- before the solve it is still -1.
    Trumpet::emit("SQ_conditions", syst.get_nbr_conditions());
    Trumpet::emit("SQ_unknowns", syst.get_nbr_unknowns());
    Trumpet::emit("SQ_defect",
                  double(syst.get_nbr_conditions() - syst.get_nbr_unknowns()));
    Trumpet::emit("CONV_error", err);
    Trumpet::emit("CONV_error0", err0);
    Trumpet::emit("CONV_rel", err0 > 0.0 ? err / err0 : 0.0);
    Trumpet::emit("CONV_iters", iter);
    Trumpet::emit("CONV_ok", (ok || (err0 > 0.0 && err / err0 < 1e-12)) ? 1.0 : 0.0);
    // stdout, not stderr: gates capture both into one file, and an unbuffered
    // cerr write can land inside a buffered cout line.  See Trumpet::emit.
    if (!ok && rank == 0)
        std::cout << "# WARNING: Newton did not reach " << prec << " in " << iter
                  << " iterations\n" << std::flush;

    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }

    // -------------------------------------------------------- diagnostics ---
    // G-BC1: dr() at INNER_BC on an imported field with an exactly known
    // derivative.  NOTE this is dr(), never dn(): der_normal's sign at
    // INNER_BC was an open item in NOTES_kadath_api.md and dr() sidesteps it.
    const double dW_num = def_at_boundary(syst, m.space, "DWCHK", 0, INNER_BC);
    Trumpet::emit("BC1_dWdr_spectral", dW_num);
    Trumpet::emit("BC1_dWdr_rel", std::fabs(dW_num - bc.dWdr)
                                      / std::max(std::fabs(bc.dWdr), 1e-300));

    // G-BC2: did the imposed inner rows actually land?
    for (const auto& nm : inn) {
        const auto& r = bc.row.at(nm);
        const std::string dn = "CHK" + nm;
        const std::string lhs = bc_lhs(r, "k" + nm, syst);
        syst.add_def((dn + " = " + lhs).c_str());
        const double got = def_at_boundary(syst, m.space, dn.c_str(), 0, INNER_BC);
        const double want = manufactured ? r.rhs : j2 * r.rhs;
        Trumpet::emit("BC2_" + nm + "_got", got);
        Trumpet::emit("BC2_" + nm + "_rel",
                      std::fabs(got - want) / std::max(std::fabs(want), 1e-300));
    }

    // the alternates, evaluated (not imposed) on this solution: the swap test
    // compares these against their research-supplied budgets
    for (const auto& kv : bc.row) {
        const auto& nm = kv.first;
        const auto& r = kv.second;
        const std::string dn = "ALT" + nm;
        const std::string lhs = bc_lhs(r, "a" + nm, syst);
        syst.add_def((dn + " = " + lhs).c_str());
        const double got = def_at_boundary(syst, m.space, dn.c_str(), 0, INNER_BC);
        Trumpet::emit("ALT_" + nm, got);
        Trumpet::emit("ALT_" + nm + "_target", j2 * r.rhs);
        Trumpet::emit("ALT_" + nm + "_budget", std::fabs(j2) * r.budget);
    }

    // G-PIN: constraint residuals.
    //
    // NORMALISATION, AND A DEGENERACY THAT ONLY APPEARS IN A SOLVE.  The T2.1
    // LOG entry defines the relative residual as |E| / (largest single term).
    // That measure BREAKS at r = infinity: the r^{-p_n} row scaling leaves
    // exactly ONE nonzero coefficient there (E_K keeps only Upp = -128; every
    // other coefficient is exactly 0), so there is nothing for the residual to
    // cancel against and the ratio is structurally 1 whatever the solution is.
    // T2.1 never saw this because its reference profile made every term zero at
    // that point, so its gate SKIPPED it (scale == 0).
    // Three numbers are therefore reported per row:
    //   _abs    max |E|                          -- no normalisation, no degeneracy
    //   _rel    |E| / max_j |term_j|, but ONLY over points where at least two
    //           terms are nonzero: the LOG_code measure where it is meaningful
    //   _relcf  |E| / (max_j |c_j| * max_j |jet_j|) -- defined everywhere, since
    //           it separates the coefficient scale from the field scale
    for (int n = 0; n < 5; n++) {
        double wabs = 0.0, wrel = 0.0, wcf = 0.0;
        int wd = -1, wi = -1, nskip = 0;
        // PASS 0: the grid-wide term scale for this row.
        //
        // WHY.  Both relative measures are |E| divided by something built from
        // the terms at that point, and at r = infinity the r^{-p_n} scaling
        // leaves E_H and E_Mr with only (Qpp, Upp) -- in the exact (4,1) ratio
        // that is these rows' global second-derivative weighting.  A DECAYING
        // solution has U'' -> 0 and Q'' -> 0 there, so at that one point the
        // residual AND its normaliser are both at roundoff and the ratio is
        // 0/0.  Measured consequence: the EXACT mass mode, which annihilates
        // E_H and E_Mr identically, scored RESID_E_H_rel = 1.0000001 -- the
        // gate would have rejected a known-exact solution.  A point whose terms
        // are negligible against the row's own grid-wide scale carries no
        // relative information, so it is skipped and counted.
        double gridscale = 0.0;
        for (int d = 0; d <= dlast; d++) {
            std::vector<const Val_domain*> T0;
            for (const auto& t : termdef[n])
                T0.push_back(&syst.give_val_def_scalar_domain(t.c_str(), d));
            Index idx0(m.space.get_domain(d)->get_nbr_points());
            for (int i = 0; i < m.nbr(d); i++) {
                idx0.set(0) = i;
                double sc0 = 0.0;
                for (const auto* t : T0)
                    sc0 = std::max(sc0, std::fabs((*t)(idx0)));
                gridscale = std::max(gridscale, sc0);
            }
        }
        for (int d = 0; d <= dlast; d++) {
            const Val_domain& E =
                syst.give_val_def_scalar_domain(Trumpet::row_defs()[n], d);
            std::vector<const Val_domain*> T;
            std::vector<std::string> jets;
            for (const auto& t : termdef[n])
                T.push_back(&syst.give_val_def_scalar_domain(t.c_str(), d));
            for (const auto& jet : m.coefs().jets.at(Trumpet::rows()[n]))
                jets.push_back(jet);
            Index idx(m.space.get_domain(d)->get_nbr_points());
            double wdom = 0.0;
            for (int i = 0; i < m.nbr(d); i++) {
                idx.set(0) = i;
                double sc = 0.0, cmax = 0.0, jmax = 0.0;
                int nz = 0;
                for (std::size_t k = 0; k < T.size(); k++) {
                    const double t = (*T[k])(idx);
                    if (t != 0.0)
                        nz++;
                    sc = std::max(sc, std::fabs(t));
                    const double c = std::fabs(
                        m.coefs().v.at(std::make_pair(std::string(Trumpet::rows()[n]),
                                                      jets[k]))[d][i])
                                    / m.row_norm(n, d, i);
                    cmax = std::max(cmax, c);
                    if (c > 0.0)
                        jmax = std::max(jmax, std::fabs(t) / c);
                }
                const bool tiny = (sc < residfloor * gridscale);
                wabs = std::max(wabs, std::fabs(E(idx)));
                if (dumpresid && rank == 0)
                    std::cout << "residpt " << Trumpet::rows()[n] << " " << d
                              << " " << i << " " << std::setprecision(17)
                              << E(idx) << " " << sc << "\n";
                if (cmax * jmax > 0.0 && !tiny)
                    wcf = std::max(wcf, std::fabs(E(idx)) / (cmax * jmax));
                if (nz < 2 || sc <= 0.0 || sc < residfloor * gridscale) {
                    nskip++;
                    continue;
                }
                const double q = std::fabs(E(idx)) / sc;
                wdom = std::max(wdom, q);
                if (q > wrel) { wrel = q; wd = d; wi = i; }
            }
            std::cout << "# resid " << Trumpet::rows()[n] << " d" << d << " rel="
                      << wdom << "\n";
        }
        {   // skip fraction, registered per research round-3 Q5
            int tot = 0;
            for (int d = 0; d <= dlast; d++)
                tot += m.nbr(d);
            Trumpet::emit(std::string("RESID_") + Trumpet::rows()[n] + "_skipfrac",
                          tot > 0 ? double(nskip) / double(tot) : 0.0);
        }
        Trumpet::emit(std::string("RESID_") + Trumpet::rows()[n] + "_abs", wabs);
        Trumpet::emit(std::string("RESID_") + Trumpet::rows()[n] + "_rel", wrel);
        Trumpet::emit(std::string("RESID_") + Trumpet::rows()[n] + "_relcf", wcf);
        std::cout << "# resid " << Trumpet::rows()[n] << " worst-rel at d" << wd
                  << " ipt " << wi << "  (points skipped: " << nskip
                  << " of " << [&] { int t = 0; for (int d = 0; d <= dlast; d++) t += m.nbr(d); return t; }()
                  << "; single-term or below " << residfloor << " of the row's "
                     "grid-wide term scale " << gridscale << ")\n";
    }

    // ------------------------------------------------- Jacobian dump --------
    // The system is LINEAR, so the Jacobian is the operator itself and its
    // singular values are the whole story about how many digits a solve can
    // return.  do_col_J(i) is public and nbr_conditions is populated by the
    // solve above, so a full dump costs nbr_unknowns column evaluations.
    // T2.3 asks for Jacobian conditioning to be logged; this is that hook.
    // --- EXPORT LAYOUT.  Placed AFTER the solve on purpose:
    // classify_equation_row_metadata needs nbr_conditions, which is only
    // populated once sec_member() has run ("Number of conditions unknown ;
    // call sec_member first").  Before the solve it throws.
    // --- EXPORT LAYOUT (detail).  Which rows land in which export block, per domain, in
    // export order.  Round 3 asks for this whenever a placement misbehaves, and
    // it is cheap enough to print always: a Vol count that is not 3*(n_d - 2)
    // names the domain that gave up conditions, and the TauBc roster names
    // where they were spent.
    if (rank == 0) {
        std::vector<Kadath::System_of_eqs::RowMetadata> rmeta;
        syst.classify_equation_row_metadata(rmeta);
        auto taxname = [](Kadath::RowTaxonomy t) -> std::string {
            switch (t) {
                case Kadath::RowTaxonomy::Vol: return "Vol";
                case Kadath::RowTaxonomy::TauBc: return "TauBc";
                case Kadath::RowTaxonomy::TauMatch: return "TauMatch";
                case Kadath::RowTaxonomy::GlobalInt: return "GlobalInt";
                default: return "Unknown";
            }
        };
        std::map<std::string, std::map<int, int>> per;
        std::vector<std::string> taubc;
        for (const auto& rm : rmeta) {
            per[taxname(rm.taxonomy)][rm.dom]++;
            if (rm.taxonomy == Kadath::RowTaxonomy::TauBc) {
                std::ostringstream os;
                os << "d" << rm.dom << "/eq" << rm.eq_index;
                taubc.push_back(os.str());
            }
        }
        for (const auto& blk : per) {
            std::cout << "# export " << std::setw(9) << std::left << blk.first << " ";
            for (int d = 0; d <= dlast; d++) {
                const auto it = blk.second.find(d);
                std::cout << " d" << d << "=" << (it == blk.second.end() ? 0 : it->second);
                if (blk.first == "Vol")
                    std::cout << "(nat " << 3 * (m.nbr(d) - 2) << ")";
            }
            std::cout << "\n";
        }
        std::cout << "# export TauBc roster (export order):";
        for (const auto& t : taubc)
            std::cout << " " << t;
        std::cout << "\n";
    }


    if (!jacdump.empty()) {
        const int n = syst.get_nbr_unknowns();
        const int mrows = syst.get_nbr_conditions();
        std::vector<Kadath::System_of_eqs::RowMetadata> rmeta;
        syst.classify_equation_row_metadata(rmeta);
        std::ofstream fh(jacdump);
        fh << "# Jacobian of the T2.2 BVP  rows " << mrows << " cols " << n << "\n";
        fh << std::setprecision(17);
        Kadath::Array<double> rhs(syst.sec_member());
        for (int i = 0; i < mrows; i++)
            fh << "rhs " << i << " " << rhs(i) << "\n";
        for (int i = 0; i < n; i++) {
            Kadath::Array<double> col(syst.do_col_J(i));
            for (int r = 0; r < mrows; r++)
                if (col(r) != 0.0)
                    fh << "J " << r << " " << i << " " << col(r) << "\n";
        }
        std::cout << "# wrote " << jacdump << " (" << mrows << "x" << n << ")\n";
        std::ofstream rm(jacdump + ".rows"), cm(jacdump + ".cols");
        syst.dump_tagged_jacobian_metadata_csv(rm, cm);
    }

    // ------------------------------------- manufactured-solution comparison --
    if (manufactured) {
        Scalar Ue(m.space), Qe(m.space), Ge(m.space);
        {
            Trumpet::L0Model& mm = m;
            const double M = mm.table().M;
            for (int d = 0; d <= dlast; d++) {
                Ue.set_domain(d) = (1.0 - mm.Wf(d)) * 0.5;
                Qe.set_domain(d) = 0.0 * mm.Wf(d);
                Ge.set_domain(d) =
                    -(mm.iR(d) - (27.0 / 8.0) * std::pow(M, 3) * Kadath::pow(mm.iR(d), 4))
                        / mm.Wf(d)
                    - 2.0;
            }
            Ue.std_base(); Qe.std_base(); Ge.std_base();
        }
        for (int f = 0; f < 3; f++) {
            const char* nm = (f == 0) ? "U" : (f == 1) ? "Q" : "G";
            double num = 0.0, den = 0.0;
            int wd = -1, wi = -1;
            for (int d = 0; d <= dlast; d++) {
                const Val_domain& got = (f == 0) ? m.U(d) : (f == 1) ? m.Q(d) : m.G(d);
                const Val_domain& exa = (f == 0) ? Ue(d) : (f == 1) ? Qe(d) : Ge(d);
                Index idx(m.space.get_domain(d)->get_nbr_points());
                double wdom = 0.0;
                for (int i = 0; i < m.nbr(d); i++) {
                    idx.set(0) = i;
                    const double e = std::fabs(got(idx) - exa(idx));
                    wdom = std::max(wdom, e);
                    den = std::max(den, std::fabs(exa(idx)));
                    if (e > num) { num = e; wd = d; wi = i; }
                }
                std::cout << "# man " << nm << " d" << d << " max=" << wdom << "\n";
            }
            Trumpet::emit(std::string("MAN_") + nm + "_abs", num);
            // Q_exact is identically zero, so a relative error is meaningless
            // for it -- report absolute only and let the gate use that.
            if (den > 1e-30)
                Trumpet::emit(std::string("MAN_") + nm + "_rel",
                              num / den);
            std::cout << "# man " << nm << " worst at d" << wd << " ipt " << wi << "\n";
        }
    }

    // ------------------------------------------------------------- tails ----
    // On Domain_oned_inf, val_boundary(OUTER_BC, .) IS the value at r=infinity,
    // so mult_r then read gives the 1/r coefficient with no fit anywhere.
    {
        const Kadath::Domain* dom = m.space.get_domain(dlast);
        Index pcf(dom->get_nbr_coefs());
        const double tU = dom->val_boundary(OUTER_BC, dom->mult_r(m.U(dlast)), pcf);
        const double tQ = dom->val_boundary(OUTER_BC, dom->mult_r(m.Q(dlast)), pcf);
        const double tG = dom->val_boundary(OUTER_BC, dom->mult_r(m.G(dlast)), pcf);
        Trumpet::emit("TAIL_tU", tU);
        Trumpet::emit("TAIL_tQ", tQ);
        Trumpet::emit("TAIL_tG", tG);
        Trumpet::emit("TAIL_komar_2tU_plus_tG", 2.0 * tU + tG);
        // the unimposed outer value is a free null test
        for (const char* f : {"U", "Q", "G"}) {
            bool imposed = false;
            for (const auto& o : outset)
                if (o.first == f)
                    imposed = true;
            const Val_domain& v = (std::strcmp(f, "U") == 0)   ? m.U(dlast)
                                  : (std::strcmp(f, "Q") == 0) ? m.Q(dlast)
                                                               : m.G(dlast);
            Trumpet::emit(std::string("OUTVAL_") + f, dom->val_boundary(OUTER_BC, v, pcf));
            Trumpet::emit(std::string("OUTIMPOSED_") + f, imposed ? 1.0 : 0.0);
        }
        std::cout << "#\n"
                  << "# ================= NOT A VERDICT =================\n"
                  << "# (tU,tQ,tG) = (" << tU << ", " << tQ << ", " << tG << ")\n"
                  << "# 2tU = " << 2.0 * tU << ",  2tU+tG = " << 2.0 * tU + tG << "\n"
                  << "# The T2.3 gate battery (throat oracle, tails gate, FD\n"
                  << "# cross-check, W0 sweep) has NOT been run.  These numbers\n"
                  << "# are assembly diagnostics, not the Phase-3 verdict.\n"
                  << "# ================================================\n";
    }

    // ------------------------------------------------------------ profile ---
    if (!profile.empty()) {
        std::ofstream fh(profile);
        if (!fh) {
            std::cerr << "FATAL: cannot write " << profile << "\n";
            MPI_Finalize();
            return 1;
        }
        fh << "# T2.2 solved l=0 profile -- l0_solve\n"
           << "# tag " << m.table().tag << "  j2 " << j2 << "  inner " << inner
           << "  outer " << outer << "\n"
           << "# sol <dom> <ipt> <r> <U> <Q> <G> <dU/dr> <dQ/dr> <dG/dr>\n"
           << "# cf  <dom> <field> <i> <coef>\n";
        syst.add_def("PdU = dr(U)");
        syst.add_def("PdQ = dr(Q)");
        syst.add_def("PdG = dr(G)");
        fh << std::setprecision(17);
        for (int d = 0; d <= dlast; d++) {
            const Val_domain& dU = syst.give_val_def_scalar_domain("PdU", d);
            const Val_domain& dQ = syst.give_val_def_scalar_domain("PdQ", d);
            const Val_domain& dG = syst.give_val_def_scalar_domain("PdG", d);
            Index idx(m.space.get_domain(d)->get_nbr_points());
            for (int i = 0; i < m.nbr(d); i++) {
                idx.set(0) = i;
                fh << "sol " << d << " " << i << " " << m.table().pts[d][i].r << " "
                   << m.U(d)(idx) << " " << m.Q(d)(idx) << " " << m.G(d)(idx) << " "
                   << dU(idx) << " " << dQ(idx) << " " << dG(idx) << "\n";
            }
            // spectral coefficients: T2.3's throat oracle reads these
            Index icf(m.space.get_domain(d)->get_nbr_coefs());
            for (const char* f : {"U", "Q", "G"}) {
                const Val_domain& v = (std::strcmp(f, "U") == 0)   ? m.U(d)
                                      : (std::strcmp(f, "Q") == 0) ? m.Q(d)
                                                                   : m.G(d);
                v.coef();
                for (int i = 0; i < m.space.get_domain(d)->get_nbr_coefs()(0); i++) {
                    icf.set(0) = i;
                    fh << "cf " << d << " " << f << " " << i << " " << v.get_coef(icf)
                       << "\n";
                }
            }
        }
        std::cout << "# wrote " << profile << "\n";
    }

    std::cout << "# done\n";
    MPI_Finalize();
    return 0;
}

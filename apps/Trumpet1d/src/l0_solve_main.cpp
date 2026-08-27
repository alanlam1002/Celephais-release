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

/** Boundary value of a named add_def, read through the Chebyshev coefficients. */
double def_at_boundary(System_of_eqs& syst, const Kadath::Space& space,
                       const char* name, int dom, int bound)
{
    Index pcf(space.get_domain(dom)->get_nbr_coefs());
    return space.get_domain(dom)->val_boundary(
        bound, syst.give_val_def_scalar_domain(name, dom), pcf);
}

} // namespace

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
    bool rownorm = false, manufactured = false;
    std::string jacdump, pinat = "inner";
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
        else if (k == "--profile") profile = next();
        else if (k == "--rownorm") rownorm = true;
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
            bc.row["mU"] = TrumpetIO::BcRow{"der1", "U", -0.5 * bc.dWdr, 0.0};
            bc.row["mQ"] = TrumpetIO::BcRow{"der1", "Q", 0.0, 0.0};
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
    if (inn.size() + out.size() + pin.size() != 6) {
        if (rank == 0)
            std::cerr << "FATAL: --inner (" << inn.size() << ") + --pins ("
                      << pin.size() << ") + --outer (" << out.size()
                      << ") must total exactly 6\n";
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

    // --- bulk: the three evolution rows, in every domain
    for (int d = 0; d <= dlast; d++) {
        syst.add_eq_inside(d, "EK = 0");
        syst.add_eq_inside(d, "EXR = 0");
        syst.add_eq_inside(d, "EXT = 0");
    }
    // --- C0 + C1 matching of all three fields at every interface
    for (int d = 0; d < dlast; d++)
        for (const char* f : {"U", "Q", "G"}) {
            syst.add_eq_matching(d, OUTER_BC, f);
            syst.add_eq_matching(d, OUTER_BC, (std::string("dr(") + f + ")").c_str());
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
        const std::string lhs =
            (it->second.kind == "der1")
                ? "dr(" + it->second.field + ")"
                : (it->second.field == "4U+Q+G" ? "4 * U + Q + G" : it->second.field);
        inner_eq.push_back(lhs + " = " + cnm);
        syst.add_eq_bc(0, INNER_BC, inner_eq.back().c_str());
    }
    // --- constraint pins, same boundary
    for (const auto& p : pin) {
        if (p != "EH" && p != "EM") {
            if (rank == 0)
                std::cerr << "FATAL: --pins entries must be EH or EM, got " << p << "\n";
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

    if (rank == 0) {
        for (const auto& e : inner_eq)
            std::cout << "# inner  add_eq_bc(0, INNER_BC, \"" << e << "\")\n";
        for (const auto& p : pin)
            std::cout << "# pin    add_eq_bc(0, INNER_BC, \"" << p << " = 0\")\n";
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
    try {
        while (iter < 12) {
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
    if (!ok && rank == 0)
        std::cerr << "WARNING: Newton did not reach " << prec << " in " << iter
                  << " iterations\n";

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
        const std::string lhs =
            (r.kind == "der1") ? "dr(" + r.field + ")"
                               : (r.field == "4U+Q+G" ? "4 * U + Q + G" : r.field);
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
        const std::string lhs =
            (r.kind == "der1") ? "dr(" + r.field + ")"
                               : (r.field == "4U+Q+G" ? "4 * U + Q + G" : r.field);
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
                wabs = std::max(wabs, std::fabs(E(idx)));
                if (cmax * jmax > 0.0)
                    wcf = std::max(wcf, std::fabs(E(idx)) / (cmax * jmax));
                if (nz < 2 || sc <= 0.0) {
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
        Trumpet::emit(std::string("RESID_") + Trumpet::rows()[n] + "_abs", wabs);
        Trumpet::emit(std::string("RESID_") + Trumpet::rows()[n] + "_rel", wrel);
        Trumpet::emit(std::string("RESID_") + Trumpet::rows()[n] + "_relcf", wcf);
        std::cout << "# resid " << Trumpet::rows()[n] << " worst-rel at d" << wd
                  << " ipt " << wi << "  (single-term points skipped: " << nskip
                  << ")\n";
    }

    // ------------------------------------------------- Jacobian dump --------
    // The system is LINEAR, so the Jacobian is the operator itself and its
    // singular values are the whole story about how many digits a solve can
    // return.  do_col_J(i) is public and nbr_conditions is populated by the
    // solve above, so a full dump costs nbr_unknowns column evaluations.
    // T2.3 asks for Jacobian conditioning to be logged; this is that hook.
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

/*
 * l0_solve2d_main.cpp -- A0c: the O(j^2) solver, ported onto the 2-D polar
 * layout.
 *
 * Research round 278 ruled the port faithful in its DIAGNOSTICS and deferred
 * the refuted remedies (--add-equil modes, --pin-farmodes, --horizon-fix,
 * --cond-deficit, --log-enrich) until P2 recurs, to be RE-TESTED rather than
 * re-applied if it does.  Round 285 ruled it onto the LINEAR 2-D layout first,
 * because Space_oned_trumpet has no log shell and so the 1-D reference numbers
 * only exist there: porting onto log first would leave the port with no
 * reference at all.
 *
 * THE KNOWN ANSWER THIS RUNG IS CHECKED AGAINST, and it is sharper than "it
 * assembles".  The O(j^2) operator is theta-INDEPENDENT: every coefficient
 * field is a function of r alone, and the source j2 is too.  So on a polar grid
 * the system is BLOCK DIAGONAL in the angular mode, with every block equal to
 * the 1-D system, and only the k = 0 block is driven.  Therefore
 *
 *     ADD_res_core, sv_min, sv_max and the manufactured oracle must come back
 *     EXACTLY at their 1-D values, at every ntheta,
 *
 * and P2's angular attribution must sit entirely at mode 0.  Anything else is
 * the port, not the physics -- which is the whole point of doing A0c against a
 * system whose answer is known.
 *
 * The row assembly is SHARED with Trumpet1d through l0_setup.hpp (L0ModelT,
 * bc_lhs), so the T2.1 gate still certifies the operator this app solves.  Only
 * the space and the fill are dimension-dependent.
 *
 * Usage: l0_solve2d <backbone> <coefs> <bc> [--ntheta N] [--j2 X] [--rownorm]
 *                   [--inner a,b] [--pins a] [--outer a,b,c] [--pin-at inner|outer]
 *                   [--add-rows compat] [--manufactured]
 */

#include <mpi.h>

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include "space/space_polar_trumpet.hpp"
#include "Trumpet1d/src/l0_setup.hpp"
#include "Trumpet1d/src/table_io.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <sstream>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" void dgelsd_(int*, int*, int*, double*, int*, double*, int*, double*,
                        double*, int*, double*, int*, int*, int*);

using Kadath::Index;
using Kadath::System_of_eqs;
using L0Model2d = Trumpet::L0ModelT<Trumpet::Space_polar_trumpet, 2>;

namespace
{

std::vector<std::string> split2(const std::string& s)
{
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (!cur.empty())
                out.push_back(cur);
            cur.clear();
        } else
            cur += c;
    }
    if (!cur.empty())
        out.push_back(cur);
    return out;
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (argc < 4) {
        if (rank == 0)
            std::cerr << "usage: l0_solve2d <backbone> <coefs> <bc> [options]\n";
        MPI_Finalize();
        return 2;
    }

    int ntheta = 5;
    double j2 = 1.0;
    bool rownorm = false, manufactured = false;
    std::string inner = "killer,MP", pins = "EH", outer = "U,Q,G";
    std::string pinat = "outer", addrows = "compat";
    for (int i = 4; i < argc; i++) {
        const std::string k = argv[i];
        auto nxt = [&]() { return std::string(argv[++i]); };
        if (k == "--ntheta") ntheta = std::stoi(nxt());
        else if (k == "--j2") j2 = std::stod(nxt());
        else if (k == "--rownorm") rownorm = true;
        else if (k == "--inner") inner = nxt();
        else if (k == "--pins") pins = nxt();
        else if (k == "--outer") outer = nxt();
        else if (k == "--pin-at") pinat = nxt();
        else if (k == "--add-rows") addrows = nxt();
        else if (k == "--manufactured") {
            manufactured = true;
            j2 = 0.0;
            inner = "mU,mQ";
            outer = "U=0,Q=0,G=-2";
            pins = "EH";
        } else {
            if (rank == 0)
                std::cerr << "FATAL: unknown option " << k << "\n";
            MPI_Finalize();
            return 2;
        }
    }

    // ⚠ ROUND 97.  At ntheta <= 3 Kadath's angular transform is the IDENTITY:
    // coef_1d_cos_even and its inverse both branch on `if (nbr > 3)` and
    // otherwise call copy_untransformed_line, so every coefficient-space
    // operation -- val_boundary at a given angular index, export_tau,
    // affecte_tau -- silently acts on grid values instead.  Measured: a
    // CONSTANT field returns angular coefficients (1, 1, 1) instead of
    // (1, 0, 0), exactly, on shells and on the compact domain alike; at
    // ntheta >= 5 the same probe returns 0.000e+00 error.  Both existing guards
    // pass it -- the line length is odd, and the r2hc size check is never
    // reached because the branch is taken first -- so it has to be refused
    // here.
    if (ntheta < 5) {
        if (rank == 0)
            std::cerr << "FATAL: ntheta must be >= 5.  At ntheta <= 3 the "
                         "angular transform is the identity (coef_1d.cpp, "
                         "`if (nbr > 3)`), so coefficients are grid values and "
                         "every tau and val_boundary read is silently wrong.\n";
        MPI_Finalize();
        return 2;
    }

    std::unique_ptr<L0Model2d> mp;
    TrumpetIO::BcTable bc;
    try {
        mp = std::make_unique<L0Model2d>(argv[1], argv[2], rownorm, ntheta);
        bc = TrumpetIO::read_bc(argv[3]);
        if (manufactured) {
            // the manufactured BVP's inner data comes from the backbone itself
            const double dW = bc.dWdr;
            TrumpetIO::BcRow mU, mQ;
            mU.kind = "der1"; mU.field = "U"; mU.rhs = -0.5 * dW;
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
    L0Model2d& m = *mp;
    const int dlast = m.nb_domains() - 1;

    if (rank == 0) {
        std::cout << "# l0_solve2d  ntheta=" << ntheta << "  j2=" << j2
                  << "  inner=" << inner << "  outer=" << outer
                  << "  pins=" << pins << "\n";
    }
    Trumpet::emit("A0C_ntheta", ntheta);
    Trumpet::emit("A0C_ndom", m.nb_domains());

    m.set_fields_zero();
    System_of_eqs syst(m.sp(), 0, dlast);
    const auto defs = m.register_rows(syst, j2);
    syst.add_cst("oorbb", m.oorf);

    // Per-TERM add_defs, so each term of each row can be read back separately:
    // the relative residual measures are |E| divided by something built from
    // the terms at that point, and that needs the terms, not just the row.
    std::vector<std::vector<std::string>> termdef(5);
    for (int n = 0; n < 5; n++) {
        int t = 0;
        for (const auto& jet : m.coefs().jets.at(Trumpet::rows()[n])) {
            std::ostringstream nm;
            nm << "T" << Trumpet::row_codes()[n] << t++;
            const std::string d =
                nm.str() + " = c" + Trumpet::row_codes()[n] + jet + " * "
                + Trumpet::apply_jet(jet, Trumpet::field_of(jet, m.coefs().fields));
            syst.add_def(d.c_str());
            termdef[n].push_back(nm.str());
        }
    }

    auto P = [&](const char* f, int o) {
        return Trumpet::phys_expr(m.coefs().fields, f, o);
    };

    // --- bulk: the three evolution rows, every domain, natural tau order -----
    for (int d = 0; d <= dlast; d++) {
        syst.add_eq_inside(d, "EK = 0");
        syst.add_eq_inside(d, "EXR = 0");
        syst.add_eq_inside(d, "EXT = 0");
    }
    // --- C0 + C1 matching of all three fields at every interface ------------
    for (int d = 0; d < dlast; d++)
        for (const std::string& f : m.coefs().fields) {
            syst.add_eq_matching(d, OUTER_BC, f.c_str());
            syst.add_eq_matching(d, OUTER_BC, ("dr(" + f + ")").c_str());
        }

    // --- the six R1 conditions ----------------------------------------------
    const auto inn = split2(inner), pin = split2(pins), out = split2(outer);
    if (inn.size() + pin.size() + out.size() != 6) {
        if (rank == 0)
            std::cerr << "FATAL: --inner + --pins + --outer must total 6, got "
                      << inn.size() + pin.size() + out.size() << "\n";
        MPI_Finalize();
        return 2;
    }
    std::vector<std::string> keep;
    for (const auto& nm : inn) {
        auto it = bc.row.find(nm);
        if (it == bc.row.end()) {
            if (rank == 0)
                std::cerr << "FATAL: no bc row '" << nm << "'\n";
            MPI_Finalize();
            return 2;
        }
        const std::string cnm = "bc" + nm;
        syst.add_cst(cnm.c_str(), manufactured ? it->second.rhs
                                               : j2 * it->second.rhs);
        keep.push_back(Trumpet::bc_lhs(it->second, "r" + nm, syst,
                                       m.coefs().fields) + " = " + cnm);
        syst.add_eq_bc(0, INNER_BC, keep.back().c_str());
    }
    for (const auto& p : pin) {
        if (p != "EH" && p != "EM") {
            if (rank == 0)
                std::cerr << "FATAL: --pins entries must be EH or EM\n";
            MPI_Finalize();
            return 2;
        }
        if (pinat == "inner")
            syst.add_eq_bc(0, INNER_BC, (p + " = 0").c_str());
        else
            syst.add_eq_bc(dlast, OUTER_BC, (p + " = 0").c_str());
    }
    for (const auto& f : out) {
        const auto eq = f.find('=');
        const std::string fld = (eq == std::string::npos) ? f : f.substr(0, eq);
        const double tgt = (eq == std::string::npos) ? 0.0
                                                     : std::stod(f.substr(eq + 1));
        const std::string cnm = "out" + fld;
        syst.add_cst(cnm.c_str(), tgt);
        keep.push_back(P(fld.c_str(), 0) + " = " + cnm);
        syst.add_eq_bc(dlast, OUTER_BC, keep.back().c_str());
    }

    // --- the two APPENDED compat rows at r(2M) ------------------------------
    // Located, not hardcoded: the interface where E_K's only second-derivative
    // coefficient vanishes against its own grid maximum.  kappa is MEASURED one
    // point inside (it is 0/0 at the interface itself) and asserted, so a table
    // without this structure fails loudly rather than imposing a wrong row.
    int n_appended = 0;
    int dh = -1;
    if (addrows == "compat") {
        double best = 1e300;
        for (int d = 0; d < dlast; d++) {
            const int i = m.nbr(d) - 1;
            double mx = 0.0;
            for (int k = 0; k < m.nbr(d); k++)
                mx = std::max(mx, std::fabs(m.phys_coef("E_K", "Upp", d, k)));
            const double rel = std::fabs(m.phys_coef("E_K", "Upp", d, i))
                               / std::max(mx, 1e-300);
            if (rel < best) { best = rel; dh = d; }
        }
        if (dh < 0 || best > 1e-12) {
            if (rank == 0)
                std::cerr << "FATAL: --add-rows compat: no interface where E_K's "
                             "Upp vanishes (best relative " << best << ")\n";
            MPI_Finalize();
            return 2;
        }
        const int ih = m.nbr(dh) - 2;           // one point inside
        const double kap = m.phys_coef("E_chi_tt", "Upp", dh, ih)
                           / m.phys_coef("E_K", "Upp", dh, ih);
        if (std::fabs(kap - 2.0) > 1e-9) {
            if (rank == 0)
                std::cerr << "FATAL: E_chi_tt.Upp / E_K.Upp = " << kap
                          << ", expected 2\n";
            MPI_Finalize();
            return 2;
        }
        Trumpet::emit("A0C_horizon_domain", dh);
        Trumpet::emit("A0C_kappa", kap);
        syst.add_cst("hkap", kap);
        syst.add_def("ECOMPAT = EXT - hkap * EK");
        syst.add_eq_bc(dh, OUTER_BC, "EK = 0");
        syst.add_eq_bc(dh, OUTER_BC, "ECOMPAT = 0");
        n_appended += 2;
    }

    // --- extract the rectangular system, exactly as the 1-D app does --------
    // The problem is LINEAR, so one step from a zero guess is exact and the
    // Jacobian is the operator.  sec_member() and do_col_J() carry no
    // squareness assertion, so the stack is taken FROM Kadath rather than
    // hand-assembled: the core rows are literally the baseline system's rows,
    // in the same order.
    m.set_fields_zero();
    Kadath::Array<double> bb(syst.sec_member());
    const int nrow = syst.get_nbr_conditions();
    const int ncol = syst.get_nbr_unknowns();
    const int ncore = nrow - n_appended * ntheta;
    Trumpet::emit("ADD_rows", nrow);
    Trumpet::emit("ADD_cols", ncol);
    Trumpet::emit("ADD_core_rows", ncore);
    Trumpet::emit("A0C_appended_per_mode", n_appended);

    std::vector<double> A(std::size_t(nrow) * ncol, 0.0), rhs(nrow, 0.0);
    for (int r = 0; r < nrow; r++)
        rhs[r] = bb(r);
    for (int c = 0; c < ncol; c++) {
        Kadath::Array<double> col(syst.do_col_J(c));
        for (int r = 0; r < nrow; r++)
            A[std::size_t(c) * nrow + r] = col(r);       // column-major, LAPACK
    }

    // equilibration: appended rows to the median core-row norm (the production
    // "appended" mode; the other two are refuted remedies and are not ported)
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
    for (int r = ncore; r < nrow; r++)
        if (rnorm[r] > 0.0)
            rscale[r] = medcore / rnorm[r];

    std::vector<double> Aw(std::size_t(nrow) * ncol, 0.0);
    for (int r = 0; r < nrow; r++)
        for (int c = 0; c < ncol; c++)
            Aw[std::size_t(c) * nrow + r] = A[std::size_t(c) * nrow + r] * rscale[r];

    int mm = nrow, nn = ncol, nrhs = 1, lda = nrow, ldb_ = std::max(nrow, ncol),
        rank_out = 0, info = 0;
    std::vector<double> sv(std::min(nrow, ncol), 0.0), btmp(ldb_, 0.0),
        Atmp(Aw.size(), 0.0);
    double rcond = -1.0;
    int lwork = -1;
    std::vector<int> iwork(std::size_t(64) * std::min(nrow, ncol) + 4096);
    {
        double wq = 0.0;
        Atmp = Aw;
        dgelsd_(&mm, &nn, &nrhs, Atmp.data(), &lda, btmp.data(), &ldb_, sv.data(),
                &rcond, &rank_out, &wq, &lwork, iwork.data(), &info);
        lwork = static_cast<int>(wq);
    }
    std::vector<double> work(std::max(lwork, 1));

    // ITERATIVE REFINEMENT, as the 1-D app does: up to 8 passes, stopping when
    // the core residual stalls.  Note the RHS is NOT negated -- xx_to_vars_delta
    // carries that convention, and negating it here (the first draft) leaves the
    // residual at twice its starting value, which reads as a broken port.
    double post_core = 0.0, post_add = 0.0;
    int refine = 0;
    for (refine = 0; refine < 8; refine++) {
        Kadath::Array<double> r0(refine == 0 ? bb : syst.sec_member());
        for (int r = 0; r < nrow; r++)
            btmp[r] = r0(r) * rscale[r];
        for (int r = nrow; r < ldb_; r++)
            btmp[r] = 0.0;
        Atmp = Aw;
        dgelsd_(&mm, &nn, &nrhs, Atmp.data(), &lda, btmp.data(), &ldb_, sv.data(),
                &rcond, &rank_out, work.data(), &lwork, iwork.data(), &info);
        if (info != 0) {
            if (rank == 0)
                std::cerr << "FATAL: dgelsd returned info = " << info << "\n";
            MPI_Finalize();
            return 1;
        }
        Kadath::Array<double> xx(ncol);
        for (int c = 0; c < ncol; c++)
            xx.set(c) = btmp[c];
        int conte = 0;
        syst.xx_to_vars_delta(xx, conte);
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
    // ---- P2's ANGULAR ATTRIBUTION (research round 285 ruling 2) ------------
    //
    // Attribute by angular MODE, not by collocation point: a null direction
    // spread over points is uninterpretable, one concentrated in a mode is not,
    // and round 90 established that aliasing is the mechanism AT THE TOP MODE --
    // so a direction sitting there is an artefact, distinguishable from a
    // physical low-l one.  That distinction exists only in coefficient space.
    //
    // The blocks are FOUND, not assumed.  A union-find over the Jacobian's
    // sparsity pattern splits rows and columns into connected components; if
    // the operator really is theta-independent there are exactly ntheta of
    // them, each the size of the 1-D system.  So the instrument verifies the
    // block-diagonal claim rather than resting on it.
    {
        // the residual AT THE SOLUTION, which the refinement loop left in the
        // fields
        Kadath::Array<double> res(syst.sec_member());
        const int nn_tot = nrow + ncol;
        std::vector<int> par(nn_tot);
        for (int i = 0; i < nn_tot; i++)
            par[i] = i;
        std::function<int(int)> find = [&](int a) {
            while (par[a] != a) { par[a] = par[par[a]]; a = par[a]; }
            return a;
        };
        auto uni = [&](int a, int b) {
            a = find(a); b = find(b);
            if (a != b) par[a] = b;
        };
        double amax = 0.0;
        for (std::size_t i = 0; i < A.size(); i++)
            amax = std::max(amax, std::fabs(A[i]));
        const double tol = 1e-14 * amax;
        for (int c = 0; c < ncol; c++)
            for (int r = 0; r < nrow; r++)
                if (std::fabs(A[std::size_t(c) * nrow + r]) > tol)
                    uni(r, nrow + c);
        std::map<int, int> blockrows, blockcols;
        std::map<int, double> blockres, blockrhs;
        for (int r = 0; r < nrow; r++) {
            const int b = find(r);
            blockrows[b]++;
            blockres[b] = std::max(blockres[b], std::fabs(res(r)));
            blockrhs[b] = std::max(blockrhs[b], std::fabs(bb(r)));
        }
        for (int c = 0; c < ncol; c++)
            blockcols[find(nrow + c)]++;
        Trumpet::emit("P2ANG_blocks", static_cast<double>(blockrows.size()));
        int idx = 0, driven = 0;
        double worstres = 0.0;
        int worstblk = -1;
        for (const auto& kv : blockrows) {
            const int b = kv.first;
            if (blockrhs[b] > 0.0)
                driven++;
            if (blockres[b] > worstres) { worstres = blockres[b]; worstblk = idx; }
            Trumpet::emit("P2ANG_rows_b" + std::to_string(idx), kv.second);
            Trumpet::emit("P2ANG_cols_b" + std::to_string(idx), blockcols[b]);
            Trumpet::emit("P2ANG_res_b" + std::to_string(idx), blockres[b]);
            Trumpet::emit("P2ANG_rhs_b" + std::to_string(idx), blockrhs[b]);
            idx++;
        }
        Trumpet::emit("P2ANG_driven_blocks", driven);
        Trumpet::emit("P2ANG_worst_block", worstblk);
        // The driven mode: the source j2 and the backbone are theta-independent,
        // so only the COS_EVEN k = 0 mode has a nonzero right-hand side.  If
        // exactly one block is driven and it is the one carrying the residual,
        // the weight sits at l = 0 and NOT at the top mode -- which is the
        // signature round 90 taught us to look for.
        Trumpet::emit("P2ANG_at_top_mode",
                      (worstblk == static_cast<int>(blockrows.size()) - 1
                       && blockrows.size() > 1) ? 1 : 0);
    }

    Trumpet::emit("ADD_rank", rank_out);
    Trumpet::emit("ADD_sv_max", sv.front());
    Trumpet::emit("ADD_sv_min", sv[std::min(nrow, ncol) - 1]);
    Trumpet::emit("ADD_refines", refine + 1);
    Trumpet::emit("ADD_res_core", post_core);
    Trumpet::emit("ADD_res_appended", post_add);

    // ------------------------------------------- RESID_*, per row ----------
    // Three measures, for the reason the 1-D app records: at r = infinity the
    // r^{-p_n} scaling leaves E_H and E_Mr with only (Qpp, Upp), and a DECAYING
    // solution has both -> 0 there, so the residual and its normaliser are both
    // at roundoff and the ratio is 0/0.  The exact mass mode scored
    // RESID_E_H_rel = 1.0000001 that way -- the gate would have rejected a
    // known-exact solution.
    //   _abs    max |E|
    //   _rel    |E| / max_j |term_j|, only where at least two terms are nonzero
    //   _relcf  |E| / (max_j |c_j| * max_j |jet_j|), defined everywhere
    {
        const double residfloor = 1e-12;
        for (int n = 0; n < 5; n++) {
            double wabs = 0.0, wrel = 0.0, wcf = 0.0, gridscale = 0.0;
            int nskip = 0, npt = 0;
            for (int d = 0; d <= dlast; d++) {
                std::vector<const Kadath::Val_domain*> T0;
                for (const auto& t : termdef[n])
                    T0.push_back(&syst.give_val_def_scalar_domain(t.c_str(), d));
                Index i0(m.sp().get_domain(d)->get_nbr_points());
                do {
                    double sc0 = 0.0;
                    for (const auto* t : T0)
                        sc0 = std::max(sc0, std::fabs((*t)(i0)));
                    gridscale = std::max(gridscale, sc0);
                } while (i0.inc());
            }
            for (int d = 0; d <= dlast; d++) {
                const Kadath::Val_domain& E =
                    syst.give_val_def_scalar_domain(Trumpet::row_defs()[n], d);
                std::vector<const Kadath::Val_domain*> T;
                std::vector<std::string> jets;
                for (const auto& t : termdef[n])
                    T.push_back(&syst.give_val_def_scalar_domain(t.c_str(), d));
                for (const auto& jet : m.coefs().jets.at(Trumpet::rows()[n]))
                    jets.push_back(jet);
                Index idx(m.sp().get_domain(d)->get_nbr_points());
                do {
                    const int i = idx(0);          // the coefficient tables are
                    double sc = 0.0, cmax = 0.0, jmax = 0.0;   // radial only
                    int nz = 0;
                    for (std::size_t k = 0; k < T.size(); k++) {
                        const double t = (*T[k])(idx);
                        if (t != 0.0)
                            nz++;
                        sc = std::max(sc, std::fabs(t));
                        const double c =
                            std::fabs(m.coefs().v.at(std::make_pair(
                                std::string(Trumpet::rows()[n]), jets[k]))[d][i])
                            / m.row_norm(n, d, i);
                        cmax = std::max(cmax, c);
                        if (c > 0.0)
                            jmax = std::max(jmax, std::fabs(t) / c);
                    }
                    npt++;
                    const bool tiny = (sc < residfloor * gridscale);
                    wabs = std::max(wabs, std::fabs(E(idx)));
                    if (cmax * jmax > 0.0 && !tiny)
                        wcf = std::max(wcf, std::fabs(E(idx)) / (cmax * jmax));
                    if (nz < 2 || sc <= 0.0 || tiny) {
                        nskip++;
                        continue;
                    }
                    wrel = std::max(wrel, std::fabs(E(idx)) / sc);
                } while (idx.inc());
            }
            Trumpet::emit(std::string("RESID_") + Trumpet::rows()[n] + "_abs", wabs);
            Trumpet::emit(std::string("RESID_") + Trumpet::rows()[n] + "_rel", wrel);
            Trumpet::emit(std::string("RESID_") + Trumpet::rows()[n] + "_relcf", wcf);
            Trumpet::emit(std::string("RESID_") + Trumpet::rows()[n] + "_skipfrac",
                          npt > 0 ? double(nskip) / npt : 0.0);
        }
    }

    // ------------------------------------------------------------- tails ----
    // ⚠ THE COMPACT DOMAIN is the one piece of the 2-D layout with no
    // inherited-assumption check of its own -- A0a exercised the shells, A0b the
    // angular basis, A0e the shells again -- and the base rate on inherited-
    // and-unexercised pieces is four for four wrong.  So this block does not
    // just read the tails: it reads EVERY angular mode's tail.  The solution is
    // driven only at mode 0, so modes 1.. must be zero, and a compact-domain
    // bug that mixed modes would show there rather than in a number that has no
    // reference.
    {
        const Kadath::Domain* dom = m.sp().get_domain(dlast);
        Index pcf(dom->get_nbr_coefs());
        const double tU = dom->val_boundary(OUTER_BC,
                                            dom->mult_r(m.Uphys(dlast)), pcf);
        const double tQ = dom->val_boundary(OUTER_BC,
                                            dom->mult_r(m.Qphys(dlast)), pcf);
        const double tG = dom->val_boundary(OUTER_BC,
                                            dom->mult_r(m.Gphys(dlast)), pcf);
        Trumpet::emit("TAIL_tU", tU);
        Trumpet::emit("TAIL_tQ", tQ);
        Trumpet::emit("TAIL_tG", tG);
        Trumpet::emit("TAIL_komar_2tU_plus_tG", 2.0 * tU + tG);
        Trumpet::emit("TAIL_tG_over_tU", tU != 0.0 ? tG / tU : 0.0);

        double offmode = 0.0;
        const int nk = dom->get_nbr_coefs()(1);
        for (int k = 1; k < nk; k++) {
            Index pk(dom->get_nbr_coefs());
            pk.set(1) = k;
            for (int f = 0; f < 3; f++) {
                const Kadath::Val_domain v = (f == 0) ? m.Uphys(dlast)
                                           : (f == 1) ? m.Qphys(dlast)
                                                      : m.Gphys(dlast);
                const double tk = std::fabs(dom->val_boundary(OUTER_BC,
                                                             dom->mult_r(v), pk));
                offmode = std::max(offmode, tk);
                Trumpet::emit("TAILK_" + std::string(f == 0 ? "U" : f == 1 ? "Q" : "G")
                                  + "_k" + std::to_string(k), tk);
            }
        }
        Trumpet::emit("TAIL_offmode_max", offmode);
        Trumpet::emit("TAIL_angular_modes", nk);

        for (const char* f : {"U", "Q", "G"}) {
            const Kadath::Val_domain v = (std::string(f) == "U") ? m.Uphys(dlast)
                                       : (std::string(f) == "Q") ? m.Qphys(dlast)
                                                                 : m.Gphys(dlast);
            Trumpet::emit(std::string("OUTVAL_") + f,
                          dom->val_boundary(OUTER_BC, v, pcf));
        }
    }

    // ------------------------------ the manufactured-solution comparison ----
    // The P1 oracle: the exact mass mode U = (1-W)/2, Q = 0, G = F_M - 2
    // annihilates all five rows at j2 = 0, so recovering it is a full-BVP test
    // with a known answer.  It is the instrument that killed four of five
    // candidate changes in 1-D, and it is the point at which "the port is
    // right" becomes checkable rather than argued.
    if (manufactured) {
        Kadath::Scalar Ue(m.sp()), Qe(m.sp()), Ge(m.sp());
        const double M = m.table().M;
        for (int d = 0; d <= dlast; d++) {
            Ue.set_domain(d) = (1.0 - m.Wf(d)) * 0.5;
            Qe.set_domain(d) = 0.0 * m.Wf(d);
            Ge.set_domain(d) =
                -(m.iR(d) - (27.0 / 8.0) * std::pow(M, 3) * Kadath::pow(m.iR(d), 4))
                    / m.Wf(d) - 2.0;
        }
        Ue.std_base();
        Qe.std_base();
        Ge.std_base();
        for (int f = 0; f < 3; f++) {
            const char* nm = (f == 0) ? "U" : (f == 1) ? "Q" : "G";
            double num = 0.0, den = 0.0;
            for (int d = 0; d <= dlast; d++) {
                const Kadath::Val_domain got = (f == 0) ? m.Uphys(d)
                                             : (f == 1) ? m.Qphys(d) : m.Gphys(d);
                const Kadath::Val_domain& exa = (f == 0) ? Ue(d)
                                              : (f == 1) ? Qe(d) : Ge(d);
                Index idx(m.sp().get_domain(d)->get_nbr_points());
                do {
                    num = std::max(num, std::fabs(got(idx) - exa(idx)));
                    den = std::max(den, std::fabs(exa(idx)));
                } while (idx.inc());
            }
            Trumpet::emit(std::string("MAN_") + nm + "_abs", num);
            if (den > 1e-30)
                Trumpet::emit(std::string("MAN_") + nm + "_rel", num / den);
        }
    }

    std::cout << "# done\n";
    MPI_Finalize();
    return 0;
}

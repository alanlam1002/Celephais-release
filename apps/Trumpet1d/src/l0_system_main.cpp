/*
 * l0_system_main.cpp -- Phase 2 / T2.1 of the spinning-trumpet program.
 *
 * Registers the l=0 row operators with a Kadath System_of_eqs as add_def
 * strings, with the backbone-dependent coefficients imported as add_cst
 * fields, and EVALUATES the five residuals without solving anything.  No
 * add_eq_* call appears here: boundary conditions, the constraint pins and the
 * Newton solve are T2.2/T2.3 (and T2.2 is blocked on SPEC_T22_innerBC.md).
 *
 * THE ORACLE.  The exact mass mode -- U = (1-W)/2, Q = 0, G = F_M - 2 -- is an
 * exact solution of the FULL five-row system at j2 = 0, so every residual must
 * vanish.  That single test exercises all 39 coefficient fields at once
 * against a known answer with no solve (playbook rule 2).  Crucially the mass
 * mode is built HERE from the imported backbone fields, and Kadath then
 * differentiates it SPECTRALLY via dr()/ddr(), while the Python reference
 * (scripts/l0_operators.py) builds the same residual from ANALYTIC
 * derivatives.  Agreement is therefore a genuine independent dual evaluation
 * (playbook rule 3), not a comparison of the same numbers twice.
 *
 * Coefficient naming avoids '_' entirely: a trailing _<char> declares a tensor
 * index in Kadath's string language (cf. "gradN_i = grad(N)"), so names are
 * c<row><jet> with row in {K, XR, XT, H, M} and jet in {U,Up,Upp,...,j2}.
 *
 * Row scaling: row n has been divided by r^p_n on the Python side so every
 * coefficient is bounded at r = infinity.  Scaling a row by a nonzero function
 * does not change its null space, so the solve is unaffected; it only makes
 * the operator representable on the compactified domain.
 *
 * Usage:  l0_system <backbone.dat> <coefs.dat> [<reference.dat>]
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Domain/oned.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include "space/space_oned_trumpet.hpp"
#include "src/table_io.hpp"

#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using Kadath::Dim_array;
using Kadath::Index;
using Kadath::Scalar;
using Kadath::System_of_eqs;
using Kadath::Val_domain;
using TrumpetIO::CoefTable;
using TrumpetIO::RefTable;
using TrumpetIO::Table;

namespace
{

const char* const ROWS[5] = {"E_K", "E_chi_rr", "E_chi_tt", "E_H", "E_Mr"};
const char* const CODE[5] = {"K", "XR", "XT", "H", "M"};
const char* const DEFN[5] = {"EK", "EXR", "EXT", "EH", "EM"};

/** Jet name -> the string that applies it to a field. */
std::string apply_jet(const std::string& jet, const std::string& f)
{
    if (jet == "U" || jet == "Q" || jet == "G")
        return f;
    if (jet == "Up" || jet == "Qp" || jet == "Gp")
        return "dr(" + f + ")";
    if (jet == "Upp" || jet == "Qpp" || jet == "Gpp")
        return "ddr(" + f + ")";
    if (jet == "j2")
        return "jsrc";
    throw std::runtime_error("unknown jet " + jet);
}

std::string field_of(const std::string& jet)
{
    if (jet[0] == 'U')
        return "U";
    if (jet[0] == 'Q')
        return "Q";
    if (jet[0] == 'G')
        return "G";
    return "";
}

void emit(const std::string& k, double v)
{
    std::cout << "RESULT " << k << " " << std::setprecision(17) << v << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "usage: l0_system <backbone.dat> <coefs.dat> [<reference.dat>]\n";
        return 2;
    }

    Table bt;
    CoefTable ct;
    try {
        bt = TrumpetIO::read_table(argv[1]);
        ct = TrumpetIO::read_coefs(argv[2], bt.doms);
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
    const int ndom = static_cast<int>(bt.doms.size());
    if (bt.mode != "excised") {
        std::cerr << "FATAL: T2.1 requires the excised layout, got " << bt.mode << "\n";
        return 1;
    }
    std::cout << "# l=0 system  backbone=" << bt.tag << "  coefs=" << ct.tag
              << "  ndom=" << ndom << "  ncoef=" << ct.ncoef << "\n";
    for (int n = 0; n < 5; n++)
        std::cout << "# row " << ROWS[n] << " scaled by r^-" << ct.scale[ROWS[n]]
                  << "  jets:" << [&] {
                         std::string s;
                         for (const auto& j : ct.jets[ROWS[n]])
                             s += " " + j;
                         return s;
                     }() << "\n";

    // ------------------------------------------------------------- space ----
    std::vector<double> bounds;
    std::vector<Dim_array> res;
    for (int d = 0; d < ndom; d++) {
        bounds.push_back(bt.doms[d].r_int);
        Dim_array n(1);
        n.set(0) = bt.doms[d].nbr;
        res.push_back(n);
    }
    Trumpet::Space_oned_trumpet space(CHEB_TYPE, res, bounds);

    // ------------------------------------------- backbone + derived fields --
    auto fill = [&](Scalar& f, auto get) {
        for (int d = 0; d < ndom; d++) {
            Val_domain& vd = f.set_domain(d);
            vd.allocate_conf();
            Index idx(space.get_domain(d)->get_nbr_points());
            for (int i = 0; i < bt.doms[d].nbr; i++) {
                idx.set(0) = i;
                vd.set(idx) = get(d, i);
            }
        }
        f.std_base();
    };

    Scalar Wf(space), Rrf(space), oorf(space);
    fill(Wf, [&](int d, int i) { return bt.pts[d][i].W; });
    fill(Rrf, [&](int d, int i) { return bt.pts[d][i].Rr; });
    fill(oorf, [&](int d, int i) { return bt.pts[d][i].oor; });

    // 1/R = oor/Rr, bounded everywhere (0 at spatial infinity)
    Scalar iR(space);
    for (int d = 0; d < ndom; d++)
        iR.set_domain(d) = oorf(d) / Rrf(d);
    iR.std_base();

    // ---------------------------------- the exact mass mode as the profile --
    // U = (1-W)/2 ;  Q = 0 ;  G = F_M - 2 with
    // F_M = -(1/R - (27/8) M^3 /R^4)/W   (the Phase-1 G6 form)
    const double M = bt.M;
    Scalar U(space), Q(space), G(space);
    for (int d = 0; d < ndom; d++) {
        U.set_domain(d) = (1.0 - Wf(d)) * 0.5;
        Q.set_domain(d) = 0.0 * Wf(d);
        G.set_domain(d) = -(iR(d) - (27.0 / 8.0) * std::pow(M, 3) * Kadath::pow(iR(d), 4))
                              / Wf(d)
                          - 2.0;
    }
    U.std_base();
    Q.std_base();
    G.std_base();

    // ------------------------------------------------------------ system ----
    System_of_eqs syst(space, 0, ndom - 1);
    syst.add_var("U", U);
    syst.add_var("Q", Q);
    syst.add_var("G", G);
    syst.add_cst("jsrc", 0.0);          // j2 = 0 for the mass-mode oracle

    // 39 coefficient fields.  Stored in a deque so the Scalars keep stable
    // addresses for the lifetime of the system (add_cst borrows a reference).
    std::deque<Scalar> cst;
    std::map<std::string, std::string> nameof;   // "row/jet" -> add_cst name
    for (int n = 0; n < 5; n++) {
        for (const auto& jet : ct.jets[ROWS[n]]) {
            const std::string nm = std::string("c") + CODE[n] + jet;
            auto key = std::make_pair(std::string(ROWS[n]), jet);
            const auto& g = ct.v.at(key);
            cst.emplace_back(space);
            Scalar& s = cst.back();
            fill(s, [&](int d, int i) { return g[d][i]; });
            syst.add_cst(nm.c_str(), s);
            nameof[std::string(ROWS[n]) + "/" + jet] = nm;
        }
    }

    // ------------------------------------------- the five rows as add_def ---
    for (int n = 0; n < 5; n++) {
        std::string rhs;
        for (const auto& jet : ct.jets[ROWS[n]]) {
            const std::string c = nameof[std::string(ROWS[n]) + "/" + jet];
            if (!rhs.empty())
                rhs += " + ";
            rhs += c + " * " + apply_jet(jet, field_of(jet));
        }
        const std::string def = std::string(DEFN[n]) + " = " + rhs;
        std::cout << "# add_def  " << def << "\n";
        syst.add_def(def.c_str());
    }

    // ------------------------------------------------------- evaluate -------
    // The reference table carries the row scale (largest single term) computed
    // from ANALYTIC jets, so B and C are judged on the same normalisation
    // (playbook rule 6).  An earlier version rebuilt the scale here in C++ and
    // read a bare der_abs() result with operator()(Index) -- that value lives
    // in COEFFICIENT space, so the reconstructed scale was garbage and B read
    // exactly 1.  Use the reference scale; it needs the reference table.
    RefTable rt;
    bool have_ref = false;
    if (argc > 3) {
        try {
            rt = TrumpetIO::read_ref(argv[3], bt.doms);
            have_ref = true;
        } catch (const std::exception& e) {
            std::cerr << "FATAL: " << e.what() << "\n";
            return 1;
        }
    }

    if (have_ref) {
        // --- localise first: how accurate are Kadath's SPECTRAL derivatives
        // against the analytic jets?  If a row residual is large this says
        // whether the cause is differentiation or the coefficients.
        struct { const char* jet; const char* def; } probe[] = {
            {"Up", "PUp = dr(U)"},   {"Upp", "PUpp = ddr(U)"},
            {"Gp", "PGp = dr(G)"},   {"Gpp", "PGpp = ddr(G)"},
        };
        for (auto& pr : probe) {
            syst.add_def(pr.def);
            std::string dn(pr.def);
            dn = dn.substr(0, dn.find(' '));
            double num = 0.0, den = 0.0;
            for (int d = 0; d < ndom; d++) {
                const Val_domain& V = syst.give_val_def_scalar_domain(dn.c_str(), d);
                Index idx(space.get_domain(d)->get_nbr_points());
                for (int i = 0; i < bt.doms[d].nbr; i++) {
                    idx.set(0) = i;
                    const double a = rt.jet.at(pr.jet)[d][i];
                    if (!std::isfinite(a))
                        continue;
                    num = std::max(num, std::fabs(V(idx) - a));
                    den = std::max(den, std::fabs(a));
                }
            }
            emit(std::string("A_spectral_deriv_rel_") + pr.jet,
                 num / std::max(den, 1e-300));
        }

        double worst_all = 0.0;
        for (int n = 0; n < 5; n++) {
            double wB = 0.0, wC = 0.0;
            for (int d = 0; d < ndom; d++) {
                const Val_domain& E = syst.give_val_def_scalar_domain(DEFN[n], d);
                Index idx(space.get_domain(d)->get_nbr_points());
                for (int i = 0; i < bt.doms[d].nbr; i++) {
                    idx.set(0) = i;
                    const double sc = rt.scale.at(ROWS[n])[d][i];
                    if (sc <= 0.0)
                        continue;
                    wB = std::max(wB, std::fabs(E(idx)) / sc);
                    wC = std::max(wC,
                                  std::fabs(E(idx) - rt.resid.at(ROWS[n])[d][i]) / sc);
                }
            }
            emit(std::string("B_massmode_rel_") + ROWS[n], wB);
            emit(std::string("C_dual_rel_") + ROWS[n], wC);
            worst_all = std::max(worst_all, wB);
            // per-domain and worst-point location: a global max hides whether
            // the floor is a bulk property or one bad collocation point
            // (playbook rule 5).
            for (int d = 0; d < ndom; d++) {
                const Val_domain& E = syst.give_val_def_scalar_domain(DEFN[n], d);
                Index idx(space.get_domain(d)->get_nbr_points());
                double wd = 0.0; int wi = -1;
                for (int i = 0; i < bt.doms[d].nbr; i++) {
                    idx.set(0) = i;
                    const double sc = rt.scale.at(ROWS[n])[d][i];
                    if (sc <= 0.0)
                        continue;
                    const double q = std::fabs(E(idx)) / sc;
                    if (q > wd) { wd = q; wi = i; }
                }
                std::cout << "# loc " << ROWS[n] << " d" << d << " worst=" << wd
                          << " at ipt=" << wi << "/" << bt.doms[d].nbr - 1 << "\n";
            }
        }
        emit("B_massmode_rel_max", worst_all);
    }

    // -------------------------------- D: ddr(f) must equal lap(f) in 1D -----
    // Two different code paths (Domain::ddr -> der_r twice, versus
    // Domain::laplacian -> der_abs twice) that must agree identically.
    syst.add_def("DDRCHK = ddr(U) - lap(U)");
    double ddr_lap = 0.0, dscale = 0.0;
    for (int d = 0; d < ndom; d++) {
        const Val_domain& D = syst.give_val_def_scalar_domain("DDRCHK", d);
        Val_domain ref = U(d).der_abs(1).der_abs(1);
        Index idx(space.get_domain(d)->get_nbr_points());
        for (int i = 0; i < bt.doms[d].nbr; i++) {
            idx.set(0) = i;
            ddr_lap = std::max(ddr_lap, std::fabs(D(idx)));
            dscale = std::max(dscale, std::fabs(ref(idx)));
        }
    }
    emit("D_ddr_minus_lap_rel", ddr_lap / std::max(dscale, 1e-300));

    std::cout << "# done\n";
    return 0;
}

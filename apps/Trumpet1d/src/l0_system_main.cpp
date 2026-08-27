/*
 * l0_system_main.cpp -- Phase 2 / T2.1 of the spinning-trumpet program.
 *
 * EVALUATES the five l=0 residuals without solving anything.  The operator
 * itself lives in src/l0_setup.hpp, shared with the T2.2 solver app so that
 * what this gate certifies is literally what T2.2 solves.  No add_eq_* call
 * appears here.
 *
 * THE ORACLE.  The exact mass mode -- U = (1-W)/2, Q = 0, G = F_M - 2 -- is an
 * exact solution of the FULL five-row system at j2 = 0, so every residual must
 * vanish.  That single test exercises all 39 coefficient fields at once
 * against a known answer with no solve (playbook rule 2).  Crucially the mass
 * mode is built from the imported backbone fields and Kadath then
 * differentiates it SPECTRALLY via dr()/ddr(), while the Python reference
 * (scripts/l0_operators.py) builds the same residual from ANALYTIC
 * derivatives.  Agreement is therefore a genuine independent dual evaluation
 * (playbook rule 3), not a comparison of the same numbers twice.
 *
 * Usage:  l0_system <backbone.dat> <coefs.dat> [<reference.dat>]
 */

#include "src/l0_setup.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

using Kadath::Index;
using Kadath::Scalar;
using Kadath::System_of_eqs;
using Kadath::Val_domain;
using TrumpetIO::RefTable;

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "usage: l0_system <backbone.dat> <coefs.dat> [<reference.dat>]\n";
        return 2;
    }

    std::unique_ptr<Trumpet::L0Model> mp;
    try {
        mp = std::make_unique<Trumpet::L0Model>(argv[1], argv[2]);
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
    Trumpet::L0Model& m = *mp;
    const int ndom = m.nb_domains();
    m.print_banner();

    m.set_fields_massmode();

    System_of_eqs syst(m.space, 0, ndom - 1);
    for (const auto& d : m.register_rows(syst, 0.0))   // j2 = 0 for the oracle
        std::cout << "# add_def  " << d << "\n";

    const char* const* ROWS = Trumpet::rows();
    const char* const* DEFN = Trumpet::row_defs();

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
            rt = TrumpetIO::read_ref(argv[3], m.table().doms);
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
            int wd = -1, wi = -1;
            std::vector<double> perdom(ndom, 0.0);
            for (int d = 0; d < ndom; d++) {
                const Val_domain& V = syst.give_val_def_scalar_domain(dn.c_str(), d);
                Index idx(m.space.get_domain(d)->get_nbr_points());
                for (int i = 0; i < m.nbr(d); i++) {
                    idx.set(0) = i;
                    const double a = rt.jet.at(pr.jet)[d][i];
                    if (!std::isfinite(a))
                        continue;
                    const double e = std::fabs(V(idx) - a);
                    perdom[d] = std::max(perdom[d], e);
                    if (e > num) { num = e; wd = d; wi = i; }
                    den = std::max(den, std::fabs(a));
                }
            }
            Trumpet::emit(std::string("A_spectral_deriv_rel_") + pr.jet,
                          num / std::max(den, 1e-300));
            // WHERE the derivative error lives.  A global max cannot tell a
            // narrow throat shell (where d/dr = (1/alpha) d/dx amplifies
            // roundoff as (N/alpha)^2) from the compactified domain (whose
            // outermost interior point stretches like N^2).  Playbook rule 5.
            std::cout << "# locA " << pr.jet << " worst d" << wd << " ipt " << wi
                      << " ; per-domain abs:";
            for (int d = 0; d < ndom; d++)
                std::cout << " d" << d << "=" << perdom[d];
            std::cout << "  (den=" << den << ")\n";
        }

        double worst_all = 0.0;
        for (int n = 0; n < 5; n++) {
            double wB = 0.0, wC = 0.0;
            for (int d = 0; d < ndom; d++) {
                const Val_domain& E = syst.give_val_def_scalar_domain(DEFN[n], d);
                Index idx(m.space.get_domain(d)->get_nbr_points());
                for (int i = 0; i < m.nbr(d); i++) {
                    idx.set(0) = i;
                    const double sc = rt.scale.at(ROWS[n])[d][i];
                    if (sc <= 0.0)
                        continue;
                    wB = std::max(wB, std::fabs(E(idx)) / sc);
                    wC = std::max(wC,
                                  std::fabs(E(idx) - rt.resid.at(ROWS[n])[d][i]) / sc);
                }
            }
            Trumpet::emit(std::string("B_massmode_rel_") + ROWS[n], wB);
            Trumpet::emit(std::string("C_dual_rel_") + ROWS[n], wC);
            worst_all = std::max(worst_all, wB);
            // per-domain and worst-point location: a global max hides whether
            // the floor is a bulk property or one bad collocation point
            // (playbook rule 5).
            for (int d = 0; d < ndom; d++) {
                const Val_domain& E = syst.give_val_def_scalar_domain(DEFN[n], d);
                Index idx(m.space.get_domain(d)->get_nbr_points());
                double wd = 0.0; int wi = -1;
                for (int i = 0; i < m.nbr(d); i++) {
                    idx.set(0) = i;
                    const double sc = rt.scale.at(ROWS[n])[d][i];
                    if (sc <= 0.0)
                        continue;
                    const double q = std::fabs(E(idx)) / sc;
                    if (q > wd) { wd = q; wi = i; }
                }
                std::cout << "# loc " << ROWS[n] << " d" << d << " worst=" << wd
                          << " at ipt=" << wi << "/" << m.nbr(d) - 1 << "\n";
            }
        }
        Trumpet::emit("B_massmode_rel_max", worst_all);
    }

    // -------------------------------- D: ddr(f) must equal lap(f) in 1D -----
    // Two different code paths (Domain::ddr -> der_r twice, versus
    // Domain::laplacian -> der_abs twice) that must agree identically.
    syst.add_def("DDRCHK = ddr(U) - lap(U)");
    double ddr_lap = 0.0, dscale = 0.0;
    for (int d = 0; d < ndom; d++) {
        const Val_domain& D = syst.give_val_def_scalar_domain("DDRCHK", d);
        Val_domain ref = m.U(d).der_abs(1).der_abs(1);
        Index idx(m.space.get_domain(d)->get_nbr_points());
        for (int i = 0; i < m.nbr(d); i++) {
            idx.set(0) = i;
            ddr_lap = std::max(ddr_lap, std::fabs(D(idx)));
            dscale = std::max(dscale, std::fabs(ref(idx)));
        }
    }
    Trumpet::emit("D_ddr_minus_lap_rel", ddr_lap / std::max(dscale, 1e-300));

    std::cout << "# done\n";
    return 0;
}

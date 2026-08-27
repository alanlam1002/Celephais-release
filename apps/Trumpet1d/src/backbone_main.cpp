/*
 * backbone_main.cpp -- Phase 1 (T1.1-T1.3) of the spinning-trumpet program.
 *
 * Imports the exact trumpet backbone onto a 1-D spectral grid, measures the
 * spectral representation, and evaluates the T1.3 backbone oracles.  It does
 * NOT build a System_of_eqs and does not solve anything -- that is Phase 2.
 *
 * Input is the table written by spinning_trumpet/scripts/backbone_tabulate.py,
 * which carries the domain bounds at full double precision AND the backbone
 * sampled at the collocation points those bounds imply.  Nothing is
 * interpolated: the app rebuilds the same space, asserts its own
 * Domain::get_radius() against the table's r column (gate G1 -- which is also
 * the check on the collocation formulas recorded in NOTES_kadath_api.md), and
 * loads the values point by point.
 *
 * STORED FIELDS.  Not R: it diverges on the compactified domain, where r =
 * infinity is a real collocation point.  We store
 *      W    -> 1        Rr = R/r -> 1        oor = 1/r -> 0
 * all bounded, all smooth in 1/r (which is LINEAR in the numerical coordinate
 * x on Domain_oned_inf).  Then 1/R = oor/Rr and a2 = 1 - 2(oor/Rr) +
 * (27/16)(oor/Rr)^4 are pure pointwise arithmetic -- needing neither div_r
 * (only on Domain_oned_ori) nor mult_r (only on Domain_oned_inf).
 *
 * Results are emitted as "RESULT <key> <value>" lines; gates/gate_phase1_backbone.sh
 * parses them and asserts against gates/oracle_values.json.
 *
 * Usage:  backbone <table.dat> [--offgrid N]
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Domain/oned.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"

#include "space/space_oned_trumpet.hpp"
#include "src/table_io.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using Kadath::Dim_array;
using Kadath::Index;
using Kadath::Scalar;
using Kadath::Val_domain;
using TrumpetIO::DomainSpec;
using TrumpetIO::PointRow;
using TrumpetIO::Table;
using TrumpetIO::read_table;

namespace
{

/** Fill one Val_domain from a per-point accessor. */
template <typename F>
void fill(Scalar& f, int d, const Table& t, F get)
{
    Val_domain& vd = f.set_domain(d);
    vd.allocate_conf();
    Index idx(f.get_space().get_domain(d)->get_nbr_points());
    for (int i = 0; i < t.doms[d].nbr; i++) {
        idx.set(0) = i;
        vd.set(idx) = get(t.pts[d][i]);
    }
}

/** log10 of the largest |coefficient| in the top `tail` slots, and a fitted
 *  exponential rate.  Per-domain only -- never a global fit (playbook rule 5). */
struct Decay
{
    std::vector<double> mag;   // |c_i| normalised by max
    double rate = 0.0;         // fitted  log10|c_i| ~ -rate * i
    double floor = 0.0;        // log10 of the smallest resolved |c_i|/max
};

Decay coef_decay(const Val_domain& vd, int ncoef)
{
    vd.coef();
    Decay out;
    Index pos(vd.get_domain()->get_nbr_coefs());
    double mx = 0.0;
    for (int i = 0; i < ncoef; i++) {
        pos.set(0) = i;
        double c = std::fabs(vd.get_coef(pos));
        out.mag.push_back(c);
        mx = std::max(mx, c);
    }
    if (mx <= 0.0)
        return out;
    for (double& c : out.mag)
        c /= mx;
    // least squares on log10|c_i| over the range before it hits the roundoff
    // floor; using the plateau would fit noise, not decay.
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int n = 0;
    for (int i = 0; i < ncoef; i++) {
        if (out.mag[i] < 1e-15)
            break;
        double y = std::log10(out.mag[i]);
        sx += i;
        sy += y;
        sxx += double(i) * i;
        sxy += double(i) * y;
        n++;
    }
    if (n >= 3)
        out.rate = -(n * sxy - sx * sy) / (n * sxx - sx * sx);
    double lo = 1.0;
    for (double c : out.mag)
        lo = std::min(lo, c > 0 ? c : lo);
    out.floor = std::log10(lo);
    return out;
}

void emit(const std::string& key, double v)
{
    std::cout << "RESULT " << key << " " << std::setprecision(17) << v << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "usage: backbone <table.dat> [--offgrid N]\n";
        return 2;
    }
    int noff = 10;
    for (int i = 2; i < argc; i++)
        if (std::string(argv[i]) == "--offgrid" && i + 1 < argc)
            noff = std::atoi(argv[++i]);

    Table t;
    try {
        t = read_table(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }

    const int ndom = static_cast<int>(t.doms.size());
    const bool stock = (t.mode == "stock");

    std::cout << "# table  " << t.tag << "  mode=" << t.mode << "  ndom=" << ndom
              << "  M=" << t.M << "\n";

    // ---------------------------------------------------------------- space --
    std::vector<double> bounds;
    std::vector<Dim_array> res;
    for (int d = 0; d < ndom; d++) {
        bounds.push_back(t.doms[d].r_int);
        Dim_array n(1);
        n.set(0) = t.doms[d].nbr;
        res.push_back(n);
    }

    // Kadath::Space has a PROTECTED destructor, so it cannot be owned through a
    // unique_ptr<Space>; hold the concrete type and alias it.
    std::unique_ptr<Kadath::Space_oned> stock_space;
    std::unique_ptr<Trumpet::Space_oned_trumpet> excised_space;
    Kadath::Space* space = nullptr;
    if (stock) {
        // Stock Kadath::Space_oned: bounds are the INTERFACES and domain 0 is
        // an r=0 nucleus, so bounds[0] (== 0) is dropped.
        Kadath::Array<double> kb(ndom - 1);
        for (int d = 1; d < ndom; d++)
            kb.set(d - 1) = t.doms[d].r_int;
        stock_space = std::make_unique<Kadath::Space_oned>(CHEB_TYPE, res[0], kb);
        space = stock_space.get();
    } else {
        excised_space =
            std::make_unique<Trumpet::Space_oned_trumpet>(CHEB_TYPE, res, bounds);
        space = excised_space.get();
    }

    if (space->get_nbr_domains() != ndom) {
        std::cerr << "FATAL: space has " << space->get_nbr_domains()
                  << " domains, table has " << ndom << "\n";
        return 1;
    }

    // ------------------------------------------ G1: collocation points match --
    double g1 = 0.0;
    for (int d = 0; d < ndom; d++) {
        Val_domain rad = space->get_domain(d)->get_radius();
        Index idx(space->get_domain(d)->get_nbr_points());
        for (int i = 0; i < t.doms[d].nbr; i++) {
            idx.set(0) = i;
            const double rk = rad(idx);
            const double rf = t.pts[d][i].r;
            if (!std::isfinite(rf) || !std::isfinite(rk)) {
                // the r = infinity collocation point of the compactified
                // domain: Kadath computes 1/(alpha*(x-1)) with x == 1 exactly,
                // so it is a signed infinity.  Only require both agree on
                // being non-finite.
                if (std::isfinite(rf) != std::isfinite(rk)) {
                    std::cerr << "FATAL: finiteness mismatch at d=" << d
                              << " i=" << i << " kadath=" << rk << " table=" << rf << "\n";
                    return 1;
                }
                continue;
            }
            g1 = std::max(g1, std::fabs(rk - rf) / std::max(1.0, std::fabs(rf)));
        }
    }
    emit("G1_radius_max_rel_diff", g1);

    // ------------------------------------------------------- import backbone --
    Scalar Wf(*space), Rrf(*space), oorf(*space);
    for (int d = 0; d < ndom; d++) {
        fill(Wf, d, t, [](const PointRow& p) { return p.W; });
        fill(Rrf, d, t, [](const PointRow& p) { return p.Rr; });
        fill(oorf, d, t, [](const PointRow& p) { return p.oor; });
    }
    Wf.std_base();
    Rrf.std_base();
    oorf.std_base();

    // ------------------------------------------------- T1.2 coefficient decay --
    std::cout << "# per-domain spectral coefficient decay (normalised |c_i|)\n";
    double worst_rate = 1e30;
    for (int d = 0; d < ndom; d++) {
        const int nc = space->get_domain(d)->get_nbr_coefs()(0);
        Decay dw = coef_decay(Wf(d), nc);
        std::cout << "# d" << d << " " << t.doms[d].kind << " rate=" << std::fixed
                  << std::setprecision(3) << dw.rate
                  << " floor=1e" << std::setprecision(1) << dw.floor << "  |c|/max:";
        for (int i = 0; i < nc; i++)
            std::cout << " " << std::scientific << std::setprecision(1) << dw.mag[i];
        std::cout << std::defaultfloat << "\n";
        emit("decay_rate_W_d" + std::to_string(d), dw.rate);
        emit("decay_floor_W_d" + std::to_string(d), dw.floor);
        if (!stock || d > 0)
            worst_rate = std::min(worst_rate, dw.rate);
    }
    emit("decay_rate_W_worst", worst_rate);

    // ------------------------------------------------ derived bounded fields --
    // 1/R = oor/Rr  (finite everywhere, 0 at spatial infinity)
    Scalar iR(*space);
    for (int d = 0; d < ndom; d++)
        iR.set_domain(d) = oorf(d) / Rrf(d);
    iR.std_base();

    // a2 = 1 - 2 M /R + 27 M^4/(16 R^4)
    const double M = t.M;
    Scalar a2(*space);
    for (int d = 0; d < ndom; d++)
        a2.set_domain(d) = 1.0 - 2.0 * M * iR(d)
                           + (27.0 / 16.0) * std::pow(M, 4) * Kadath::pow(iR(d), 4);
    a2.std_base();

    // ------------------------------------------ G2: a2(R) - W^2 on the grid --
    double g2 = 0.0;
    for (int d = 0; d < ndom; d++) {
        if (stock && d == 0)
            continue;              // nucleus: Rr/oor are sentinels at r=0
        Val_domain res_vd = a2(d) - Wf(d) * Wf(d);
        Index idx(space->get_domain(d)->get_nbr_points());
        for (int i = 0; i < t.doms[d].nbr; i++) {
            idx.set(0) = i;
            g2 = std::max(g2, std::fabs(res_vd(idx)));
        }
    }
    emit("G2_a2_minus_W2_max", g2);

    // -------------------------------------- G3: same, at off-grid points ----
    // Tests the INTERPOLANT, not the imported data.  Skip the compactified
    // domain's outer reach where r is unbounded; sample its finite part.
    double g3 = 0.0;
    for (int d = 0; d < ndom; d++) {
        const double a = t.doms[d].r_int;
        double b = t.doms[d].r_ext;
        if (!std::isfinite(b))
            b = a * 8.0;                       // finite window inside the inf domain
        if (stock && d == 0)
            continue;                          // nucleus: r=0 branch point, see control
        for (int k = 1; k <= noff; k++) {
            const double rr = a + (b - a) * (double(k) - 0.5) / double(noff);
            Kadath::Point pt(1);
            pt.set(1) = rr;
            const double wv = Wf.val_point(pt);
            const double iv = iR.val_point(pt);
            const double av = 1.0 - 2.0 * M * iv + (27.0 / 16.0) * std::pow(M, 4)
                                                       * std::pow(iv, 4);
            g3 = std::max(g3, std::fabs(av - wv * wv));
        }
    }
    emit("G3_a2_minus_W2_offgrid_max", g3);

    // ------------------------------ G4: spectral ODE residual  dR/dr = R W/r --
    // In bounded variables (R = r Rr):   dRr/dr - Rr (W-1)/r = 0,
    // i.e.  Rr' - Rr (W-1) oor = 0.  Uses Kadath's own der_abs(1) = d/dr, so
    // this tests the derivative operator Phase 2 depends on.
    double g4 = 0.0;
    for (int d = 0; d < ndom; d++) {
        if (stock && d == 0)
            continue;              // nucleus: Rr/oor are sentinels at r=0
        Val_domain dRr = Rrf(d).der_abs(1);
        Val_domain resid = dRr - Rrf(d) * (Wf(d) - 1.0) * oorf(d);
        Index idx(space->get_domain(d)->get_nbr_points());
        double num = 0.0, scale = 0.0;
        for (int i = 0; i < t.doms[d].nbr; i++) {
            idx.set(0) = i;
            num = std::max(num, std::fabs(resid(idx)));
            scale = std::max(scale, std::fabs(dRr(idx)));
        }
        const double rel = num / std::max(scale, 1e-300);
        emit("G4_ode_rel_d" + std::to_string(d), rel);
        g4 = std::max(g4, rel);
    }
    emit("G4_ode_rel_max", g4);

    // ------------------------------------------------ G5: W at the excision --
    if (!stock) {
        Index pcf(space->get_domain(0)->get_nbr_coefs());
        const double win = space->get_domain(0)->val_boundary(INNER_BC, Wf(0), pcf);
        emit("G5_W_at_inner", win);
        emit("G5_W_at_inner_err", std::fabs(win - t.W0));
    }

    // --------------------------------------------- G6: mass-mode tails at inf --
    // U_M = (1-W)/2 ~ tU/r,  F_M = -(1/R - (27/8)M^3/R^4)/W ~ tG/r.
    // On Domain_oned_inf, mult_r exists and val_boundary(OUTER_BC,.) IS the
    // value at r = infinity, so the tail amplitude is a point read.  This is
    // design decision D2, validated here against exactly known answers.
    {
        const int dl = ndom - 1;
        const Kadath::Domain* dom = space->get_domain(dl);
        Val_domain U_M = (1.0 - Wf(dl)) * 0.5;
        Val_domain F_M = -(iR(dl) - (27.0 / 8.0) * std::pow(M, 3) * Kadath::pow(iR(dl), 4))
                         / Wf(dl);
        // Q_M vanishes identically for the mass mode.  T4.1 proved the Q-sector
        // tail vanishes in EVERY physical decaying mode, making tQ a null test,
        // so read it through the same mult_r/val_boundary path rather than
        // asserting zero by inspection.
        Val_domain Q_M = 0.0 * Wf(dl);
        Index pcf(dom->get_nbr_coefs());
        const double tU = dom->val_boundary(OUTER_BC, dom->mult_r(U_M), pcf);
        const double tG = dom->val_boundary(OUTER_BC, dom->mult_r(F_M), pcf);
        const double tQ = dom->val_boundary(OUTER_BC, dom->mult_r(Q_M), pcf);
        emit("G6_tU", tU);
        emit("G6_tQ", tQ);
        emit("G6_tG", tG);
        // T4.1: the physical decaying tail direction is (1,0,-2), so the Komar
        // combination 2 tU + tG is the contamination-immune observable and must
        // vanish.  This is the Phase-3 verdict quantity, exercised here one
        // phase early on a case whose answer is known exactly.
        emit("G6_komar_2tU_plus_tG", 2.0 * tU + tG);
        // ratio tG/tU = -2 is the direction itself, reported for the record
        emit("G6_tG_over_tU", tG / tU);
        // and the plain (untailed) values at infinity, which must vanish
        emit("G6_U_at_inf", dom->val_boundary(OUTER_BC, U_M, pcf));
        emit("G6_W_at_inf", dom->val_boundary(OUTER_BC, Wf(dl), pcf));
    }

    // ------------------------------------- G7: throat series R = 3/2 + c_k W^k --
    if (!stock) {
        // R(W) = 3/2 + sum c_k W^k, by exact series reversion at dps=50; see
        // spinning_trumpet/scripts/backbone_constants.py.  Eight terms leave a
        // truncation error of 5.94e-5 at the domain-0 outer edge W = 3sqrt3/16.
        static const double c[8] = {1.0606601717798212866,  1.0,
                                    0.98700321540622258614, 0.98611111111111111111,
                                    0.98853773533067255907, 0.99151234567901234568,
                                    0.99413447049956926588, 0.99617412551440329218};
        double g7 = 0.0;
        Index idx(space->get_domain(0)->get_nbr_points());
        for (int i = 0; i < t.doms[0].nbr; i++) {
            idx.set(0) = i;
            const double w = t.pts[0][i].W;
            const double Rv = t.pts[0][i].Rr * t.pts[0][i].r;
            double ser = 1.5 * M;
            double wp = 1.0;
            for (double ck : c) {
                wp *= w;
                ser += ck * wp * M;
            }
            g7 = std::max(g7, std::fabs(Rv - ser));
        }
        emit("G7_throat_series_max_abs", g7);
    }

    std::cout << "# done\n";
    return 0;
}

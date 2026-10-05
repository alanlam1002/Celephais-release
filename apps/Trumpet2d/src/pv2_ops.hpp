// ⚠ PUNCTURE V2, PHASE 2a (research round 703; finiteJ_probe_bulk2 only, TRUMPET_BULK2).  Kadath's parser has exp and log
// but no expm1: exponentials of REMAINDERS (puncture formulation, PUNCTURE_V2_PLAN.md) must go through expm1, so that a
// small remainder is not swamped by the 1 that exp(x) - 1 subtracts.  This is a user operator for System_of_eqs::add_ope
// (the library is not touched), registered under the name `expmone` (no digit in a parser name).
//   value lane:       expm1(x) at every configuration point (std::expm1), the argument's spectral base kept (as Ope_exp);
//   derivative lanes: d expm1(x) = exp(x) dx in EVERY tangent lane present (as the library's own scalar unary helper,
//                     src/Ope_eq/ope_scalar_unary_operator.hpp, does for exp) -- there is no cancellation in this lane.
// Scalars only (valence 0), TERM_T and TERM_D.
#pragma once
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <array>
#include <tuple>
#include <cctype>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <ostream>
#include <vector>

namespace pv2 {

inline Kadath::Val_domain expm1_vd(const Kadath::Val_domain& so)
{
    if (so.check_if_zero()) return so;                     // expm1(0) = 0
    so.coef_i();
    Kadath::Val_domain res(so.get_domain());
    res.allocate_conf();
    Kadath::Index pos(so.get_conf().get_dimensions());
    do res.set(pos) = std::expm1(so(pos)); while (pos.inc());
    res.set_base() = so.get_base();
    return res;
}

inline Kadath::Val_domain dexpm1_vd(const Kadath::Val_domain& der, const Kadath::Val_domain& val)
{
    if (der.check_if_zero()) return der;
    return der * exp(val);
}

inline Kadath::Scalar scalar_lane(const Kadath::Tensor& T, const Kadath::Tensor& V, int dom, bool value)
{
    if (T.get_valence() != 0) throw std::runtime_error("expmone: defined for scalars only");
    Kadath::Scalar s(T, true);
    Kadath::Scalar out(s, false);
    if (value) out.set_domain(dom) = expm1_vd(s(dom));
    else {
        const Kadath::Scalar v(V, true);
        out.set_domain(dom) = dexpm1_vd(s(dom), v(dom));
    }
    return out;
}

// the add_ope entry point
inline Kadath::Term_eq expm1_action(const Kadath::Term_eq& t, Kadath::Param*)
{
    const int dom = t.get_dom();
    if (t.get_type_data() == Kadath::TERM_D) {
        Kadath::Term_eq r(dom, std::expm1(t.get_val_d()));
        const double e = std::exp(t.get_val_d());
        if (t.has_der_d(0)) r.set_der_d(t.get_der_d() * e);
        for (int lane = 1; lane < t.get_derivative_lane_count(); ++lane)
            if (t.has_der_d(lane)) r.set_der_d(lane, t.get_der_d(lane) * e);
        return r;
    }
    if (t.get_type_data() != Kadath::TERM_T) throw std::runtime_error("expmone: unknown Term_eq storage");
    const Kadath::Tensor& V = t.get_val_t();
    Kadath::Term_eq r(dom, scalar_lane(V, V, dom, true));
    if (t.has_der_t(0)) r.set_der_t(scalar_lane(t.get_der_t(0), V, dom, false));
    for (int lane = 1; lane < t.get_derivative_lane_count(); ++lane)
        if (t.has_der_t(lane)) r.set_der_t(lane, scalar_lane(t.get_der_t(lane), V, dom, false));
    return r;
}


// ⚠ GATE (research round 703): --pv2-expm1-test FILE, then exit.  At arguments a in {1e-12, 1e-6, 1e-2, 1} (and -a), on a
// smooth field X = a (1 + 0.25 cos^2 th / (1 + r)) over every domain of the run's own space:
//   (A) the operator called directly on a Term_eq carrying TWO tangent lanes (D0 = 1 + 0.5 sin^2 th, D1 = cos th^2 r/(1+r)):
//       value against std::expm1 at every node (and, for scale, the naive exp(x) - 1); each lane against the fourth-order
//       central difference (-f(2h) + 8 f(h) - 8 f(-h) + f(-2h)) / 12h of std::expm1(X + s D), h = 1e-3 |a|;
//   (B) through the parser (add_ope on a fresh System_of_eqs over domains 0..dmax, X a VARIABLE): the def XE = expmone(XT)
//       read back against std::expm1; the row expmone(XT) = 0 (add_eq_full, zeroth order) -- its Jacobian columns (do_col_J,
//       one sequential pass) against the same difference of sec_member in that coefficient (Kadath's own update).
inline double expm1_selftest(const Kadath::Space& space, int dmax, std::ostream& out)
{
    double worst = 0.0;
    auto fd4 = [](auto f, double h) { return (-f(2 * h) + 8 * f(h) - 8 * f(-h) + f(-2 * h)) / (12 * h); };
    for (double a0 : {1e-12, 1e-6, 1e-2, 1.0})
        for (double a : {a0, -a0}) {
            Kadath::Scalar X(space), D0(space), D1(space);
            for (int d = 0; d < space.get_nbr_domains(); d++) {
                const Kadath::Domain* dm = space.get_domain(d);
                for (Kadath::Scalar* s : {&X, &D0, &D1}) s->set_domain(d).allocate_conf();
                Kadath::Index ix(dm->get_nbr_points());
                do {
                    const double r = dm->get_radius()(ix), th = dm->get_coloc(2)(ix(1));
                    const double g = std::isfinite(r) ? 1.0 / (1.0 + r) : 0.0, gr = std::isfinite(r) ? r / (1.0 + r) : 1.0;
                    X.set_domain(d).set(ix) = a * (1.0 + 0.25 * std::cos(th) * std::cos(th) * g);
                    D0.set_domain(d).set(ix) = 1.0 + 0.5 * std::sin(th) * std::sin(th);
                    D1.set_domain(d).set(ix) = std::cos(th) * std::cos(th) * gr;
                } while (ix.inc());
            }
            X.std_base(); D0.std_base(); D1.std_base();
            double eA = 0, eN = 0, eL[2] = {0, 0}, eB = 0, eJ = 0;
            for (int d = 0; d <= dmax; d++) {
                Kadath::Term_eq t(d, X, D0);
                t.set_derivative_lane_count(2);
                t.set_der_t(1, D1);
                const Kadath::Term_eq r = expm1_action(t, nullptr);
                const Kadath::Scalar rv(r.get_val_t(), true), r0(r.get_der_t(0), true), r1(r.get_der_t(1), true);
                Kadath::Index ix(space.get_domain(d)->get_nbr_points());
                do {
                    const double x = X(d)(ix), ex = std::expm1(x);
                    eA = std::max(eA, std::fabs(rv(d)(ix) - ex) / std::fabs(ex));
                    eN = std::max(eN, std::fabs((std::exp(x) - 1.0) - ex) / std::fabs(ex));
                    const Kadath::Scalar* L[2] = {&D0, &D1};
                    const Kadath::Scalar* R[2] = {&r0, &r1};
                    for (int l = 0; l < 2; l++) {
                        const double dl = (*L[l])(d)(ix);
                        if (dl == 0.0) continue;
                        const double hs = 1e-3 * std::fabs(a) / std::fabs(dl);      // the argument moves by 1e-3 |a|
                        const double fd = fd4([&](double s) { return std::expm1(x + s * dl); }, hs);
                        eL[l] = std::max(eL[l], std::fabs((*R[l])(d)(ix) - fd) / std::max(std::fabs(fd), 1e-300));
                    }
                } while (ix.inc());
            }
            // (B) through the parser, a VARIABLE on a fresh system
            {
                Kadath::System_of_eqs ts(space, 0, dmax);
                Kadath::Scalar XV(X);
                static Kadath::Param par;
                ts.add_var("XT", XV);
                ts.add_ope("expmone", &expm1_action, &par);
                for (int d = 0; d <= dmax; d++) {
                    ts.add_def(d, "XE = expmone(XT)");
                    const Kadath::Val_domain& v = ts.give_val_def_scalar_domain("XE", d);
                    Kadath::Index ix(space.get_domain(d)->get_nbr_points());
                    do {
                        const double ex = std::expm1(X(d)(ix));
                        eB = std::max(eB, std::fabs(v(ix) - ex) / std::fabs(ex));
                    } while (ix.inc());
                }
                for (int d = 0; d <= dmax; d++) ts.add_eq_full(d, "expmone(XT) = 0");
                const Kadath::Array<double> s0(ts.sec_member());
                const int nu = ts.get_nbr_unknowns(), nr = s0.get_size(0);
                std::vector<int> want;
                for (int c = 0; c < nu; c += std::max(1, nu / 7)) want.push_back(c);
                std::map<int, std::vector<double>> jc;
                ts.reset_do_col_J_cache();
                for (int c = 0; c < nu; c++) {
                    const Kadath::Array<double> col(ts.do_col_J(c));
                    if (std::find(want.begin(), want.end(), c) != want.end()) {
                        std::vector<double>& v = jc[c];
                        for (int i = 0; i < nr; i++) v.push_back(col(i));
                    }
                }
                const double h = 1e-3 * std::fabs(a);
                for (int c : want) {
                    auto shift = [&](double dd) {
                        Kadath::Array<double> Xd(nu);
                        Xd = 0.0;
                        Xd.set(c) = -dd;
                        int conte = 0;
                        ts.xx_to_vars_delta(Xd, conte);
                    };
                    std::vector<std::vector<double>> F;
                    for (double sgn : {1.0, 2.0, -1.0, -2.0}) {
                        shift(sgn * h);
                        const Kadath::Array<double> sv(ts.sec_member());
                        F.emplace_back(nr);
                        for (int i = 0; i < nr; i++) F.back()[i] = sv(i);
                        shift(-sgn * h);
                    }
                    double jm = 0, dm = 0;
                    for (int i = 0; i < nr; i++) {
                        const double fd = (-F[1][i] + 8 * F[0][i] - 8 * F[2][i] + F[3][i]) / (12 * h);
                        jm = std::max(jm, std::fabs(jc[c][i]));
                        dm = std::max(dm, std::fabs(jc[c][i] - fd));
                    }
                    eJ = std::max(eJ, jm > 0 ? dm / jm : dm);
                }
                out << "    (B) " << nu << " unknowns, " << nr << " rows, " << want.size() << " columns checked\n";
            }
            out << std::setprecision(3) << "  a " << std::setw(7) << a << ":  (A) value " << eA << " (naive exp-1 " << eN
                << ")  lane0 " << eL[0] << "  lane1 " << eL[1] << "   (B) parser value " << eB << "  Jacobian columns vs FD "
                << eJ << "\n";
            worst = std::max({worst, eA, eL[0], eL[1], eB, eJ});
        }
    return worst;
}

}  // namespace pv2

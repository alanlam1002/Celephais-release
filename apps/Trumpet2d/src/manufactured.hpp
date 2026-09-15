/*
 * manufactured.hpp -- the manufactured field shared by A0's measurement rungs.
 *
 * Extracted from theta2d_main.cpp (round 90) when floor2d (round 91) needed the
 * same field.  The defining property is that BOTH factors are exactly known:
 * the angular one has closed-form spectral coefficients q^k, and the radial one
 * is a simple rational whose first two r-derivatives are written analytically.
 * So every reference value is exact to machine precision and no error in the
 * battery can be blamed on the reference.
 */
#ifndef TRUMPET2D_MANUFACTURED_HPP
#define TRUMPET2D_MANUFACTURED_HPP

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

namespace Trumpet
{

/** q^KMAX must be below double roundoff for every q in use: 0.6^80 ~ 1e-18. */
constexpr int KMAX = 80;

inline void emit(const std::string& key, double v)
{
    std::cout << "RESULT " << key << " " << std::setprecision(17) << v << "\n";
}

/**
 * Angular profile with exactly known spectral content.  kind: 0 even, 1 odd,
 * 2 flat.
 *
 *   even (COS_EVEN, cos 2k.theta -- the scalars)
 *       f_e = 1/(1 - 2q cos2theta + q^2),  c_0 = 1/(1-q^2), c_k = 2q^k/(1-q^2)
 *   odd  (COS_ODD,  cos(2k+1)theta -- beta^theta, Qbar)
 *       f_o = (1-q) cos theta/(1 - 2q cos2theta + q^2),  c_k = q^k
 *
 * f, f' and f'' are summed TERM BY TERM rather than differentiated in closed
 * form, so the reference carries no hand-algebra risk.
 */
struct Angular {
    int kind;
    double q;

    double coef(int k) const
    {
        if (kind == 2)
            return (k == 0) ? 1.0 : 0.0;
        if (kind == 1)
            return std::pow(q, k);
        return (k == 0) ? 1.0 / (1.0 - q * q) : 2.0 * std::pow(q, k) / (1.0 - q * q);
    }
    double d(int deg, double th) const
    {
        if (kind == 2)
            return (deg == 0) ? 1.0 : 0.0;
        double s = 0.0;
        for (int k = 0; k <= KMAX; k++) {
            const double m = (kind == 1) ? (2 * k + 1) : (2 * k);
            const double c = coef(k);
            switch (deg) {
                case 0: s += c * std::cos(m * th); break;
                case 1: s += -c * m * std::sin(m * th); break;
                default: s += -c * m * m * std::cos(m * th); break;
            }
        }
        return s;
    }
};

/** Radial profile A(r) = 1/(1+(r/L)^2), decaying like r^-2 so the compact
 *  domain's variable represents it, with its first two r-derivatives. */
struct Radial {
    double L;
    double d(int deg, double r) const
    {
        const double s = r / L, u = 1.0 + s * s;
        if (deg == 0)
            return 1.0 / u;
        if (deg == 1)
            return -2.0 * s / (L * u * u);
        return -2.0 * (1.0 - 3.0 * s * s) / (L * L * u * u * u);
    }
};

/**
 * One error channel: the max absolute deviation and the max |exact| beside it.
 *
 * The operators carry very different scales (d_tt multiplies by m^2, the
 * laplacian by 1/r^2), so an absolute max would compare incommensurables.  Where
 * the exact quantity is identically zero the absolute error is reported instead
 * -- dividing by a zero scale is how the round-90 first draft reported 1e+286.
 */
struct Chan {
    double err = 0, scale = 0;
    void add(double got, double want)
    {
        const double e = std::fabs(got - want);
        if (e > err) err = e;
        if (std::fabs(want) > scale) scale = std::fabs(want);
    }
    double rel() const { return (scale > 1e-290) ? err / scale : err; }
};

} // namespace Trumpet

#endif

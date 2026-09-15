/*
    Copyright 2017 Philippe Grandclement

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Kadath is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kadath.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Modifications (Celephais):
 *   2026-09-15  Added the polar counterpart of Domain_shell_log.
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
namespace Kadath
{
    Val_domain Domain_polar_shell_log::mult_r(const Val_domain& so) const
    {
        Val_domain res(so * get_radius());
        res.base = so.base;
        return res;
    }

    Val_domain Domain_polar_shell_log::der_r(const Val_domain& so) const
    {
        return (so.der_var(1) / alpha / get_radius());
    }

    // With s = log r one has f_r = f_s / r and f_rr = (f_ss - f_s) / r^2, so that
    // grad^2 f = (f_ss + f_s + f_thth + cot(theta) f_th) / r^2
    Val_domain Domain_polar_shell_log::laplacian(const Val_domain& so, int m) const
    {
        Val_domain ders(so.der_var(1) / alpha);
        Val_domain dert(so.der_var(2));
        Val_domain res(
            div_r(div_r(ders.der_var(1) / alpha + ders + dert.der_var(2) + dert.mult_cos_theta().div_sin_theta())));
        if (m != 0)
            res -= m * m * div_r(div_r(so.div_sin_theta().div_sin_theta()));
        return res;
    }

    // Same reduction, for f_rr + f_r / r + f_thth / r^2 = (f_ss + f_thth) / r^2
    Val_domain Domain_polar_shell_log::laplacian2(const Val_domain& so, int m) const
    {
        Val_domain ders(so.der_var(1) / alpha);
        Val_domain dert(so.der_var(2));
        Val_domain res(div_r(div_r(ders.der_var(1) / alpha + dert.der_var(2))));
        if (m != 0)
            res -= m * m * div_r(div_r(so.div_sin_theta().div_sin_theta()));
        return res;
    }

    double Domain_polar_shell_log::integrale(const Val_domain&) const
    {
        KADATH_THROW("Case not yet implemented in Domain_polar_shell_log::integrale");
    }

    double Domain_polar_shell_log::integ_volume(const Val_domain&) const
    {
        KADATH_THROW("Case not yet implemented in Domain_polar_shell_log::integ_volume");
    }
} // namespace Kadath

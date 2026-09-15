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
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
namespace Kadath
{
    // Standard constructor
    Domain_polar_shell_log::Domain_polar_shell_log(int num, int ttype, double rint, double rext, const Point& cr,
                                                   const Dim_array& nbr)
        : Domain_polar_shell(num, ttype, rint, rext, cr, nbr)
    {
        alpha = (log(rext) - log(rint)) / 2.;
        beta = (log(rext) + log(rint)) / 2.;
        assert(nbr.get_ndim() == 2);
        assert(cr.get_ndim() == 2);
        do_coloc();
    }

    // Constructor by copy
    Domain_polar_shell_log::Domain_polar_shell_log(const Domain_polar_shell_log& so) : Domain_polar_shell(so)
    {

        // Update the alpha and beta
        double rint = (beta - alpha);
        double rext = (beta + alpha);
        alpha = (log(rext) - log(rint)) / 2.;
        beta = (log(rext) + log(rint)) / 2.;
    }

    Domain_polar_shell_log::Domain_polar_shell_log(int num, BinarySource& source) : Domain_polar_shell(num, source)
    {
        do_coloc();
    }

    // Destructor
    Domain_polar_shell_log::~Domain_polar_shell_log() {}

    void Domain_polar_shell_log::save(BinarySink& sink) const
    {
        nbr_points.save(sink);
        nbr_coefs.save(sink);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        center.save(sink);
        sink.write<double>(alpha);
        sink.write<double>(beta);
    }

    ostream& Domain_polar_shell_log::print(ostream& o) const
    {
        o << "Polar shell log" << endl;
        o << exp(beta - alpha) << " < r < " << exp(beta + alpha) << endl;
        o << "Center  = " << center << endl;
        o << "Nbr pts = " << nbr_points << endl;
        o << endl;
        return o;
    }

    Val_domain Domain_polar_shell_log::der_normal(const Val_domain& so, int bound) const
    {

        if ((bound != OUTER_BC) && (bound != INNER_BC)) {
            KADATH_THROW("Unknown boundary case in Domain_polar_shell_log::der_normal");
        }
        return der_r(so);
    }

    // Computes the Cartesian coordinates
    void Domain_polar_shell_log::do_absol() const
    {
        for (int i = 0; i < 2; i++)
            assert(coloc[i] != nullptr);
        for (int i = 0; i < 2; i++)
            assert(absol[i] == nullptr);
        for (int i = 0; i < 2; i++) {
            absol[i] = new Val_domain(this);
            absol[i]->allocate_conf();
        }
        Index index(nbr_points);

        do {
            absol[0]->set(index) = exp(alpha * ((*coloc[0])(index(0))) + beta) * sin((*coloc[1])(index(1))) + center(1);
            absol[1]->set(index) = exp(alpha * ((*coloc[0])(index(0))) + beta) * cos((*coloc[1])(index(1))) + center(2);
        } while (index.inc());
    }

    // Computes the radius
    void Domain_polar_shell_log::do_radius() const
    {
        for (int i = 0; i < 2; i++)
            assert(coloc[i] != nullptr);
        assert(radius == nullptr);
        radius = new Val_domain(this);
        radius->allocate_conf();
        Index index(nbr_points);
        do
            radius->set(index) = exp(alpha * ((*coloc[0])(index(0))) + beta);
        while (index.inc());
        // Needed here but not in Domain_polar_shell : mult_r is a configuration
        // space operation for this mapping and so requires the radius to have a base.
        radius->std_base();
    }

    // Computes the Cartesian coordinates
    void Domain_polar_shell_log::do_cart() const
    {
        for (int i = 0; i < 2; i++)
            assert(coloc[i] != nullptr);
        for (int i = 0; i < 2; i++)
            assert(cart[i] == nullptr);
        for (int i = 0; i < 2; i++) {
            cart[i] = new Val_domain(this);
            cart[i]->allocate_conf();
        }
        Index index(nbr_points);

        do {
            cart[0]->set(index) = exp(alpha * ((*coloc[0])(index(0))) + beta) * sin((*coloc[1])(index(1))) + center(1);
            cart[1]->set(index) = exp(alpha * ((*coloc[0])(index(0))) + beta) * cos((*coloc[1])(index(1))) + center(2);
        } while (index.inc());
    }

    // Check if a point is inside this domain
    bool Domain_polar_shell_log::is_in(const Point& xx, double prec) const
    {

        assert(xx.get_ndim() == 2);

        double rho_loc = xx(1) - center(1);
        double z_loc = xx(2) - center(2);
        double air_loc = sqrt(rho_loc * rho_loc + z_loc * z_loc);

        bool res = ((air_loc <= exp(beta + alpha) + prec) && (air_loc >= exp(beta - alpha) - prec)) ? true : false;
        return res;
    }

    const Point Domain_polar_shell_log::absol_to_num(const Point& abs) const
    {

        assert(is_in(abs));
        Point num(2);

        double rho_loc = fabs(abs(1) - center(1));
        double z_loc = abs(2) - center(2);
        double air = sqrt(rho_loc * rho_loc + z_loc * z_loc);
        num.set(1) = (log(air) - beta) / alpha;

        if (rho_loc == 0) {
            // On the axis ?
            num.set(2) = (z_loc >= 0) ? 0 : M_PI;
        } else {
            num.set(2) = atan(rho_loc / z_loc);
        }

        if (num(2) < 0)
            num.set(2) = M_PI + num(2);

        return num;
    }

    void Domain_polar_shell_log::do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const
    {

        Val_domain dr(*der_var[0] / alpha / get_radius());
        dr.set_base() = der_var[0]->get_base();
        Val_domain dtsr(*der_var[1] / get_radius());
        dtsr.set_base() = der_var[1]->get_base();

        // D / drho
        der_abs[0] = new Val_domain(dr.mult_sin_theta() + dtsr.mult_cos_theta());
        // d/dz :
        der_abs[1] = new Val_domain(dr.mult_cos_theta() - dtsr.mult_sin_theta());
    }

} // namespace Kadath

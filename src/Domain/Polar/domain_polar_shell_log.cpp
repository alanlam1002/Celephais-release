/*
 * domain_polar_shell_log.cpp -- polar shell with log r = alpha x + beta.
 *
 * The polar analogue of Domain_shell_log (src/Domain/Spheric/), written against
 * that file so the two stay recognisably the same construction.  See
 * include/For_Kadath/Domain/polar.hpp for why it exists and for the ONE place
 * the two families differ: Domain_shell has no laplacian() of its own, so the
 * spheric log shell inherits Domain::laplacian and gets it free from
 * do_der_abs_from_der_var; Domain_polar_shell DOES define laplacian(), so it
 * must be overridden here or it silently computes the linear-mapping answer.
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

namespace Kadath
{

Domain_polar_shell_log::Domain_polar_shell_log(int num, int ttype, double rint, double rext,
                                               const Point& cr, const Dim_array& nbr)
    : Domain_polar_shell(num, ttype, rint, rext, cr, nbr)
{
    // The base has set alpha, beta for r = alpha x + beta.  Reinterpret them
    // for ln r, then do_coloc() to drop everything the base cached off the old
    // mapping (del_deriv() inside it clears the radius).
    alpha = (log(rext) - log(rint)) / 2.;
    beta = (log(rext) + log(rint)) / 2.;
    assert(nbr.get_ndim() == 2);
    assert(cr.get_ndim() == 2);
    do_coloc();
}

Domain_polar_shell_log::Domain_polar_shell_log(const Domain_polar_shell_log& so)
    : Domain_polar_shell(so)
{
    // Domain_polar_shell's copy constructor re-derives alpha and beta as though
    // they were linear, so undo that: the values it copied ARE the log ones.
    const double rint = (beta - alpha);
    const double rext = (beta + alpha);
    alpha = (log(rext) - log(rint)) / 2.;
    beta = (log(rext) + log(rint)) / 2.;
}

Domain_polar_shell_log::Domain_polar_shell_log(int num, BinarySource& source)
    : Domain_polar_shell(num, source)
{
    // save() wrote the log alpha and beta, so the base read them back verbatim;
    // only the cached geometry has to be rebuilt.
    do_coloc();
}

Domain_polar_shell_log::~Domain_polar_shell_log() {}

void Domain_polar_shell_log::save(BinarySink& sink) const
{
    nbr_points.save(sink);
    nbr_coefs.save(sink);
    sink.write<int>(ndim);
    sink.write<int>(type_base);
    get_center().save(sink);
    sink.write<double>(alpha);
    sink.write<double>(beta);
}

ostream& Domain_polar_shell_log::print(ostream& o) const
{
    o << "Polar shell log" << endl;
    o << exp(beta - alpha) << " < r < " << exp(beta + alpha) << endl;
    o << "Center  = " << get_center() << endl;
    o << "Nbr pts = " << nbr_points << endl;
    o << endl;
    return o;
}

// ---------------------------------------------------------------- geometry --

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
    // ⚠ Domain_polar_shell::do_radius does NOT do this, because its mult_r is a
    // coefficient-space operation (alpha T(x) + beta) that never touches the
    // radius.  Ours multiplies by it in configuration space, as the spheric log
    // shell does, so the radius needs a basis or the first mult_r throws
    // "Base not defined in Val_domain::coef".  Domain_shell_log::do_radius has
    // this line; copying the POLAR template rather than the SPHERIC one is how
    // it went missing.
    radius->std_base();
}

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
    const Point cr(get_center());
    Index index(nbr_points);
    do {
        const double r = exp(alpha * ((*coloc[0])(index(0))) + beta);
        absol[0]->set(index) = r * sin((*coloc[1])(index(1))) + cr(1);
        absol[1]->set(index) = r * cos((*coloc[1])(index(1))) + cr(2);
    } while (index.inc());
}

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
    const Point cr(get_center());
    Index index(nbr_points);
    do {
        const double r = exp(alpha * ((*coloc[0])(index(0))) + beta);
        cart[0]->set(index) = r * sin((*coloc[1])(index(1))) + cr(1);
        cart[1]->set(index) = r * cos((*coloc[1])(index(1))) + cr(2);
    } while (index.inc());
}

bool Domain_polar_shell_log::is_in(const Point& xx, double prec) const
{
    assert(xx.get_ndim() == 2);
    const Point cr(get_center());
    const double rho_loc = xx(1) - cr(1);
    const double z_loc = xx(2) - cr(2);
    const double air_loc = sqrt(rho_loc * rho_loc + z_loc * z_loc);
    return ((air_loc <= exp(beta + alpha) + prec) && (air_loc >= exp(beta - alpha) - prec));
}

const Point Domain_polar_shell_log::absol_to_num(const Point& abs) const
{
    assert(is_in(abs));
    Point num(2);
    const Point cr(get_center());
    const double rho_loc = fabs(abs(1) - cr(1));
    const double z_loc = abs(2) - cr(2);
    const double air = sqrt(rho_loc * rho_loc + z_loc * z_loc);
    num.set(1) = (log(air) - beta) / alpha;       // the ONE line that differs

    if (rho_loc == 0)
        num.set(2) = (z_loc >= 0) ? 0 : M_PI;
    else
        num.set(2) = atan(rho_loc / z_loc);
    if (num(2) < 0)
        num.set(2) = M_PI + num(2);
    return num;
}

// ------------------------------------------------------------- derivatives --

void Domain_polar_shell_log::do_der_abs_from_der_var(Val_domain** der_var,
                                                     Val_domain** der_abs) const
{
    // d/dr = (1/(alpha r)) d/dx, against the linear shell's (1/alpha) d/dx.
    // Dividing by get_radius() is a configuration-space operation and loses the
    // basis, so it is restored -- exactly as the linear shell already does for
    // its dtsr.
    Val_domain dr(*der_var[0] / alpha / get_radius());
    dr.set_base() = der_var[0]->get_base();
    Val_domain dtsr(*der_var[1] / get_radius());
    dtsr.set_base() = der_var[1]->get_base();

    // d/drho
    der_abs[0] = new Val_domain(dr.mult_sin_theta() + dtsr.mult_cos_theta());
    // d/dz
    der_abs[1] = new Val_domain(dr.mult_cos_theta() - dtsr.mult_sin_theta());
}

Val_domain Domain_polar_shell_log::der_r(const Val_domain& so) const
{
    return (so.der_var(1) / alpha / get_radius());
}

Val_domain Domain_polar_shell_log::der_normal(const Val_domain& so, int bound) const
{
    if ((bound != OUTER_BC) && (bound != INNER_BC))
        KADATH_THROW("Unknown boundary case in Domain_polar_shell_log::der_normal");
    return der_r(so);
}

Val_domain Domain_polar_shell_log::mult_r(const Val_domain& so) const
{
    // The linear shell does this in COEFFICIENT space, as alpha*T(x) + beta,
    // which is only valid because r is affine in x.  Here it is not.
    Val_domain res(so * get_radius());
    res.set_base() = so.get_base();
    return res;
}

/**
 * THE ONE THE SPHERIC LOG SHELL DID NOT NEED.
 *
 * With s = ln r and f_s = der_var(1)/alpha,
 *     f_r  = f_s / r
 *     f_rr = (f_ss - f_s) / r^2
 * so
 *     grad^2 f = f_rr + (2/r) f_r + (1/r^2)(f_thth + cot(theta) f_th)
 *              = (1/r^2) [ f_ss + f_s + f_thth + cot(theta) f_th ]
 * -- constant coefficient in s apart from the overall r^-2, which is where the
 * whole accuracy argument comes from.  The m != 0 term is carried over from the
 * linear shell unchanged: it is algebraic in theta, not in r.
 */
Val_domain Domain_polar_shell_log::laplacian(const Val_domain& so, int m) const
{
    Val_domain fs(so.der_var(1) / alpha);
    Val_domain fss(fs.der_var(1) / alpha);
    Val_domain dert(so.der_var(2));
    Val_domain res(div_r(div_r(fss + fs + dert.der_var(2)
                               + dert.mult_cos_theta().div_sin_theta())));
    if (m != 0)
        res -= m * m * div_r(div_r(so.div_sin_theta().div_sin_theta()));
    return res;
}

/** f_rr + (1/r) f_r + (1/r^2) f_thth = (1/r^2)[ f_ss + f_thth ]. */
Val_domain Domain_polar_shell_log::laplacian2(const Val_domain& so, int m) const
{
    Val_domain fs(so.der_var(1) / alpha);
    Val_domain fss(fs.der_var(1) / alpha);
    Val_domain dert(so.der_var(2));
    Val_domain res(div_r(div_r(fss + dert.der_var(2))));
    if (m != 0)
        res -= m * m * div_r(div_r(so.div_sin_theta().div_sin_theta()));
    return res;
}

// --------------------------------------------------------------- refusals --

double Domain_polar_shell_log::integrale(const Val_domain&) const
{
    KADATH_THROW("Domain_polar_shell_log::integrale is not implemented: the "
                 "inherited quadrature assumes dr = alpha dx, and here "
                 "dr = alpha r dx.  Refusing rather than returning a wrong "
                 "volume integral.");
}

double Domain_polar_shell_log::integ_volume(const Val_domain&) const
{
    KADATH_THROW("Domain_polar_shell_log::integ_volume is not implemented: see "
                 "integrale.");
}

} // namespace Kadath

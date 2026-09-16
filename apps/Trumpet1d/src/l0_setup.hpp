#pragma once
/*
 * l0_setup.hpp -- the l=0 row operators, shared by the T2.1 evaluation app
 * (l0_system_main.cpp) and the T2.2 solver app (l0_solve_main.cpp).
 *
 * WHY THIS IS A HEADER AND NOT A CONVENIENCE.  T2.1 gated an operator: five
 * add_def rows over 39 bounded add_cst coefficient fields, verified against a
 * manufactured solution and an independent Python evaluation.  T2.2 solves a
 * BVP built on that operator.  If the two apps assembled the rows separately,
 * the T2.1 gate would no longer certify what T2.2 solves.  Sharing the
 * assembly is what keeps the gate meaningful.
 *
 * Coefficient naming avoids '_' entirely: a trailing _<char> declares a tensor
 * index in Kadath's string language (cf. "gradN_i = grad(N)"), so names are
 * c<row><jet> with row in {K, XR, XT, H, M} and jet in {U,Up,Upp,...,j2}.
 *
 * Row scaling: row n has already been divided by r^p_n on the Python side so
 * every coefficient is bounded at r = infinity.  Scaling a row by a nonzero
 * function does not move its zero set, so neither the solve nor a constraint
 * pin on EH/EM is affected; it only makes the operator representable on the
 * compactified domain.
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Domain/oned.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include "space/space_oned_trumpet.hpp"
#include "src/table_io.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace Trumpet
{

/** The five rows: three evolution rows then the two constraints. */
inline const char* const* rows()
{
    static const char* const R[5] = {"E_K", "E_chi_rr", "E_chi_tt", "E_H", "E_Mr"};
    return R;
}
/** Short code used in coefficient field names. */
inline const char* const* row_codes()
{
    static const char* const C[5] = {"K", "XR", "XT", "H", "M"};
    return C;
}
/** add_def name of each row. */
inline const char* const* row_defs()
{
    static const char* const D[5] = {"EK", "EXR", "EXT", "EH", "EM"};
    return D;
}

/** Split a jet name into (field stem, derivative order): "Sp" -> ("S", 1). */
inline std::pair<std::string, int> split_jet(const std::string& jet)
{
    if (jet.size() > 2 && jet.compare(jet.size() - 2, 2, "pp") == 0)
        return {jet.substr(0, jet.size() - 2), 2};
    if (jet.size() > 1 && jet.back() == 'p')
        return {jet.substr(0, jet.size() - 1), 1};
    return {jet, 0};
}

/**
 * FIRST-ORDER REDUCTION (round 91; research round 279 asked for it MEASURED,
 * not decided).
 *
 * When on, every first radial derivative is a separate UNKNOWN `<f>d` instead
 * of `dr(<f>)`, and every second derivative is `dr(<f>d)`.  No row then takes
 * two radial derivatives of a spectral interpolant, which round 90 measured as
 * the binding accuracy constraint: the Laplacian's roundoff floor RISES with
 * radial resolution while the field stays resolved to 6e-16.  The price is
 * twice the unknowns -- expensive against a hard 33-point ceiling -- and
 * whatever it does to cond(A_c), which is why both are reported.
 *
 * This is a property of the whole discretisation, not of one call site, so it
 * is a module-level switch.  deriv() is the single point that every row, every
 * matching and every boundary condition passes through (phys_expr routes here
 * too), and a per-call-site flag would guarantee an inconsistency somewhere.
 */
inline bool& first_order_mode()
{
    static bool on = false;
    return on;
}

/** The three derivative unknowns, in the order of `fields`. */
inline std::string dvar(const std::string& f) { return f + "d"; }

/** Derivative order `n` of field `f` as a Kadath expression string. */
inline std::string deriv(const std::string& f, int n)
{
    if (!first_order_mode())
        return n == 0 ? f : (n == 1 ? "dr(" + f + ")" : "ddr(" + f + ")");
    return n == 0 ? f : (n == 1 ? dvar(f) : "dr(" + dvar(f) + ")");
}

/**
 * Jet name -> the string that applies it to a field.
 *
 * Suffix-driven, so it is FIELD-SET AGNOSTIC: it serves (U,Q,G) and the
 * round-11 (S,T,G) route with no per-set code.  `j2` is the source amplitude.
 */
inline std::string apply_jet(const std::string& jet, const std::string& f)
{
    if (jet == "j2")
        return "jsrc";
    return deriv(f, split_jet(jet).second);
}

/** Which of `fields` this jet belongs to; "" for the source column. */
inline std::string field_of(const std::string& jet,
                            const std::vector<std::string>& fields)
{
    const std::string stem = split_jet(jet).first;
    for (const auto& f : fields)
        if (stem == f)
            return f;
    return "";
}

/**
 * THE PHYSICAL VARIABLES (U,Q,G), expressed in whatever the table's fields are.
 *
 * Every gate, oracle, inner row and tail in this project is defined in (U,Q,G),
 * and round 11 asks for the gates to be ported AS-IS.  The (S,T,G) route is the
 * linear change S = 4U + Q, T = U, so the inverse
 *
 *     U = T,   Q = S - 4 T,   G = G
 *
 * is exact with constant coefficients, and every physical condition can be
 * written in the native fields without re-deriving anything.  For the (U,Q,G)
 * table this is the identity.
 */
inline std::string phys_expr(const std::vector<std::string>& fields,
                             const std::string& phys, int order)
{
    const bool stg = (fields.size() == 3 && fields[0] == "S" && fields[1] == "T");
    if (!stg)
        return deriv(phys, order);
    if (phys == "U")
        return deriv("T", order);
    if (phys == "G")
        return deriv("G", order);
    if (phys == "Q")
        return "(" + deriv("S", order) + " - 4 * " + deriv("T", order) + ")";
    throw std::runtime_error("phys_expr: unknown physical field " + phys);
}

inline void emit(const std::string& k, double v)
{
    // Built whole, then written in ONE << and flushed.  Gates gather these logs
    // with 2>&1, and std::cerr is unbuffered while std::cout is not: a warning
    // written from Kadath or from this app mid-flush lands INSIDE a RESULT line
    // and silently destroys it (seen: "RESULT WARNING: ...\nBC1_dWdr_rel 2.7e-13",
    // which cost the gate its BC1 assertion with a bare KeyError).  One atomic
    // write plus a flush closes the window.
    std::ostringstream os;
    os << "RESULT " << k << " " << std::setprecision(17) << v << "\n";
    std::cout << os.str() << std::flush;
}

/**
 * Owns the space, the imported backbone, the 39 coefficient fields and the
 * three unknowns, and registers the five rows with a System_of_eqs.
 *
 * Lifetime contract: a System_of_eqs borrows references to the Scalars passed
 * to add_var/add_cst, so an L0Model must outlive every system built from it.
 * Kadath::Space::~Space() is protected, so the space is held by VALUE as the
 * concrete type -- it cannot live in a unique_ptr<Space>.
 */
/**
 * Inner-BC left-hand side, from a SPEC v2 bc row.
 *
 * Shared between the 1-D and 2-D solver apps (round 96): the rows are quoted on
 * the PHYSICAL jet and phys_expr transforms them rather than re-deriving, so a
 * second copy in the 2-D app would be a second chance to get the transform
 * wrong.
 */
inline std::string bc_lhs(const TrumpetIO::BcRow& r, const std::string& prefix,
                          Kadath::System_of_eqs& syst,
                          const std::vector<std::string>& fields)
{
    // SPEC v2's rows are quoted on the PHYSICAL jet (U, dr(U), Q, dr(Q), G,
    // dr(G)) and round 11 asks for them to be transformed, not re-derived.
    // phys_expr does exactly that, and is the identity for a (U,Q,G) table.
    auto P = [&](const char* f, int o) { return Trumpet::phys_expr(fields, f, o); };
    if (!r.general)
        return (r.kind == "der1") ? P(r.field.c_str(), 1)
             : (r.field == "4U+Q+G"
                    ? "4 * " + P("U", 0) + " + " + P("Q", 0) + " + " + P("G", 0)
                    : P(r.field.c_str(), 0));
    const std::string JETSTR[6] = {P("U", 0), P("U", 1), P("Q", 0), P("Q", 1),
                                   P("G", 0), P("G", 1)};
    static const char* SUF[6] = {"cU", "cUp", "cQ", "cQp", "cG", "cGp"};
    std::string out;
    for (int k = 0; k < 6; k++) {
        if (r.coef[k] == 0.0)
            continue;                       // a genuinely absent slot (Q_W is 0)
        const std::string cn = prefix + SUF[k];
        syst.add_cst(cn.c_str(), r.coef[k]);
        if (!out.empty())
            out += " + ";
        out += cn + " * " + JETSTR[k];
    }
    if (out.empty())
        throw std::runtime_error("bc row " + prefix + " is identically zero");
    return out;
}

/**
 * ROUND 95/96 -- templated on the space so ONE row assembly serves 1-D and 2-D.
 *
 * The header's own reason for existing is that the T2.1 gate certifies the
 * operator T2.2 solves; if the 2-D port assembled the rows separately, the gate
 * would stop certifying what the 2-D app solves and the whole A0 argument would
 * lose its reference.  Everything dimension-dependent is confined to make_res()
 * and fill(); the row assembly is strings and is dimension-blind.
 *
 * `L0Model` below is the 1-D instantiation, unchanged for every existing caller.
 */
template <class SpaceT, int NDIM>
class L0ModelT
{
public:
    /**
     * @param rownorm  divide every coefficient of row n, at each collocation
     *   point, by max_j |c_{n,j}| there.  Playbook rule 6: the rows carry a
     *   r^{-p_n} scaling with p up to 10, so between the excision radius and
     *   spatial infinity their magnitudes span ~6 orders BEFORE the operator
     *   is even applied, and the Jacobian inherits that spread.  Dividing a
     *   row pointwise by a positive function does not move its zero set, so
     *   neither the solution nor a pin on EH/EM changes.
     */
    L0ModelT(const std::string& backbone_path, const std::string& coefs_path,
             bool rownorm = false, int ntheta = 1)
        : bt(TrumpetIO::read_table(backbone_path)),
          ct(TrumpetIO::read_coefs(coefs_path, bt.doms)),
          ndom(static_cast<int>(bt.doms.size())),
          ntheta_(NDIM == 1 ? 1 : ntheta),
          space(CHEB_TYPE, make_res(bt, NDIM == 1 ? 1 : ntheta), make_bounds(bt)),
          Wf(space), Rrf(space), oorf(space), iR(space),
          U(space), Q(space), G(space), Ud(space), Qd(space), Gd(space)
    {
        if (bt.mode != "excised")
            throw std::runtime_error("the l=0 system requires the excised layout, got "
                                     + bt.mode);
        fill(Wf, [&](int d, int i) { return bt.pts[d][i].W; });
        fill(Rrf, [&](int d, int i) { return bt.pts[d][i].Rr; });
        fill(oorf, [&](int d, int i) { return bt.pts[d][i].oor; });
        // 1/R = oor/Rr, bounded everywhere (0 at spatial infinity)
        for (int d = 0; d < ndom; d++)
            iR.set_domain(d) = oorf(d) / Rrf(d);
        iR.std_base();

        set_fields_zero();

        // Per-row, per-point normaliser: max_j |c_{n,j}|, or 1 if not asked for.
        norm.assign(5, std::vector<std::vector<double>>());
        for (int n = 0; n < 5; n++) {
            norm[n].resize(ndom);
            for (int d = 0; d < ndom; d++)
                norm[n][d].assign(bt.doms[d].nbr, 1.0);
            if (!rownorm)
                continue;
            for (int d = 0; d < ndom; d++)
                for (int i = 0; i < bt.doms[d].nbr; i++) {
                    double mx = 0.0;
                    for (const auto& jet : ct.jets[rows()[n]])
                        mx = std::max(mx, std::fabs(
                            ct.v.at(std::make_pair(std::string(rows()[n]), jet))[d][i]));
                    if (!(mx > 0.0))
                        throw std::runtime_error(
                            std::string("row ") + rows()[n] + " has ALL coefficients "
                            "zero at domain " + std::to_string(d) + " point "
                            + std::to_string(i) + ": the row is degenerate there and "
                            "cannot be normalised");
                    norm[n][d][i] = mx;
                }
        }

        // the 39 coefficient fields, in a deque so the Scalars keep stable
        // addresses for the lifetime of any system (add_cst borrows a reference)
        for (int n = 0; n < 5; n++) {
            for (const auto& jet : ct.jets[rows()[n]]) {
                const auto& g = ct.v.at(std::make_pair(std::string(rows()[n]), jet));
                cst.emplace_back(space);
                const int nn = n;
                fill(cst.back(), [&, nn](int d, int i) { return g[d][i] / norm[nn][d][i]; });
                nameof[std::string(rows()[n]) + "/" + jet] =
                    std::string("c") + row_codes()[n] + jet;
            }
        }
    }

    /** max_j |c_{n,j}| at (d,i); 1 everywhere unless row normalisation is on. */
    double row_norm(int n, int d, int i) const { return norm[n][d][i]; }

    /** Ratio of the largest to the smallest row normaliser -- the spread the
     *  Jacobian would otherwise inherit. */
    double norm_spread(int n) const
    {
        double lo = 1e300, hi = 0.0;
        for (int d = 0; d < ndom; d++)
            for (int i = 0; i < bt.doms[d].nbr; i++) {
                lo = std::min(lo, norm[n][d][i]);
                hi = std::max(hi, norm[n][d][i]);
            }
        return hi / std::max(lo, 1e-300);
    }

    L0ModelT(const L0ModelT&) = delete;
    L0ModelT& operator=(const L0ModelT&) = delete;

    /** Point count of domain d. */
    int nbr(int d) const { return bt.doms[d].nbr; }
    int nb_domains() const { return ndom; }
    const TrumpetIO::Table& table() const { return bt; }
    const TrumpetIO::CoefTable& coefs() const { return ct; }
    SpaceT& sp() { return space; }
    int ntheta() const { return ntheta_; }

    /** Write `get(d,i)` into every collocation point of a Scalar on this space. */
    template <class F>
    void fill(Kadath::Scalar& f, F get) const
    {
        for (int d = 0; d < ndom; d++) {
            Kadath::Val_domain& vd = f.set_domain(d);
            vd.allocate_conf();
            // ⚠ The 1-D version set only idx(0) and wrote once per radial
            // point.  In 2-D that fills the theta = 0 line and leaves the rest
            // of the Val_domain uninitialised, which is a silent wrong answer
            // rather than a crash.  The backbone and the coefficient fields are
            // theta-independent, so every theta gets the same radial value.
            Kadath::Index idx(space.get_domain(d)->get_nbr_points());
            do {
                vd.set(idx) = get(d, idx(0));
            } while (idx.inc());
        }
        f.std_base();
    }

    void set_fields_zero()
    {
        for (int d = 0; d < ndom; d++) {
            U.set_domain(d) = 0.0 * Wf(d);
            Q.set_domain(d) = 0.0 * Wf(d);
            G.set_domain(d) = 0.0 * Wf(d);
            Ud.set_domain(d) = 0.0 * Wf(d);
            Qd.set_domain(d) = 0.0 * Wf(d);
            Gd.set_domain(d) = 0.0 * Wf(d);
        }
        U.std_base();
        Q.std_base();
        G.std_base();
        Ud.std_base();
        Qd.std_base();
        Gd.std_base();
    }

    /**
     * The exact mass mode: U = (1-W)/2, Q = 0, G = F_M - 2 with
     * F_M = -(1/R - (27/8) M^3/R^4)/W  (the Phase-1 G6 form).
     * An exact solution of the full five-row system at j2 = 0 and c1 = 2.
     */
    void set_fields_massmode()
    {
        const double M = bt.M;
        // In (S,T,G) the same solution is S = 4U + Q = 2(1-W) and T = U, since
        // the mass mode has Q identically zero.  Exact either way.
        for (int d = 0; d < ndom; d++) {
            const Kadath::Val_domain u = (1.0 - Wf(d)) * 0.5;
            const Kadath::Val_domain g =
                -(iR(d) - (27.0 / 8.0) * std::pow(M, 3) * Kadath::pow(iR(d), 4)) / Wf(d)
                - 2.0;
            if (is_stg()) {
                U.set_domain(d) = 4.0 * u;          // S
                Q.set_domain(d) = u;                // T
            } else {
                U.set_domain(d) = u;
                Q.set_domain(d) = 0.0 * Wf(d);
            }
            G.set_domain(d) = g;
        }
        U.std_base();
        Q.std_base();
        G.std_base();
    }

    /// Is this model in the round-11 (S,T,G) variables?
    bool is_stg() const
    {
        return ct.fields.size() == 3 && ct.fields[0] == "S" && ct.fields[1] == "T";
    }

    /** The physical (U,Q,G) fields, whatever the native ones are. */
    Kadath::Val_domain Uphys(int d) const { return is_stg() ? Q(d) : U(d); }
    Kadath::Val_domain Qphys(int d) const
    {
        return is_stg() ? (U(d) - 4.0 * Q(d)) : Q(d);
    }
    Kadath::Val_domain Gphys(int d) const { return G(d); }

    /**
     * Coefficient of a PHYSICAL jet ("Upp", "Qp", ...) in row n, recovered from
     * a table written in the native fields.  With S = 4U+Q, T = U the stored
     * coefficients satisfy  d_S. = c_Q.  and  d_T. = c_U. - 4 c_Q. , so
     *     c_U. = d_T. + 4 d_S. ,   c_Q. = d_S. ,   c_G. = d_G.
     * Verified against the (U,Q,G) table at the horizon interface: exact to the
     * last bit, and it returns kappa = 2 in both variable sets.
     */
    double phys_coef(const std::string& row, const std::string& physjet,
                     int d, int i) const
    {
        const auto sj = split_jet(physjet);
        const int od = sj.second;
        auto at = [&](const char* stem) -> double {
            const std::string jn = std::string(stem)
                                   + (od == 0 ? "" : (od == 1 ? "p" : "pp"));
            const auto it = ct.v.find(std::make_pair(row, jn));
            return it == ct.v.end() ? 0.0 : it->second[d][i];
        };
        if (!is_stg())
            return at(sj.first.c_str());
        if (sj.first == "U")
            return at("T") + 4.0 * at("S");
        if (sj.first == "Q")
            return at("S");
        return at("G");
    }

    /**
     * add_var the unknowns, add_cst the source amplitude and the 39
     * coefficient fields, add_def the five rows.  Returns the row strings for
     * logging.  No add_eq_* is issued here: the equation set is the caller's
     * business (T2.1 adds none, T2.2 adds six conditions).
     */
    std::vector<std::string> register_rows(Kadath::System_of_eqs& syst, double j2)
    {
        syst.add_var(ct.fields[0].c_str(), U);
        syst.add_var(ct.fields[1].c_str(), Q);
        syst.add_var(ct.fields[2].c_str(), G);
        if (first_order_mode()) {
            Kadath::Scalar* const D[3] = {&Ud, &Qd, &Gd};
            for (int i = 0; i < 3; i++) {
                syst.add_var(dvar(ct.fields[i]).c_str(), *D[i]);
                // The DEFINING rows.  Written with a literal dr(), not deriv():
                // deriv(f,1) is now the unknown itself, and using it here would
                // define the unknown to be equal to itself.
                dvardefs.push_back("D" + ct.fields[i] + " = " + dvar(ct.fields[i])
                                   + " - dr(" + ct.fields[i] + ")");
                syst.add_def(dvardefs.back().c_str());
            }
        }
        syst.add_cst("jsrc", j2);

        std::size_t k = 0;
        for (int n = 0; n < 5; n++)
            for (const auto& jet : ct.jets[rows()[n]])
                syst.add_cst(nameof.at(std::string(rows()[n]) + "/" + jet).c_str(),
                             cst[k++]);

        std::vector<std::string> defs;
        for (int n = 0; n < 5; n++) {
            std::string rhs;
            for (const auto& jet : ct.jets[rows()[n]]) {
                if (!rhs.empty())
                    rhs += " + ";
                rhs += nameof.at(std::string(rows()[n]) + "/" + jet) + " * "
                       + apply_jet(jet, field_of(jet, ct.fields));
            }
            defs.push_back(std::string(row_defs()[n]) + " = " + rhs);
            syst.add_def(defs.back().c_str());
        }
        return defs;
    }

    /**
     * FAR-FIELD LOG ENRICHMENT (round-6 option (a)).
     *
     * The constraints source a ln(r) far field that no polynomial basis in 1/r
     * represents, so the compactified domain's ansatz is enriched with
     * L_k(r) = ln(r)/r^k, amplitude an ordinary unknown.  Three points of care:
     *
     * ANALYTIC DERIVATIVES.  Filling one Scalar with L and asking Kadath for
     * dr() of it would differentiate the INTERPOLANT, carrying exactly the
     * representation error the enrichment exists to remove.  L, L' and L'' are
     * therefore three separate add_cst fields, each analytic at the collocation
     * points, so every row's log column is an exact pointwise combination of
     * tabulated numbers.
     *
     * BUILT FROM 1/r, NOT r.  The compactified domain's last point is r = inf;
     * oor = 1/r is 0 there and finite everywhere, so the profiles are written in
     * oor and set to exactly 0 when oor == 0.
     *
     * SUPPORTED ON THE LAST DOMAIN ONLY.  ln(r)/r reads -12.0 at r_in = 0.155;
     * a globally supported log would corrupt the inner rows, whose census says
     * the local family at the throat is analytic.  Zero inside means the SAME
     * compound expressions can be used on every domain and interface without
     * special-casing anything.
     *
     * Returns the registered names {L, L', L''} for the caller to build rows.
     */
    std::array<std::string, 3> enrich_log(Kadath::System_of_eqs& syst, int k,
                                          const std::string& tagname)
    {
        const int dlast = ndom - 1;
        auto val = [&](int order, int d, int i) -> double {
            if (d != dlast)
                return 0.0;
            const double u = bt.pts[d][i].oor;          // u = 1/r
            if (u <= 0.0)
                return 0.0;                             // r = infinity
            const double lr = -std::log(u);             // ln r
            const double uk = std::pow(u, k);
            switch (order) {
                case 0: return lr * uk;                         // ln(r)/r^k
                case 1: return uk * u * (1.0 - k * lr);         // d/dr
                default: return uk * u * u * (k * (k + 1) * lr - (2 * k + 1));
            }
        };
        std::array<std::string, 3> nm{tagname, tagname + "p", tagname + "pp"};
        for (int order = 0; order < 3; order++) {
            logprof.emplace_back(space);
            fill(logprof.back(), [&](int d, int i) { return val(order, d, i); });
            syst.add_cst(nm[order].c_str(), logprof.back());
        }
        return nm;
    }

    /**
     * The row operator applied to a log profile in ONE field direction: the same
     * coefficient-times-jet sum register_rows() builds, with (L, L', L'') in
     * place of (f, dr(f), ddr(f)) and the j2 column dropped -- the source does
     * not multiply the amplitude.  Empty when the row has no jets in that field.
     */
    std::string log_row(int n, const std::string& field,
                        const std::array<std::string, 3>& L) const
    {
        std::string rhs;
        for (const auto& jet : ct.jets.at(rows()[n])) {
            if (jet == "j2" || field_of(jet, ct.fields) != field)
                continue;
            const int order = (jet.size() == 1) ? 0 : int(jet.size()) - 1;
            if (!rhs.empty())
                rhs += " + ";
            rhs += nameof.at(std::string(rows()[n]) + "/" + jet) + " * " + L[order];
        }
        return rhs;
    }

    /** Header lines describing the imported operator. */
    void print_banner() const
    {
        std::cout << "# l=0 system  backbone=" << bt.tag << "  coefs=" << ct.tag
                  << "  ndom=" << ndom << "  ncoef=" << ct.ncoef << "\n";
        std::cout << "# points per domain:";
        for (int d = 0; d < ndom; d++)
            std::cout << " " << bt.doms[d].nbr;
        std::cout << "\n";
        for (int n = 0; n < 5; n++) {
            std::cout << "# row " << rows()[n] << " scaled by r^-"
                      << ct.scale.at(rows()[n]) << "  jets:";
            for (const auto& j : ct.jets.at(rows()[n]))
                std::cout << " " << j;
            std::cout << "\n";
        }
    }

    // public so the apps can read/probe them directly
    TrumpetIO::Table bt;
    TrumpetIO::CoefTable ct;
    int ndom;
    int ntheta_ = 1;
    SpaceT space;
    Kadath::Scalar Wf, Rrf, oorf, iR;
    Kadath::Scalar U, Q, G;
    /// First-order-reduction unknowns; registered only when
    /// first_order_mode() is on, but always constructed so the
    /// System_of_eqs references they are handed stay valid.
    Kadath::Scalar Ud, Qd, Gd;
    /// Names of the first-order defining rows, kept alive for add_def.
    std::vector<std::string> dvardefs;
    /// Analytic log profiles for the far-field enrichment; see enrich_log().
    std::vector<Kadath::Scalar> logprof;

private:
    static std::vector<Kadath::Dim_array> make_res(const TrumpetIO::Table& t, int nt)
    {
        std::vector<Kadath::Dim_array> res;
        for (const auto& d : t.doms) {
            Kadath::Dim_array n(NDIM);
            n.set(0) = d.nbr;
            if (NDIM > 1)
                n.set(1) = nt;
            res.push_back(n);
        }
        return res;
    }
    static std::vector<double> make_bounds(const TrumpetIO::Table& t)
    {
        std::vector<double> b;
        for (const auto& d : t.doms)
            b.push_back(d.r_int);
        return b;
    }

    std::deque<Kadath::Scalar> cst;
    std::map<std::string, std::string> nameof;   // "row/jet" -> add_cst name
    std::vector<std::vector<std::vector<double>>> norm;   // [row][domain][point]
};

/** The 1-D instantiation: unchanged for every existing caller. */
using L0Model = L0ModelT<Trumpet::Space_oned_trumpet, 1>;

// ------------------------------------------------------------------------
// THE HORIZON COMPATIBILITY ROW.
//
// Shared because it had been transcribed three times -- the 1-D --additive
// branch, the 1-D --horizon-order branch and the 2-D port -- and the copies
// diverged.  Round 100: the 2-D copy carried the mechanism and dropped the
// normaliser correction, and the divergence was invisible for six rounds
// because each copy passed its own checks.  Duplicated construction that has
// diverged once will diverge again, and it is only visible from outside.
// ------------------------------------------------------------------------

/** What the compatibility row needs, measured at the interface. */
struct HorizonCompat
{
    int dom = -1;             ///< Domain whose OUTER face carries the degeneracy.
    double kappa = 0.;        ///< \f$ E^{tt}_{\chi,U''} / E^{K}_{U''} \f$ near it.
    double norm_ratio = 1.;   ///< \f$ N_K / N_T \f$ at the face where the row lands.
    double kappa_eff = 0.;    ///< The constant to REGISTER: kappa * norm_ratio.
    std::string error;        ///< Empty on success; the whole message otherwise.
    explicit operator bool() const { return error.empty(); }
};

/**
 * Locate the interface whose OUTER face is \f$ r(2M) \f$.
 *
 * Found, never hardcoded: it is the face where E_K's only second-derivative
 * coefficient vanishes against the grid maximum, so a table without this
 * structure yields -1 and the caller fails loudly instead of acting on the
 * wrong point.  The degenerate direction is U'' whatever the table's own
 * fields are, and phys_coef reconstructs it, so this finds the same interface
 * in (U,Q,G) and in (S,T,G) alike.
 */
template <class Model>
int locate_horizon_interface(const Model& m, int dlast)
{
    auto kUpp = [&](int d, int i) { return m.phys_coef("E_K", "Upp", d, i); };
    double mx = 0.;
    for (int d = 0; d <= dlast; d++)
        for (int i = 0; i < m.nbr(d); i++)
            mx = std::max(mx, std::fabs(kUpp(d, i)));
    for (int d = 0; d < dlast; d++)
        if (std::fabs(kUpp(d, m.nbr(d) - 1)) < 1e-12 * mx)
            return d;
    return -1;
}

/**
 * Measure kappa at the interface and convert it to the registered constant.
 *
 * ⚠ ROW NORMALISATION CHANGES THE CONSTANT, and this is what the 2-D port got
 * wrong.  With row normalisation on, the registered defs are \c EK = E_K/N_K
 * and \c EXT = E_chi_tt/N_T with \f$ N_T \neq N_K \f$, so the string
 * "EXT - hkap*EK" means
 *
 *   \f$ E^{tt}_\chi/N_T - hkap\, E^K/N_K
 *      = (1/N_T)[E^{tt}_\chi - hkap (N_T/N_K) E^K] \f$
 *
 * which is proportional to \f$ E^{tt}_\chi - \kappa E^K \f$ only if
 * \c hkap is \f$ \kappa N_K/N_T \f$.  Registering the bare kappa destroys
 * the bit-zero cancellation that is the whole point of the difference form:
 * at the interface that difference has two nonzero slots where bare E_chi_tt
 * has six of order 1e4..1e5 which must cancel against each other.
 *
 * ⚠ And note WHICH quantity may be asserted.  |kappa - 2| < tol is a true
 * statement about the unnormalised ratio and says nothing about what may be
 * registered; the 2-D port made exactly that check, it passed, and it licensed
 * the wrong use.  So kappa is validated against its own neighbour instead --
 * constancy near the interface -- and kappa_eff is reported rather than
 * asserted, because its value depends on the normalisation the caller chose.
 *
 * @param m         the model.
 * @param dh        the interface, from \c locate_horizon_interface.
 * @param norm_dom  domain of the face where the row is actually imposed.
 * @param norm_i    point index of that face.
 * @param who       caller name, so the message names the option that failed.
 */
template <class Model>
HorizonCompat horizon_compat(const Model& m, int dh, int norm_dom, int norm_i,
                             const char* who)
{
    HorizonCompat hc;
    auto C = [&](const char* row, const char* jet, int d, int i) {
        return m.phys_coef(row, jet, d, i);
    };
    if (dh < 0) {
        hc.error = std::string(who) + ": no interface found where E_K's Upp vanishes";
        return hc;
    }
    const int ih = m.nbr(dh) - 1;
    if (ih < 2) {
        hc.error = std::string(who) + ": domain " + std::to_string(dh)
                   + " has too few points to measure kappa off the interface";
        return hc;
    }
    // kappa is measured one point INSIDE: it is 0/0 at the interface itself.
    const double k1 = C("E_chi_tt", "Upp", dh, ih - 1) / C("E_K", "Upp", dh, ih - 1);
    const double k2 = C("E_chi_tt", "Upp", dh, ih - 2) / C("E_K", "Upp", dh, ih - 2);
    if (std::fabs(k1 - k2) > 1e-10 * std::fabs(k1)) {
        hc.error = std::string(who) + ": E_chi_tt.Upp / E_K.Upp is not constant near "
                   "the interface (" + std::to_string(k1) + " vs "
                   + std::to_string(k2) + ")";
        return hc;
    }
    // Every OTHER jet slot of E_chi_tt - kappa*E_K must vanish at the face; Qp
    // is the one that must NOT, since it carries the condition.
    double rowmax = 0.;
    for (const char* jt : {"G", "Gp", "Q", "Qp", "U", "Up", "Upp"})
        rowmax = std::max(rowmax, std::fabs(C("E_chi_tt", jt, dh, ih)));
    for (const char* jt : {"G", "Gp", "Q", "U", "Up", "Upp"}) {
        const double v = C("E_chi_tt", jt, dh, ih) - k1 * C("E_K", jt, dh, ih);
        if (std::fabs(v) > 1e-12 * rowmax) {
            hc.error = std::string(who) + ": slot " + jt + " of E_chi_tt - "
                       + std::to_string(k1) + "*E_K is " + std::to_string(v)
                       + ", not zero";
            return hc;
        }
    }
    hc.dom = dh;
    hc.kappa = k1;
    hc.norm_ratio = m.row_norm(0, norm_dom, norm_i) / m.row_norm(2, norm_dom, norm_i);
    hc.kappa_eff = k1 * hc.norm_ratio;
    return hc;
}

} // namespace Trumpet

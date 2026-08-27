#pragma once
/*
 * Space_oned_trumpet -- 1-D multi-domain space with the puncture EXCISED.
 *
 * Kadath::Space_oned (include/For_Kadath/Domain/oned.hpp:481) unconditionally
 * makes domain 0 a Domain_oned_ori nucleus containing r = 0.  For the trumpet
 * program r = 0 is the throat, where W ~ w0 r^sqrt(2): a branch point with an
 * irrational exponent, so a Chebyshev basis there converges only algebraically.
 * Design decision D1 excises at a finite W0 instead, which needs a space whose
 * innermost domain has an inner boundary.
 *
 * This class is that space:  qcq shells [b0,b1], [b1,b2], ... [b_{n-2},b_{n-1}]
 * followed by one compactified Domain_oned_inf [b_{n-1}, inf).  It touches no
 * library file -- Kadath::Space exposes nbr_domains / ndim / type_base /
 * domains to subclasses and its destructor already deletes them
 * (src/Space/space.cpp:40-47), and all three Domain_oned_* classes are public.
 *
 * Per-domain resolution is supported (the Dim_array-per-domain overload),
 * mirroring Space_spheric's Dim_array** constructor.
 */

#include "For_Kadath/Domain/oned.hpp"
#include "For_Kadath/Space/space.hpp"

#include <cassert>
#include <vector>

namespace Trumpet
{

class Space_oned_trumpet : public Kadath::Space
{
  public:
    /** Uniform resolution.
     *  @param ttype  CHEB_TYPE or LEG_TYPE (integrale() is Chebyshev-only).
     *  @param res    points per domain.
     *  @param bounds b0 < b1 < ... < b_{n-1}; yields n domains, the last
     *                compactified from b_{n-1} to infinity.
     */
    Space_oned_trumpet(int ttype, const Kadath::Dim_array& res,
                       const std::vector<double>& bounds)
        : Space_oned_trumpet(ttype, std::vector<Kadath::Dim_array>(bounds.size(), res), bounds)
    {
    }

    /** Per-domain resolution; res.size() must equal bounds.size(). */
    Space_oned_trumpet(int ttype, const std::vector<Kadath::Dim_array>& res,
                       const std::vector<double>& bounds)
    {
        assert(bounds.size() >= 2);
        assert(res.size() == bounds.size());

        ndim = 1;
        type_base = ttype;
        nbr_domains = static_cast<int>(bounds.size());
        domains = new Kadath::Domain*[nbr_domains];

        // qcq shells between consecutive bounds ...
        for (int i = 0; i < nbr_domains - 1; i++)
            domains[i] = new Kadath::Domain_oned_qcq(i, ttype, bounds[i], bounds[i + 1], res[i]);
        // ... then one compactified domain reaching spatial infinity.
        domains[nbr_domains - 1] =
            new Kadath::Domain_oned_inf(nbr_domains - 1, ttype, bounds[nbr_domains - 1],
                                        res[nbr_domains - 1]);
    }

    explicit Space_oned_trumpet(Kadath::BinarySource& source)
    {
        nbr_domains = source.read<int>();
        ndim = source.read<int>();
        type_base = source.read<int>();
        domains = new Kadath::Domain*[nbr_domains];
        for (int i = 0; i < nbr_domains - 1; i++)
            domains[i] = new Kadath::Domain_oned_qcq(i, source);
        domains[nbr_domains - 1] = new Kadath::Domain_oned_inf(nbr_domains - 1, source);
    }

    ~Space_oned_trumpet() override = default;

    void save(Kadath::BinarySink& sink) const override
    {
        sink.write<int>(nbr_domains);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        for (int i = 0; i < nbr_domains; i++)
            domains[i]->save(sink);
    }
};

} // namespace Trumpet

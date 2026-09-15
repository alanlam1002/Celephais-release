#pragma once
/*
 * Space_polar_trumpet -- 2-D axisymmetric multi-domain space with the puncture
 * EXCISED.  The direct transliteration of Trumpet1d's Space_oned_trumpet.
 *
 * Kadath::Space_polar (Domain/polar.hpp:292) unconditionally makes domain 0 a
 * Domain_polar_nucleus containing r = 0, exactly as Space_oned does in one
 * dimension.  For the trumpet program r = 0 is the throat, where
 * W ~ w0 r^sqrt(2) is a branch point with an irrational exponent, so design
 * decision D1 excises at a finite W0 instead.
 *
 * This class is that space: Domain_polar_shell on [b0,b1], ... [b_{n-2},b_{n-1}]
 * followed by one Domain_polar_compact [b_{n-1}, inf).  It touches no library
 * file -- Kadath::Space exposes nbr_domains / ndim / type_base / domains to
 * subclasses and its destructor already deletes them, and all three
 * Domain_polar_* classes are public.
 *
 * Per-domain resolution is supported, as in the 1-D version, because the
 * O(j^2) layout depends on it (inputs/layout_W005.json res_offset).
 */

#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Space/space.hpp"

#include <cassert>
#include <vector>

namespace Trumpet
{

class Space_polar_trumpet : public Kadath::Space
{
  public:
    /** Uniform resolution. */
    Space_polar_trumpet(int ttype, const Kadath::Point& center,
                        const Kadath::Dim_array& res,
                        const std::vector<double>& bounds)
        : Space_polar_trumpet(ttype, center,
                              std::vector<Kadath::Dim_array>(bounds.size(), res), bounds)
    {
    }

    /** Per-domain resolution; res.size() must equal bounds.size(). */
    Space_polar_trumpet(int ttype, const Kadath::Point& center,
                        const std::vector<Kadath::Dim_array>& res,
                        const std::vector<double>& bounds)
    {
        assert(bounds.size() >= 2);
        assert(res.size() == bounds.size());

        ndim = 2;
        type_base = ttype;
        nbr_domains = static_cast<int>(bounds.size());
        domains = new Kadath::Domain*[nbr_domains];

        for (int i = 0; i < nbr_domains - 1; i++)
            domains[i] = new Kadath::Domain_polar_shell(i, ttype, bounds[i], bounds[i + 1],
                                                        center, res[i]);
        domains[nbr_domains - 1] =
            new Kadath::Domain_polar_compact(nbr_domains - 1, ttype, bounds[nbr_domains - 1],
                                             center, res[nbr_domains - 1]);
    }

    explicit Space_polar_trumpet(Kadath::BinarySource& source)
    {
        nbr_domains = source.read<int>();
        ndim = source.read<int>();
        type_base = source.read<int>();
        domains = new Kadath::Domain*[nbr_domains];
        for (int i = 0; i < nbr_domains - 1; i++)
            domains[i] = new Kadath::Domain_polar_shell(i, source);
        domains[nbr_domains - 1] = new Kadath::Domain_polar_compact(nbr_domains - 1, source);
    }

    ~Space_polar_trumpet() override = default;

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

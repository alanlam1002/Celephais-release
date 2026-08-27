#pragma once
/*
 * table_io.hpp -- readers for the plain-text tables the spinning_trumpet
 * Python side emits.  Shared by backbone_main.cpp (Phase 1) and
 * l0_system_main.cpp (Phase 2 / T2.1).
 *
 * Record formats are whitespace tokens, one record per line, '#' comments:
 *   backbone : point <dom> <ipt> <r> <W> <Rr> <oor>
 *   coefs    : coef  <row> <jet> <dom> <ipt> <value>
 *   reference: ref   <row> <dom> <ipt> <residual> <scale>
 * "inf" parses to an infinity so the compactified domain's r = infinity
 * collocation point round-trips.
 */

#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace TrumpetIO
{

constexpr double kInf = std::numeric_limits<double>::infinity();

struct DomainSpec
{
    std::string kind;      // "qcq" | "inf" | "ori"
    double r_int = 0.0;
    double r_ext = 0.0;
    int nbr = 0;
};

struct PointRow
{
    int dom = 0;
    int ipt = 0;
    double r = 0.0, W = 0.0, Rr = 0.0, oor = 0.0;
};

struct Table
{
    std::string tag, mode;
    double M = 1.0, W0 = 0.0;
    std::vector<DomainSpec> doms;
    std::vector<std::vector<PointRow>> pts;   // [domain][ipoint]
};

inline double parse_double(const std::string& s)
{
    if (s == "inf")
        return kInf;
    if (s == "-inf")
        return -kInf;
    return std::stod(s);
}

inline Table read_table(const std::string& path)
{
    std::ifstream fh(path);
    if (!fh)
        throw std::runtime_error("cannot open table: " + path);

    Table t;
    int ndom = -1;
    std::string line;
    while (std::getline(fh, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream is(line);
        std::string key;
        is >> key;
        if (key == "version") {
            int v;
            is >> v;
            if (v != 1)
                throw std::runtime_error("unsupported table version");
        } else if (key == "tag") {
            is >> t.tag;
        } else if (key == "mode") {
            is >> t.mode;
        } else if (key == "M") {
            std::string v;
            is >> v;
            t.M = parse_double(v);
        } else if (key == "W0") {
            std::string v;
            is >> v;
            t.W0 = parse_double(v);
        } else if (key == "ndom") {
            is >> ndom;
            t.doms.resize(ndom);
            t.pts.resize(ndom);
        } else if (key == "domain") {
            int d;
            std::string kind, ri, re;
            int nbr;
            is >> d >> kind >> ri >> re >> nbr;
            t.doms.at(d) = DomainSpec{kind, parse_double(ri), parse_double(re), nbr};
        } else if (key == "point") {
            PointRow p;
            std::string r, W, Rr, oor;
            is >> p.dom >> p.ipt >> r >> W >> Rr >> oor;
            p.r = parse_double(r);
            p.W = parse_double(W);
            p.Rr = parse_double(Rr);
            p.oor = parse_double(oor);
            t.pts.at(p.dom).push_back(p);
        }
    }
    if (ndom < 0)
        throw std::runtime_error("table has no ndom record");
    for (int d = 0; d < ndom; d++)
        if (static_cast<int>(t.pts[d].size()) != t.doms[d].nbr)
            throw std::runtime_error("point count mismatch in domain "
                                     + std::to_string(d));
    return t;
}


/** coef <row> <jet> <dom> <ipt> <value>, plus the per-row r^p scaling. */
struct CoefTable
{
    std::string tag;
    int ndom = 0, ncoef = 0;
    std::map<std::string, int> scale;                     // row -> p
    std::map<std::string, std::vector<std::string>> jets; // row -> jet names
    // (row, jet) -> [domain][ipoint]
    std::map<std::pair<std::string, std::string>, std::vector<std::vector<double>>> v;
};

inline CoefTable read_coefs(const std::string& path, const std::vector<DomainSpec>& doms)
{
    std::ifstream fh(path);
    if (!fh)
        throw std::runtime_error("cannot open coef table: " + path);
    CoefTable t;
    t.ndom = static_cast<int>(doms.size());
    std::string line;
    while (std::getline(fh, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream is(line);
        std::string key;
        is >> key;
        if (key == "tag") {
            is >> t.tag;
        } else if (key == "ncoef") {
            is >> t.ncoef;
        } else if (key == "scale") {
            std::string row; int p; is >> row >> p; t.scale[row] = p;
        } else if (key == "jets") {
            std::string row, csv; is >> row >> csv;
            std::stringstream ss(csv); std::string tok;
            while (std::getline(ss, tok, ','))
                if (!tok.empty()) t.jets[row].push_back(tok);
        } else if (key == "coef") {
            std::string row, jet; int d, i; std::string val;
            is >> row >> jet >> d >> i >> val;
            auto k = std::make_pair(row, jet);
            auto& g = t.v[k];
            if (g.empty()) {
                g.resize(doms.size());
                for (std::size_t q = 0; q < doms.size(); q++)
                    g[q].assign(doms[q].nbr, 0.0);
            }
            g.at(d).at(i) = parse_double(val);
        }
    }
    return t;
}

/** ref <row> <dom> <ipt> <residual> <scale> */
struct RefTable
{
    std::map<std::string, std::vector<std::vector<double>>> resid, scale;
    // ANALYTIC jets of the reference profile, [jetname][domain][ipoint].
    std::map<std::string, std::vector<std::vector<double>>> jet;
};

inline RefTable read_ref(const std::string& path, const std::vector<DomainSpec>& doms)
{
    std::ifstream fh(path);
    if (!fh)
        throw std::runtime_error("cannot open reference table: " + path);
    RefTable t;
    std::string line;
    auto ensure = [&](std::map<std::string, std::vector<std::vector<double>>>& m,
                      const std::string& row) -> std::vector<std::vector<double>>& {
        auto& g = m[row];
        if (g.empty()) {
            g.resize(doms.size());
            for (std::size_t q = 0; q < doms.size(); q++)
                g[q].assign(doms[q].nbr, 0.0);
        }
        return g;
    };
    while (std::getline(fh, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream is(line);
        std::string key;
        is >> key;
        if (key == "jet") {
            static const char* JN[9] = {"U", "Up", "Upp", "Q", "Qp", "Qpp",
                                        "G", "Gp", "Gpp"};
            int d, i; is >> d >> i;
            for (int k = 0; k < 9; k++) {
                std::string v; is >> v;
                ensure(t.jet, JN[k]).at(d).at(i) = parse_double(v);
            }
            continue;
        }
        if (key != "ref")
            continue;
        std::string row, rv, sv; int d, i;
        is >> row >> d >> i >> rv >> sv;
        ensure(t.resid, row).at(d).at(i) = parse_double(rv);
        ensure(t.scale, row).at(d).at(i) = parse_double(sv);
    }
    return t;
}

/** bc <name> <kind> <field> <rhs_per_j2> <budget>, plus the excision geometry. */
struct BcRow
{
    std::string kind;    // "der1" (d/dr, a FIRST derivative) | "val"
    std::string field;   // "U" | "Q" | "G" | "4U+Q+G"
    double rhs = 0.0;    // per unit j2
    double budget = 0.0; // the research session's own error budget
};

struct BcTable
{
    std::string tag;
    double W0 = 0.0, r_in = 0.0, R_in = 0.0, dWdr = 0.0;
    std::map<std::string, BcRow> row;
};

inline BcTable read_bc(const std::string& path)
{
    std::ifstream fh(path);
    if (!fh)
        throw std::runtime_error("cannot open bc table: " + path);
    BcTable t;
    std::string line;
    while (std::getline(fh, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream is(line);
        std::string key;
        is >> key;
        if (key == "version") {
            int v; is >> v;
            if (v != 1)
                throw std::runtime_error("unsupported bc table version");
        } else if (key == "tag") {
            is >> t.tag;
        } else if (key == "W0" || key == "r_in" || key == "R_in" || key == "dWdr") {
            std::string v; is >> v;
            const double d = parse_double(v);
            if (key == "W0") t.W0 = d;
            else if (key == "r_in") t.r_in = d;
            else if (key == "R_in") t.R_in = d;
            else t.dWdr = d;
        } else if (key == "bc") {
            std::string nm, rv, bv;
            BcRow r;
            is >> nm >> r.kind >> r.field >> rv >> bv;
            r.rhs = parse_double(rv);
            r.budget = parse_double(bv);
            t.row[nm] = r;
        }
    }
    if (t.row.empty())
        throw std::runtime_error("bc table has no bc records: " + path);
    return t;
}

} // namespace TrumpetIO

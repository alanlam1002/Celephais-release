// Mellin throat, M4a (code round 463; research rounds 1034, 1037-1038; spinning_trumpet NOTES_M4_DESIGN.md section 1): THE SETUP BUNDLE AT A
// CYLINDER, natively -- in place of m2_bundle.py's minimal engine set-up at each stage-B iterate.  Of the bundle only the cylinder depends on
// grade 0 (the stage-B unknowns theta): its grid values (cyl_vals: U, P, b, q, Qb, bt on the kernel's theta grid), the constants' pieces
// (cyl_c0: 1 / r, Ph / r, r -- Ph the lapse normalisation K), bnorm (= c), n; and the operator's derived quantities (dE, rw, cw, rw_out, kE,
// cdE, rwc, gt_dE, live), which M3.5's build about the new cylinder supplies.  The tape, the columns' bases, the Gauss tables, the contour,
// the matching data and the configuration are copied from a base bundle of the same set-up.
//
// The cylinder file is n2a_nl's (analysis_stageD/kerr_cylinder_modes.json keys; n3b_linthroat.cyl_dict): u, P_shape, q, Q, b (b_0 = c), bt,
// c, n.  The grid values are n2a_nl.Background('kerr')'s expressions (terms |v| <= 1e-16 dropped, as its msum):
//   U = sum u_j cos 2j th          P = sum P_j cos 2j th          q = sin^2 th sum q_j cos 2j th        Qb = sin^4 th sum Q_j cos (2j+1) th
//   b = sum b_j cos 2j th if any b_j (j >= 1) is nonzero, else c                bt = sum bt_k sin 2(k+1) th (0 if none)
#pragma once
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mk {
namespace cy {

// the numbers of a flat JSON object: key -> its number, or its array of numbers (nested objects / strings are not expected)
inline std::map<std::string, std::vector<double>> read_json_numbers(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("mk: cannot read the cylinder file " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    std::map<std::string, std::vector<double>> out;
    size_t i = 0;
    auto skip = [&]() { while (i < s.size() && (std::isspace(static_cast<unsigned char>(s[i])) || s[i] == ',')) i++; };
    auto number = [&]() {
        size_t j = i;
        while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '-' || s[j] == '+' || s[j] == '.' || s[j] == 'e' || s[j] == 'E')) j++;
        if (j == i) throw std::runtime_error("mk: cylinder file: a number expected at " + std::to_string(i));
        double v = std::stod(s.substr(i, j - i));
        i = j;
        return v;
    };
    i = s.find('{');
    if (i == std::string::npos) throw std::runtime_error("mk: cylinder file: not a JSON object");
    i++;
    while (true) {
        skip();
        if (i >= s.size() || s[i] == '}') break;
        if (s[i] != '"') throw std::runtime_error("mk: cylinder file: a key expected at " + std::to_string(i));
        size_t e = s.find('"', i + 1);
        std::string key = s.substr(i + 1, e - i - 1);
        i = s.find(':', e) + 1;
        skip();
        std::vector<double> v;
        if (s[i] == '[') {
            i++;
            while (true) {
                skip();
                if (s[i] == ']') { i++; break; }
                v.push_back(number());
            }
        } else if (s.compare(i, 4, "null") == 0) {
            i += 4;
        } else {
            v.push_back(number());
        }
        out[key] = v;
    }
    return out;
}

struct Cylinder {
    std::vector<double> u, P, q, Q, b, bt;
    double c = 0.0, n = 0.0;
};

inline Cylinder cylinder_from(const std::map<std::string, std::vector<double>>& m) {
    auto get = [&](const char* k, bool need) {
        auto it = m.find(k);
        if (it == m.end()) {
            if (need) throw std::runtime_error(std::string("mk: cylinder file without '") + k + "'");
            return std::vector<double>();
        }
        return it->second;
    };
    Cylinder c;
    c.u = get("u", true);
    c.P = get("P_shape", true);
    c.q = get("q", true);
    c.Q = get("Q", true);
    c.b = get("b", false);
    c.bt = get("bt", false);
    c.c = get("c", true).at(0);
    c.n = get("n", true).at(0);
    return c;
}

// the six grid values (U, P, b, q, Qb, bt) on th, row-major (6, NF)
inline std::vector<double> cyl_vals(const Cylinder& c, const std::vector<double>& th) {
    const size_t NF = th.size();
    std::vector<double> out(6 * NF, 0.0);
    bool bfree = false, btfree = false;
    for (size_t j = 1; j < c.b.size(); j++) bfree = bfree || std::abs(c.b[j]) > 0;
    for (double v : c.bt) btfree = btfree || std::abs(v) > 0;
    for (size_t g = 0; g < NF; g++) {
        const double t = th[g], s = std::sin(t);
        double U = 0, P = 0, q = 0, Qs = 0, b = 0, bt = 0;
        for (size_t j = 0; j < c.u.size(); j++) if (std::abs(c.u[j]) > 1e-16) U += c.u[j] * std::cos(2.0 * double(j) * t);
        for (size_t j = 0; j < c.P.size(); j++) if (std::abs(c.P[j]) > 1e-16) P += c.P[j] * std::cos(2.0 * double(j) * t);
        for (size_t j = 0; j < c.q.size(); j++) if (std::abs(c.q[j]) > 1e-16) q += c.q[j] * std::cos(2.0 * double(j) * t);
        for (size_t j = 0; j < c.Q.size(); j++) if (std::abs(c.Q[j]) > 1e-16) Qs += c.Q[j] * std::cos((2.0 * double(j) + 1.0) * t);
        if (bfree) {
            for (size_t j = 0; j < c.b.size(); j++) if (std::abs(c.b[j]) > 1e-16) b += c.b[j] * std::cos(2.0 * double(j) * t);
        } else {
            b = c.c;
        }
        if (btfree)
            for (size_t k = 0; k < c.bt.size(); k++) if (std::abs(c.bt[k]) > 1e-16) bt += c.bt[k] * std::sin(2.0 * (double(k) + 1.0) * t);
        out[0 * NF + g] = U;
        out[1 * NF + g] = P;
        out[2 * NF + g] = b;
        out[3 * NF + g] = s * s * q;
        out[4 * NF + g] = s * s * s * s * Qs;
        out[5 * NF + g] = bt;
    }
    return out;
}

}  // namespace cy
}  // namespace mk

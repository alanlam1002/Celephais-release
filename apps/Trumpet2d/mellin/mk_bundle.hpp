// Mellin throat recursion (code round 463 M2; research rounds 1018-1020): the SETUP BUNDLE loader.  spinning_trumpet/scripts/m2_bundle.py
// writes, from the Python engine's own setup code, a directory of .npy arrays and config.txt ('key value' per line); this reads them and
// configures a Kernel (the tape, the columns' bases, the cylinder) and the projections (Proj) with no Python involved.
// The .npy reader handles what the writer emits: format 1.0 / 2.0, little-endian '<i4', '<f8', '<c16', C order.
#pragma once
#include "mk_operator.hpp"

#include <fstream>
#include <map>
#include <sstream>

namespace mk {

struct Npy {
    std::string dtype;
    std::vector<long> shape;
    std::vector<char> raw;
    size_t count() const { size_t n = 1; for (long s : shape) n *= size_t(s); return n; }
};

inline Npy npy_read(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("mk: cannot open " + path);
    char magic[6];
    f.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") throw std::runtime_error("mk: not a .npy file: " + path);
    unsigned char ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    size_t hl = 0;
    if (ver[0] == 1) { unsigned char b[2]; f.read(reinterpret_cast<char*>(b), 2); hl = b[0] | (b[1] << 8); }
    else { unsigned char b[4]; f.read(reinterpret_cast<char*>(b), 4); hl = b[0] | (b[1] << 8) | (b[2] << 16) | (size_t(b[3]) << 24); }
    std::string h(hl, ' ');
    f.read(&h[0], long(hl));
    Npy a;
    auto val = [&](const std::string& key) {
        size_t p = h.find("'" + key + "'");
        if (p == std::string::npos) throw std::runtime_error("mk: .npy header without " + key + ": " + path);
        return h.substr(h.find(':', p) + 1);
    };
    std::string d = val("descr");
    a.dtype = d.substr(d.find('\'') + 1, d.find('\'', d.find('\'') + 1) - d.find('\'') - 1);
    if (val("fortran_order").find("False") == std::string::npos) throw std::runtime_error("mk: Fortran-ordered .npy: " + path);
    std::string s = val("shape");
    s = s.substr(s.find('(') + 1, s.find(')') - s.find('(') - 1);
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (tok.find_first_of("0123456789") != std::string::npos) a.shape.push_back(std::stol(tok));
    size_t item = a.dtype == "<i4" ? 4 : a.dtype == "<f8" ? 8 : a.dtype == "<c16" ? 16 : 0;
    if (!item) throw std::runtime_error("mk: unsupported dtype " + a.dtype + ": " + path);
    a.raw.resize(a.count() * item);
    f.read(a.raw.data(), long(a.raw.size()));
    if (!f) throw std::runtime_error("mk: truncated .npy: " + path);
    return a;
}
template <class T> std::vector<T> npy_as(const Npy& a) {
    std::vector<T> v(a.count());
    if (a.dtype == "<i4") { const int32_t* p = reinterpret_cast<const int32_t*>(a.raw.data()); for (size_t i = 0; i < v.size(); i++) v[i] = T(p[i]); }
    else if (a.dtype == "<f8") { const double* p = reinterpret_cast<const double*>(a.raw.data()); for (size_t i = 0; i < v.size(); i++) v[i] = T(p[i]); }
    else throw std::runtime_error("mk: npy_as: complex array read as real");
    return v;
}
inline std::vector<cd> npy_cplx(const Npy& a) {
    std::vector<cd> v(a.count());
    if (a.dtype == "<c16") { const cd* p = reinterpret_cast<const cd*>(a.raw.data()); for (size_t i = 0; i < v.size(); i++) v[i] = p[i]; }
    else { auto r = npy_as<double>(a); for (size_t i = 0; i < v.size(); i++) v[i] = r[i]; }
    return v;
}

struct Bundle {
    std::map<std::string, std::string> cfg;
    std::vector<int> C0_col, C0_isint;
    std::vector<double> C0_key;
    std::vector<cd> C0_mom;
    double getd(const std::string& k, double d) const { auto i = cfg.find(k); return i == cfg.end() ? d : std::stod(i->second); }
    int geti(const std::string& k, int d) const { auto i = cfg.find(k); return i == cfg.end() ? d : std::stoi(i->second); }
};

// a Kernel and its projections from a bundle directory
inline void load_bundle(const std::string& dir, Kernel& k, Proj& P, Bundle& B) {
    std::ifstream f(dir + "/config.txt");
    if (!f) throw std::runtime_error("mk: no config.txt in " + dir);
    std::string line;
    while (std::getline(f, line)) {
        size_t sp = line.find(' ');
        if (sp != std::string::npos) B.cfg[line.substr(0, sp)] = line.substr(sp + 1);
    }
    auto A = [&](const std::string& n) { return npy_read(dir + "/" + n + ".npy"); };
    int K = B.geti("K", 6), NF = B.geti("NF", 64);
    k.c.init(K, NF, B.getd("n", 0.0), B.getd("Rm", 0.0), npy_cplx(A("D")));
    // the tape
    auto ops = npy_as<int>(A("tape_ops"));
    auto io = npy_as<int>(A("tape_io"));
    Tape& t = k.tp;
    size_t nops = ops.size() / 3;
    t.op.resize(nops); t.a.resize(nops); t.b.resize(nops);
    for (size_t i = 0; i < nops; i++) { t.op[i] = ops[3 * i]; t.a[i] = ops[3 * i + 1]; t.b[i] = ops[3 * i + 2]; }
    t.sc = npy_cplx(A("tape_sc"));
    t.ce = npy_as<int>(A("tape_ce"));
    auto cv = npy_cplx(A("tape_cv"));
    t.cv.clear();
    for (size_t i = 0; i < t.ce.size(); i++) t.cv.emplace_back(cv.begin() + long(i * NF), cv.begin() + long((i + 1) * NF));
    for (int i = 0; i < 6; i++) { t.inputs[i] = io[i]; t.outputs[i] = io[6 + i]; }
    // the columns
    k.cols.ncol = npy_as<int>(A("ncol"));
    k.cols.B.assign(6, {});
    static const char* FAMS_[6] = {"QF", "PS", "PH", "BR", "BT", "QB"};
    for (int fi = 0; fi < 6; fi++) k.cols.B[fi] = npy_as<double>(A(std::string("B_") + FAMS_[fi]));
    k.colfam = npy_as<int>(A("colfam"));
    k.colidx = npy_as<int>(A("colidx"));
    // the cylinder
    auto vals = npy_as<double>(A("cyl_vals"));
    auto sl = [&](int i) { return std::vector<double>(vals.begin() + long(i) * NF, vals.begin() + long(i + 1) * NF); };
    k.cyl.U = sl(0); k.cyl.P = sl(1); k.cyl.b = sl(2); k.cyl.q = sl(3); k.cyl.Qb = sl(4); k.cyl.bt = sl(5);
    k.cyl.bnorm = B.getd("bnorm", 1.0);
    auto c0 = npy_cplx(A("cyl_c0"));
    auto c0e = npy_as<int>(A("cyl_c0_e"));
    k.cyl.e_inv_r = c0e[0]; k.cyl.e_Ph_r = c0e[1]; k.cyl.e_r = c0e[2];
    k.cyl.v_inv_r.assign(c0.begin(), c0.begin() + NF);
    k.cyl.v_Ph_r.assign(c0.begin() + NF, c0.begin() + 2 * NF);
    k.cyl.v_r.assign(c0.begin() + 2 * NF, c0.begin() + 3 * NF);
    // the projections
    P.nn = B.geti("nn", 0); P.nc = B.geti("nc", 0); P.nb = 1;
    P.gfam = npy_as<int>(A("gfam")); P.dE = npy_as<int>(A("dE")); P.W = npy_cplx(A("W"));
    P.cfam = npy_as<int>(A("cfam")); P.cdE = npy_as<int>(A("cdE")); P.Wc = npy_cplx(A("Wc"));
    P.kofs = npy_as<int>(A("kE"));
    // the initial series
    B.C0_col = npy_as<int>(A("C0_col")); B.C0_isint = npy_as<int>(A("C0_isint")); B.C0_key = npy_as<double>(A("C0_key"));
    B.C0_mom = npy_cplx(A("C0_mom"));
}

}  // namespace mk

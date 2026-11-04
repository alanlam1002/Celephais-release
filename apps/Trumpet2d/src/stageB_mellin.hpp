// Code round 463 M4d (research rounds 1034, 1038, 1040-1041; spinning_trumpet NOTES_M4_DESIGN.md): the MELLIN THROAT AS NATIVE STAGE-B
// UNKNOWNS -- a drop-in for sbe::LinEval under TRUMPET_STAGEB_MELLIN.  No outer loop and no linearised file: at every point the stage-B
// Newton visits, the throat is the native engine's (libmellin_kernel: M4a's bundle at the cylinder, M4b's parameters and forward, the
// fit's output map); the Jacobian is M4c's columns -- the grade-0 parameters by central differences (the operator rebuilt about each
// perturbed cylinder), the Mellin parameters by the native tangents -- computed COLLECTIVELY, the columns split across the MPI ranks.
//   unknowns   theta (the cylinder's 'cyl X' parameters, n3b_linthroat's names) and p (n3a_fit's names)
//   rows       the Mellin throat's equations ('eq' outputs) reduced to their independent combinations U_r^T eq -- the column-scaled SVD of
//              d eq / du at the J step's FIRST Jacobian fixes the rank r (research round 1038), the basis U_r is recomputed at every
//              Jacobian -- plus the Tikhonov rows sw u_i / s_i (sw the square-root weight, as the linear throat's tik_sw; labelled TIK)
//   matching   the 'mk' outputs (the data file's keys, in its order)
// Collective calls (research round 1040): values() is reached from Kadath's residual on every rank (rank 0 runs the forward and
// broadcasts); the Jacobian is computed by precompute_jacobian(), called by the application on every rank before Kadath's column loop,
// and jacobian() then returns the cached matrix (an error if u moved).
// Config (whitespace-separated, written by scripts/n3b_linthroat.py m4):
//   MELLIN 1 | BASE dir | CYL template.json | WORK dir | RM r | KCYL index-of-'cyl K' | UNK NU, NU x (kind name-with-underscores u0 h)
//   | MKEYS NMK, NMK x (F d k kept) | CUT value | RANK r (-1: by the cut at the first Jacobian) | TIK NT sw, NT x (index scale)
//   | TANOPT options | END
#pragma once
#include "stageB_eval.hpp"

#include <mpi.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
int mk_bundle_at(const char* base, const char* cyl, double K, const char* out, const char* opts, char* err, int errcap);
int mk_set_params(const char* dir, const char* params, char* err, int errcap);
int mk_run_bundle_tan(const char* dir, const char* out, const char* opts, char* err, int errcap);
}

namespace sbe {
namespace mel {

// ---- minimal readers: the fit's JSON (labels, base), the tangent JSON (names -> index), .npy arrays (f8 / c16, 1-d) ----
inline std::string slurp(const std::string& p)
{
    std::ifstream f(p);
    if (!f) throw std::runtime_error("mellin: cannot read " + p);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}
inline size_t key_at(const std::string& s, const std::string& key)
{
    size_t i = s.find("\"" + key + "\"");
    if (i == std::string::npos) throw std::runtime_error("mellin: no '" + key + "' in a JSON");
    return s.find('[', i);
}
inline std::vector<std::string> json_strings(const std::string& s, const std::string& key)
{
    std::vector<std::string> out;
    size_t i = key_at(s, key) + 1;
    while (i < s.size()) {
        while (i < s.size() && (std::isspace(static_cast<unsigned char>(s[i])) || s[i] == ',')) i++;
        if (s[i] == ']') break;
        size_t e = s.find('"', i + 1);
        out.push_back(s.substr(i + 1, e - i - 1));
        i = e + 1;
    }
    return out;
}
inline std::vector<double> json_numbers(const std::string& s, const std::string& key)
{
    std::vector<double> out;
    size_t i = key_at(s, key) + 1;
    while (i < s.size()) {
        while (i < s.size() && (std::isspace(static_cast<unsigned char>(s[i])) || s[i] == ',')) i++;
        if (s[i] == ']') break;
        size_t e = i;
        while (e < s.size() && s[e] != ',' && s[e] != ']') e++;
        out.push_back(std::stod(s.substr(i, e - i)));
        i = e;
    }
    return out;
}
inline std::vector<double> npy_real(const std::string& p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("mellin: cannot read " + p);
    char mg[8];
    f.read(mg, 8);
    unsigned short hl = 0;
    f.read(reinterpret_cast<char*>(&hl), 2);
    std::string h(hl, ' ');
    f.read(&h[0], hl);
    const bool cplx = h.find("<c16") != std::string::npos;
    size_t a = h.find('('), b = h.find(')');
    size_t n = 1;
    std::string sh = h.substr(a + 1, b - a - 1);
    std::stringstream ss(sh);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (tok.find_first_of("0123456789") != std::string::npos) n *= static_cast<size_t>(std::stoul(tok));
    std::vector<double> out(n);
    if (cplx) {
        std::vector<double> buf(2 * n);
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<long>(16 * n));
        for (size_t i = 0; i < n; i++) out[i] = buf[2 * i];
    } else {
        f.read(reinterpret_cast<char*>(out.data()), static_cast<long>(8 * n));
    }
    return out;
}
// the label with its off-lattice centres (six-decimal numbers) snapped to the nearest of 'cents' within 1e-3 (n3b_linthroat.snap); "" if none
inline std::string snap(const std::string& l, const std::vector<double>& cents)
{
    static const std::regex num("[0-9]+\\.[0-9]{6}");
    std::string out;
    auto it = std::sregex_iterator(l.begin(), l.end(), num);
    size_t last = 0;
    for (; it != std::sregex_iterator(); ++it) {
        double v = std::stod(it->str());
        double best = 1e300, c = 0;
        for (double x : cents) if (std::fabs(x - v) < best) { best = std::fabs(x - v); c = x; }
        if (best > 1e-3) return "";
        char b[32];
        std::snprintf(b, sizeof b, "%.6f", c);
        out += l.substr(last, size_t(it->position()) - last) + b;
        last = size_t(it->position() + it->length());
    }
    return out + l.substr(last);
}

}  // namespace mel

class MellinEval {
  public:
    int NU = 0, NROW = 0, NTHR = 0, NMK = 0, NE = 2;
    double RM = 0;
    std::vector<EqInfo> eqs;
    std::vector<RowInfo> rows;
    std::vector<UnkInfo> unk;
    std::vector<MKey> mk;
    double tval = 0, tjac = 0, tmatch = 0;
    // configuration
    std::string base, cyltpl, work, tanopt;
    std::vector<std::string> uname, kind;
    std::vector<double> u0, h;
    int kcyl = -1, rank_fixed = -1, ntik = 0;
    double cut = 1e-6, sw = 0;
    std::vector<int> tik_i;
    std::vector<double> tik_s;
    // the cache
    std::vector<double> uv, uj;                  // the points of the cached values and Jacobian
    std::vector<double> H, M;                    // eq and mk outputs at uv
    std::vector<double> JH, JM;                  // their full Jacobians at uj (row-major, rows x NU)
    std::vector<double> Ur;                      // the cut's basis (n_eq x r, column-major per combination)
    std::vector<std::string> eqlab, mklab;
    std::vector<double> cents0;
    int neq = 0, rk = -1, ncall = 0;
    int mpirank = 0, mpisize = 1;

    explicit MellinEval(const std::string& path)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
        std::ifstream in(path);
        if (!in) throw std::runtime_error("mellin: cannot read " + path);
        std::string tag;
        int ver = 0;
        auto need = [&](const char* t) { in >> tag; if (tag != t) throw std::runtime_error(std::string("mellin config: expected ") + t + ", got " + tag); };
        need("MELLIN"); in >> ver;
        need("BASE"); in >> base;
        need("CYL"); in >> cyltpl;
        need("WORK"); in >> work;
        need("RM"); in >> RM;
        need("KCYL"); in >> kcyl;
        need("UNK"); in >> NU;
        unk.resize(NU); uname.resize(NU); kind.resize(NU); u0.resize(NU); h.resize(NU);
        for (int q = 0; q < NU; q++) {
            in >> kind[q] >> uname[q] >> u0[q] >> h[q];
            for (auto& ch : uname[q]) if (ch == '~') ch = ' ';         // (the config encodes the names' spaces as ~)
            unk[q].t = 'p'; unk[q].key = q; unk[q].ia = 0; unk[q].u0 = u0[q];
        }
        need("MKEYS"); in >> NMK;
        mk.resize(NMK);
        for (auto& m : mk) in >> m.F >> m.d >> m.k >> m.kept;
        need("CUT"); in >> cut;
        need("RANK"); in >> rank_fixed;
        need("TIK"); in >> ntik >> sw;
        tik_i.resize(ntik); tik_s.resize(ntik);
        for (int k = 0; k < ntik; k++) in >> tik_i[k] >> tik_s[k];
        need("TANOPT"); in >> tanopt;
        need("END");
        if (!in) throw std::runtime_error("mellin: truncated config " + path);
        mkdir(work.c_str(), 0755);
        eqs.resize(2);
        eqs[0].name = "MELLIN"; eqs[1].name = "TIK";
        for (auto& e : eqs) { e.c0 = e.c1 = e.nth = e.nb = 0; }
        // the rows are known after the first Jacobian (the rank); set up at construction from a full evaluation at u0
        std::vector<double> rv, mv;
        values(u0, 0.0, rv, mv);
        rows.clear();
        for (int i = 0; i < rk; i++) rows.push_back(RowInfo{0, 0, i, 0, 0, 1.0});
        for (int k = 0; k < ntik; k++) rows.push_back(RowInfo{1, 9, k, 0, 0, 1.0});
        NROW = NTHR = static_cast<int>(rows.size());
    }

    // ---- one engine evaluation (this rank alone): the outputs' labels and values; tangent directions 'which' ('|'-separated) when not
    // empty, their output derivatives returned in dv (per direction, over the same labels) ----
    void engine(const std::vector<double>& u, int pert, double dlt, const std::string& which, const std::string& tag,
                std::vector<std::string>& lab, std::vector<double>& val, std::map<std::string, std::vector<double>>* dv)
    {
        std::string d = work + "/" + tag;
        mkdir(d.c_str(), 0755);
        // the cylinder file: the template's numbers with theta set (n3b_linthroat.cyl_dict)
        std::map<std::string, std::vector<double>> cy;
        {
            std::string s = mel::slurp(cyltpl);
            for (const char* key : {"u", "P_shape", "q", "Q", "b", "bt", "c", "n", "J"}) {
                size_t i = s.find(std::string("\"") + key + "\"");
                if (i == std::string::npos) continue;
                size_t c = s.find(':', i) + 1;
                while (std::isspace(static_cast<unsigned char>(s[c]))) c++;
                if (s[c] == '[') cy[key] = mel::json_numbers(s, key);
                else { size_t e = s.find_first_of(",}", c); cy[key] = {std::stod(s.substr(c, e - c))}; }
            }
        }
        double K = 1.0;
        for (int q = 0; q < NU; q++) {
            if (kind[q] != "t") continue;
            double v = u[q] + (q == pert ? dlt : 0.0);
            std::string k = uname[q].substr(4);                 // 'cyl X'
            if (k == "c") { cy["c"] = {v}; if (!cy["b"].empty()) cy["b"][0] = v; }
            else if (k == "n") cy["n"] = {v};
            else if (k == "K") K = v;
            else {
                size_t dg = k.find_first_of("0123456789");
                std::string fam = k.substr(0, dg);
                size_t j = std::stoul(k.substr(dg));
                auto& arr = cy[fam];
                if (arr.size() <= j) arr.resize(j + 1, 0.0);
                arr[j] = v;
            }
        }
        {
            std::ofstream f(d + "/cyl.json");
            f.precision(17);
            f << "{";
            bool first = true;
            for (auto& kv : cy) {
                f << (first ? "" : ", ") << "\"" << kv.first << "\": ";
                first = false;
                if (kv.second.size() == 1 && (kv.first == "c" || kv.first == "n" || kv.first == "J")) f << kv.second[0];
                else { f << "["; for (size_t i = 0; i < kv.second.size(); i++) f << (i ? ", " : "") << kv.second[i]; f << "]"; }
            }
            f << "}\n";
        }
        {
            std::ofstream f(d + "/params.txt");
            f.precision(17);
            for (int q = 0; q < NU; q++)
                if (kind[q] == "p") f << uname[q] << " " << u[q] << "\n";
        }
        char err[1024] = {0};
        if (mk_bundle_at(base.c_str(), (d + "/cyl.json").c_str(), K, (d + "/b").c_str(), "", err, sizeof err))
            throw std::runtime_error(std::string("mellin: bundle-at: ") + err);
        if (mk_set_params((d + "/b").c_str(), (d + "/params.txt").c_str(), err, sizeof err))
            throw std::runtime_error(std::string("mellin: set-params: ") + err);
        std::string opts = which.empty() ? std::string("which=none") : ("which=" + which + ";" + tanopt);
        if (mk_run_bundle_tan((d + "/b").c_str(), (d + "/run").c_str(), opts.c_str(), err, sizeof err))
            throw std::runtime_error(std::string("mellin: engine: ") + err);
        std::string fj = mel::slurp(d + "/run/fit.json");
        lab = mel::json_strings(fj, "labels");
        val = mel::json_numbers(fj, "base");
        if (dv && !which.empty()) {
            std::string tj = mel::slurp(d + "/run/tangent.json");
            // directions: [{"name": ..., "index": ...}, ...]
            size_t i = tj.find("\"directions\"");
            while ((i = tj.find("\"name\"", i)) != std::string::npos) {
                size_t a = tj.find('"', tj.find(':', i) + 1), b = tj.find('"', a + 1);
                std::string nm = tj.substr(a + 1, b - a - 1);
                size_t j = tj.find("\"index\"", b);
                int idx = std::stoi(tj.substr(tj.find(':', j) + 1));
                (*dv)[nm] = mel::npy_real(d + "/run/tangent/" + std::to_string(idx) + "_dv.npy");
                i = b;
            }
        }
    }

    // the eq / mk outputs of an evaluation, in the base's label order (labels snapped to the base's centres)
    void select(const std::vector<std::string>& lab, const std::vector<double>& val, std::vector<double>& e, std::vector<double>& m)
    {
        std::map<std::string, double> mp;
        for (size_t i = 0; i < lab.size(); i++) {
            std::string s = cents0.empty() ? lab[i] : mel::snap(lab[i], cents0);
            if (!s.empty()) mp[s] = val[i];
        }
        e.assign(eqlab.size(), 0.0);
        m.assign(mklab.size(), 0.0);
        for (size_t i = 0; i < eqlab.size(); i++) {
            auto it = mp.find(eqlab[i]);
            if (it == mp.end()) throw std::runtime_error("mellin: output missing: " + eqlab[i]);
            e[i] = it->second;
        }
        for (size_t i = 0; i < mklab.size(); i++) {
            auto it = mp.find(mklab[i]);
            if (it == mp.end()) throw std::runtime_error("mellin: output missing: " + mklab[i]);
            m[i] = it->second;
        }
    }

    // ---- the forward at u: rank 0 runs it, every rank receives H, M (collective) ----
    void forward(const std::vector<double>& u)
    {
        const double t0 = MPI_Wtime();
        int nE = 0, nM = 0;
        if (mpirank == 0) {
            std::vector<std::string> lab;
            std::vector<double> val;
            engine(u, -1, 0.0, "", "v", lab, val, nullptr);
            if (eqlab.empty()) {                                  // the label sets from the first evaluation (the base's)
                std::vector<double> cs;
                static const std::regex num("[0-9]+\\.[0-9]{6}");
                for (const auto& l : lab) {
                    if (l.rfind("eq ", 0) == 0) eqlab.push_back(l);
                    if (l.rfind("mk ", 0) == 0) mklab.push_back(l);
                    for (auto it = std::sregex_iterator(l.begin(), l.end(), num); it != std::sregex_iterator(); ++it) cs.push_back(std::stod(it->str()));
                }
                std::sort(cs.begin(), cs.end());
                cs.erase(std::unique(cs.begin(), cs.end()), cs.end());
                cents0 = cs;
                if (static_cast<int>(mklab.size()) != NMK) throw std::runtime_error("mellin: " + std::to_string(mklab.size()) + " mk outputs vs NMK " + std::to_string(NMK));
            }
            select(lab, val, H, M);
            nE = static_cast<int>(H.size());
            nM = static_cast<int>(M.size());
        }
        MPI_Bcast(&nE, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nM, 1, MPI_INT, 0, MPI_COMM_WORLD);
        H.resize(nE); M.resize(nM);
        MPI_Bcast(H.data(), nE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(M.data(), nM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        neq = nE;
        uv = u;
        tval += MPI_Wtime() - t0;
    }

    // ---- the Jacobian at u (collective): the grade-0 columns by central differences, the Mellin columns by the tangents; tasks split
    // across the ranks; then the cut (rank fixed at the first Jacobian, the basis recomputed) ----
    void precompute_jacobian(const std::vector<double>& u)
    {
        if (!uj.empty() && uj == u) return;
        const double t0 = MPI_Wtime();
        if (uv != u) forward(u);
        // the label sets to every rank (the snapping needs them)
        {
            std::string all;
            if (mpirank == 0) {
                std::ostringstream o;
                o.precision(17);
                o << eqlab.size() << "\n";
                for (auto& l : eqlab) o << l << "\n";
                o << mklab.size() << "\n";
                for (auto& l : mklab) o << l << "\n";
                o << cents0.size() << "\n";
                for (double c : cents0) o << c << "\n";
                all = o.str();
            }
            int len = static_cast<int>(all.size());
            MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
            all.resize(static_cast<size_t>(len));
            MPI_Bcast(&all[0], len, MPI_CHAR, 0, MPI_COMM_WORLD);
            if (mpirank != 0) {
                std::istringstream is(all);
                std::string l;
                size_t n;
                eqlab.clear(); mklab.clear(); cents0.clear();
                is >> n; std::getline(is, l);
                for (size_t i = 0; i < n; i++) { std::getline(is, l); eqlab.push_back(l); }
                is >> n; std::getline(is, l);
                for (size_t i = 0; i < n; i++) { std::getline(is, l); mklab.push_back(l); }
                is >> n;
                for (size_t i = 0; i < n; i++) { double c; is >> c; cents0.push_back(c); }
            }
        }
        const int ne = static_cast<int>(eqlab.size()), nm = static_cast<int>(mklab.size()), no = ne + nm;
        // tasks: (q, +1), (q, -1) for every theta; (q, 0) for every p (a tangent)
        std::vector<std::pair<int, int>> tasks;
        for (int q = 0; q < NU; q++) {
            if (kind[q] == "t") { tasks.push_back({q, 1}); tasks.push_back({q, -1}); }
            else tasks.push_back({q, 0});
        }
        std::vector<double> mine;                                 // per task of this rank: q, s, then no values
        for (size_t t = static_cast<size_t>(mpirank); t < tasks.size(); t += static_cast<size_t>(mpisize)) {
            const int q = tasks[t].first, s = tasks[t].second;
            std::vector<std::string> lab;
            std::vector<double> val, e, m;
            std::string tag = "j" + std::to_string(mpirank);
            if (s != 0) {
                engine(u, q, s * h[q], "", tag, lab, val, nullptr);
                select(lab, val, e, m);
            } else {
                std::map<std::string, std::vector<double>> dv;
                engine(u, -1, 0.0, uname[q], tag, lab, val, &dv);
                if (!dv.count(uname[q])) throw std::runtime_error("mellin: no tangent for " + uname[q]);
                select(lab, dv[uname[q]], e, m);
            }
            mine.push_back(q);
            mine.push_back(s);
            mine.insert(mine.end(), e.begin(), e.end());
            mine.insert(mine.end(), m.begin(), m.end());
        }
        // gather to every rank (each rank holds the whole Jacobian; it is small)
        int cnt = static_cast<int>(mine.size());
        std::vector<int> cnts(mpisize), offs(mpisize, 0);
        MPI_Allgather(&cnt, 1, MPI_INT, cnts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        for (int p = 1; p < mpisize; p++) offs[p] = offs[p - 1] + cnts[p - 1];
        std::vector<double> all(static_cast<size_t>(offs[mpisize - 1] + cnts[mpisize - 1]));
        MPI_Allgatherv(mine.data(), cnt, MPI_DOUBLE, all.data(), cnts.data(), offs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        std::vector<double> plus(static_cast<size_t>(NU) * no, 0.0), minus = plus;
        JH.assign(static_cast<size_t>(ne) * NU, 0.0);
        JM.assign(static_cast<size_t>(nm) * NU, 0.0);
        for (size_t a = 0; a < all.size(); a += 2 + static_cast<size_t>(no)) {
            const int q = static_cast<int>(all[a]), s = static_cast<int>(all[a + 1]);
            const double* v = &all[a + 2];
            if (s == 1) std::memcpy(&plus[static_cast<size_t>(q) * no], v, sizeof(double) * no);
            else if (s == -1) std::memcpy(&minus[static_cast<size_t>(q) * no], v, sizeof(double) * no);
            else {
                for (int r = 0; r < ne; r++) JH[static_cast<size_t>(r) * NU + q] = v[r];
                for (int r = 0; r < nm; r++) JM[static_cast<size_t>(r) * NU + q] = v[ne + r];
            }
        }
        for (int q = 0; q < NU; q++) {
            if (kind[q] != "t") continue;
            const double* p = &plus[static_cast<size_t>(q) * no];
            const double* mm = &minus[static_cast<size_t>(q) * no];
            for (int r = 0; r < ne; r++) JH[static_cast<size_t>(r) * NU + q] = (p[r] - mm[r]) / (2 * h[q]);
            for (int r = 0; r < nm; r++) JM[static_cast<size_t>(r) * NU + q] = (p[ne + r] - mm[ne + r]) / (2 * h[q]);
        }
        cut_basis();
        uj = u;
        tjac += MPI_Wtime() - t0;
        if (mpirank == 0) {                                       // the record (for the gates): labels, values, the full Jacobians, u
            std::ofstream f(work + "/jac_" + std::to_string(ncall + 1) + ".txt");
            f.precision(17);
            f << "NU " << NU << " NE " << ne << " NM " << nm << " RANK " << rk << "\n";
            for (int q = 0; q < NU; q++) f << uname[q] << "\t" << u[q] << "\n";
            for (int r = 0; r < ne; r++) f << eqlab[r] << "\t" << H[r] << "\n";
            for (int r = 0; r < nm; r++) f << mklab[r] << "\t" << M[r] << "\n";
            for (double v : JH) f << v << "\n";
            for (double v : JM) f << v << "\n";
        }
        if (mpirank == 0)
            std::cout << "#  mellin: Jacobian " << ++ncall << " (" << tasks.size() << " tasks over " << mpisize << " ranks) in " << (MPI_Wtime() - t0)
                      << " s; the cut's rank " << rk << " of " << ne << " eq rows\n";
    }

    // the column-scaled SVD of d eq / du (write_lin_full's): U_r, r frozen at the first (or the config's)
    void cut_basis()
    {
        const int ne = neq;
        std::vector<double> Dc(NU, 0.0);
        for (int q = 0; q < NU; q++) {
            double a = 0, b = 0;
            for (int r = 0; r < ne; r++) a += JH[static_cast<size_t>(r) * NU + q] * JH[static_cast<size_t>(r) * NU + q];
            for (size_t r = 0; r < mklab.size(); r++) b += JM[r * NU + q] * JM[r * NU + q];
            Dc[q] = std::max(std::sqrt(a), std::sqrt(b));
            if (Dc[q] == 0) Dc[q] = 1.0;
        }
        // the SVD of the ne x NU matrix by the eigen-decomposition of A A^T is ill-conditioned; use one-sided Jacobi on A^T (NU x ne)
        const int m = NU, n = ne;
        std::vector<double> At(static_cast<size_t>(m) * n);                 // column j of A^T = row j of A (scaled)
        for (int r = 0; r < n; r++)
            for (int q = 0; q < m; q++) At[static_cast<size_t>(r) * m + q] = JH[static_cast<size_t>(r) * NU + q] / Dc[q];
        std::vector<double> V(static_cast<size_t>(n) * n, 0.0);
        for (int i = 0; i < n; i++) V[static_cast<size_t>(i) * n + i] = 1.0;
        for (int sweep = 0; sweep < 60; sweep++) {
            double off = 0;
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++) {
                    double a = 0, b = 0, c = 0;
                    double* ci = &At[static_cast<size_t>(i) * m];
                    double* cj = &At[static_cast<size_t>(j) * m];
                    for (int k = 0; k < m; k++) { a += ci[k] * ci[k]; b += cj[k] * cj[k]; c += ci[k] * cj[k]; }
                    if (std::fabs(c) <= 1e-15 * std::sqrt(a * b)) continue;
                    off = std::max(off, std::fabs(c) / std::sqrt(a * b));
                    double z = (b - a) / (2 * c), t = (z >= 0 ? 1.0 : -1.0) / (std::fabs(z) + std::sqrt(1 + z * z));
                    double cs = 1 / std::sqrt(1 + t * t), sn = cs * t;
                    for (int k = 0; k < m; k++) { double x = ci[k], y = cj[k]; ci[k] = cs * x - sn * y; cj[k] = sn * x + cs * y; }
                    double* vi = &V[static_cast<size_t>(i) * n];
                    double* vj = &V[static_cast<size_t>(j) * n];
                    for (int k = 0; k < n; k++) { double x = vi[k], y = vj[k]; vi[k] = cs * x - sn * y; vj[k] = sn * x + cs * y; }
                }
            if (off < 1e-15) break;
        }
        // singular values = column norms of the rotated A^T; the left singular vectors of A (in eq space) = the columns of V
        std::vector<std::pair<double, int>> sv;
        for (int i = 0; i < n; i++) {
            double a = 0;
            for (int k = 0; k < m; k++) a += At[static_cast<size_t>(i) * m + k] * At[static_cast<size_t>(i) * m + k];
            sv.push_back({std::sqrt(a), i});
        }
        std::sort(sv.begin(), sv.end(), [](const std::pair<double, int>& x, const std::pair<double, int>& y) { return x.first > y.first; });
        if (rk < 0) {
            if (rank_fixed > 0) rk = rank_fixed;
            else { rk = 0; for (auto& p : sv) if (p.first > cut * sv[0].first) rk++; }
        }
        Ur.assign(static_cast<size_t>(rk) * n, 0.0);
        for (int c = 0; c < rk; c++) {
            const int i = sv[c].second;
            for (int k = 0; k < n; k++) Ur[static_cast<size_t>(c) * n + k] = V[static_cast<size_t>(i) * n + k];
        }
    }

    void values(const std::vector<double>& u, double, std::vector<double>& rv, std::vector<double>& mv)
    {
        if (uv != u) forward(u);
        if (Ur.empty()) precompute_jacobian(u);                 // the first call: the rank and the basis
        rv.assign(static_cast<size_t>(rk + ntik), 0.0);
        for (int c = 0; c < rk; c++) {
            double acc = 0;
            for (int k = 0; k < neq; k++) acc += Ur[static_cast<size_t>(c) * neq + k] * H[k];
            rv[c] = acc;
        }
        for (int k = 0; k < ntik; k++) rv[rk + k] = sw * u[tik_i[k]] / tik_s[k];   // (sw: the square-root weight, write_lin_full's tik_sw)
        mv = M;
    }

    void jacobian(const std::vector<double>& u, double, std::vector<double>& Jr, std::vector<double>& Jm)
    {
        if (uj != u) throw std::runtime_error("mellin: the Jacobian requested at a point it was not precomputed at (precompute_jacobian on every rank first)");
        Jr.assign(static_cast<size_t>(rk + ntik) * NU, 0.0);
        for (int c = 0; c < rk; c++)
            for (int q = 0; q < NU; q++) {
                double acc = 0;
                for (int k = 0; k < neq; k++) acc += Ur[static_cast<size_t>(c) * neq + k] * JH[static_cast<size_t>(k) * NU + q];
                Jr[static_cast<size_t>(c) * NU + q] = acc;
            }
        for (int k = 0; k < ntik; k++) Jr[static_cast<size_t>(rk + k) * NU + tik_i[k]] = sw / tik_s[k];
        Jm = JM;
    }
};
}  // namespace sbe

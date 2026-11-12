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
//   | MKEYS NMK, NMK x (F d k kept) | CUT value, or 'rulec EPS_TAN' | RANK r (-1: by the cut at the first Jacobian) | TIK NT sw, NT x (index scale centre)
//   | [EQDROP N, N x label (spaces as ~)] | TANOPT options | END
// CUT rulec (research rounds 1008-1009, 1038: n3b_linthroat's 'rulec', eps_J measured once per J step): at the J step's first Jacobian the
// grade-0 columns also at +-2h (Richardson), eps_J central = ||(J_c(2h) - J_c(h)) / 3||_2 / s_0 in the cut's column-scaled eq space (eps_J
// forward = ||J_f - J_c||_2 / s_0 reported), tau = the largest 'monrow' output at u; cut = max(10 tau, 10 eps_J central, 10 EPS_TAN).
// The first Jacobian is cached (WORK/jac_cache.txt: u, labels, centres, values, Jacobians, rank, cut) and loaded when u0 matches it bitwise;
// The operator's discrete decisions (its rows, their levels, the gate rows' liveness) are those of the J step's first build (WORK/levels.txt;
// mk_op_build levels_out / levels_in): a non-Kerr b(theta) moves a 1e-7 level test inside a finite difference otherwise (J 0.10: b1 ~ 4e-9).
// MELLIN_PROBE=1 (env) ends the run after the construction (the first Jacobian, its rank and cut -> WORK/probe.txt), so a driver can size
// the defect file's rows before the solve -- which then starts from the cache.
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
#include <set>
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
// the centre map of an evaluation's labels onto the reference centres (n3b_linthroat.load_aligned, research round 1030): each reference
// centre takes its nearest new centre within a quarter of the references' smallest separation, one to one (an outer step -- here a Newton
// step -- moves the off-lattice centres 1 + s / n with n and the cylinder, by more than 1e-3 at 0.45 and in a J step); the others dropped
inline std::map<std::string, std::string> centre_map(const std::vector<std::string>& lab, const std::vector<double>& cents)
{
    static const std::regex num("[0-9]+\\.[0-9]{6}");
    double gap = 1.0;
    for (size_t i = 1; i < cents.size(); i++) gap = std::min(gap, cents[i] - cents[i - 1]);
    const double tol = 0.25 * gap;
    std::vector<double> nv;
    for (const auto& l : lab)
        for (auto it = std::sregex_iterator(l.begin(), l.end(), num); it != std::sregex_iterator(); ++it) nv.push_back(std::stod(it->str()));
    std::sort(nv.begin(), nv.end());
    nv.erase(std::unique(nv.begin(), nv.end()), nv.end());
    std::map<double, double> best;                                // reference centre -> its new centre
    for (double v : nv) {
        double bd = 1e300, c = 0;
        for (double x : cents) if (std::fabs(x - v) < bd) { bd = std::fabs(x - v); c = x; }
        if (bd <= tol && (!best.count(c) || std::fabs(c - v) < std::fabs(c - best[c]))) best[c] = v;
    }
    std::map<std::string, std::string> vm;
    char a[32], b[32];
    for (auto& cv : best) {
        std::snprintf(a, sizeof a, "%.6f", cv.second);
        std::snprintf(b, sizeof b, "%.6f", cv.first);
        vm[a] = b;
    }
    return vm;
}
// the label with its off-lattice centres (six-decimal numbers) mapped by centre_map; "" if one has no reference centre
inline std::string snap(const std::string& l, const std::map<std::string, std::string>& vm)
{
    static const std::regex num("[0-9]+\\.[0-9]{6}");
    std::string out;
    auto it = std::sregex_iterator(l.begin(), l.end(), num);
    size_t last = 0;
    for (; it != std::sregex_iterator(); ++it) {
        auto f = vm.find(it->str());
        if (f == vm.end()) return "";
        out += l.substr(last, size_t(it->position()) - last) + f->second;
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
    bool rule = false;                           // CUT rulec: the cut from the measured accuracies at the first Jacobian
    double eps_tan = 0, tau = 0, eps_c = -1, eps_f = -1;
    std::vector<int> tik_i;
    std::vector<double> tik_s, tik_c;            // (tik_c: the prior's centre -- research round 1043, the J-predicted value; 0 before)
    // the cache
    std::vector<double> uv, uj;                  // the points of the cached values and Jacobian
    std::vector<double> H, M;                    // eq and mk outputs at uv
    std::vector<double> JH, JM;                  // their full Jacobians at uj (row-major, rows x NU)
    std::vector<double> Ur;                      // the cut's basis (n_eq x r, column-major per combination)
    std::vector<std::string> eqlab, mklab;
    std::set<std::string> eqdrop;                                 // EQDROP: 'eq' outputs not among the equations
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
        need("CUT");
        {
            std::string ct;
            in >> ct;
            if (ct == "rulec") { rule = true; in >> eps_tan; }
            else cut = std::stod(ct);
        }
        need("RANK"); in >> rank_fixed;
        need("TIK"); in >> ntik >> sw;
        tik_i.resize(ntik); tik_s.resize(ntik); tik_c.assign(ntik, 0.0);
        for (int k = 0; k < ntik; k++) in >> tik_i[k] >> tik_s[k] >> tik_c[k];
        // research round 1051 (spec section 33, the l-spectra test (i)): optional 'EQDROP N label~with~tildes ...' -- 'eq' outputs left out of the
        // equations (the top degrees' aligned-1 rows)
        in >> tag;
        if (tag == "EQDROP") {
            int nd = 0;
            in >> nd;
            for (int k = 0; k < nd; k++) {
                std::string l;
                in >> l;
                for (auto& ch : l) if (ch == '~') ch = ' ';
                eqdrop.insert(l);
            }
            in >> tag;
        }
        if (tag != "TANOPT") throw std::runtime_error(std::string("mellin config: expected TANOPT, got ") + tag);
        in >> tanopt;
        need("END");
        if (!in) throw std::runtime_error("mellin: truncated config " + path);
        mkdir(work.c_str(), 0755);
        eqs.resize(2);
        eqs[0].name = "MELLIN"; eqs[1].name = "TIK";
        for (auto& e : eqs) { e.c0 = e.c1 = e.nth = e.nb = 0; }
        // the rows are known after the first Jacobian (the rank); set up at construction from a full evaluation at u0
        std::vector<double> rv, mv;
        values(u0, 0.0, rv, mv);
        if (std::getenv("MELLIN_PROBE")) {                        // the driver's probe: the first Jacobian's rank and cut, then stop
            if (mpirank == 0) {
                std::ofstream f(work + "/probe.txt");
                f.precision(17);
                f << "RANK " << rk << "\nCUT " << cut << "\nEPSC " << eps_c << "\nEPSF " << eps_f << "\nTAU " << tau << "\nNTIK " << ntik
                  << "\nTJAC " << tjac << "\nTVAL " << tval << "\n";
                std::cout << "#  mellin: probe -- rank " << rk << ", cut " << cut << " -> " << work << "/probe.txt\n" << std::flush;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            std::exit(0);
        }
        rows.clear();
        for (int i = 0; i < rk; i++) rows.push_back(RowInfo{0, 0, i, 0, 0, 1.0});
        for (int k = 0; k < ntik; k++) rows.push_back(RowInfo{1, 9, k, 0, 0, 1.0});
        NROW = NTHR = static_cast<int>(rows.size());
    }

    // ---- one engine evaluation (this rank alone): the outputs' labels and values; tangent directions 'which' ('|'-separated) when not
    // empty, their output derivatives returned in dv (per direction, over the same labels) ----
    std::vector<std::string> last_lab;           // the last forward's outputs (forward(): the engine call guarded, then read here)
    std::vector<double> last_val;
    int nfail = 0;                               // failed forwards at trial points (round 1043)
    static constexpr double FAIL_VALUE = 1e100;  // a failed forward's outputs: an infinite residual to the line search, finite in its norms
    void engine_last(std::vector<std::string>& lab, std::vector<double>& val) const { lab = last_lab; val = last_val; }

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
        // the operator's structure frozen per J step (M4e): the first build's discrete decisions (rows, levels, liveness) -> WORK/levels.txt,
        // reused by every later build -- only its coefficients follow the cylinder (as the cut's rank)
        const std::string lv = work + "/levels.txt";
        const std::string bopt = (std::ifstream(lv) ? "levels_in=" : "levels_out=") + lv;
        if (mk_bundle_at(base.c_str(), (d + "/cyl.json").c_str(), K, (d + "/b").c_str(), bopt.c_str(), err, sizeof err))
            throw std::runtime_error(std::string("mellin: bundle-at: ") + err);
        if (mk_set_params((d + "/b").c_str(), (d + "/params.txt").c_str(), err, sizeof err))
            throw std::runtime_error(std::string("mellin: set-params: ") + err);
        std::string opts = which.empty() ? std::string("which=none") : ("which=" + which + ";" + tanopt);
        if (mk_run_bundle_tan((d + "/b").c_str(), (d + "/run").c_str(), opts.c_str(), err, sizeof err))
            throw std::runtime_error(std::string("mellin: engine: ") + err);
        std::string fj = mel::slurp(d + "/run/fit.json");
        lab = mel::json_strings(fj, "labels");
        val = mel::json_numbers(fj, "base");
        if (tag == "v") { last_lab = lab; last_val = val; }
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
        const auto vm = mel::centre_map(lab, cents0);
        for (size_t i = 0; i < lab.size(); i++) {
            std::string s = cents0.empty() ? lab[i] : mel::snap(lab[i], vm);
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
        int nE = 0, nM = 0, failed = 0;
        if (mpirank == 0) {
            std::vector<std::string> lab;
            std::vector<double> val;
            // research round 1043: a failed forward at a trial point (an engine error, or a non-finite output) counts as an INFINITE residual
            // -- the line search backtracks -- and is logged; at the first evaluation (the start) it is an error
            try {
                engine(u, -1, 0.0, "", "v", lab, val, nullptr);
                for (double v : val) if (!std::isfinite(v)) throw std::runtime_error("a non-finite output");
            } catch (const std::exception& ex) {
                if (eqlab.empty()) throw;
                failed = 1;
                std::cout << "#  mellin: the forward FAILED at a trial point (" << ex.what() << "): an infinite residual, the line search backtracks\n" << std::flush;
            }
        }
        MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (failed) {
            nfail++;
            H.assign(eqlab.empty() ? static_cast<size_t>(neq) : eqlab.size(), FAIL_VALUE);
            M.assign(static_cast<size_t>(NMK), FAIL_VALUE);
            neq = static_cast<int>(H.size());
            uv = u;
            tval += MPI_Wtime() - t0;
            return;
        }
        if (mpirank == 0) {
            std::vector<std::string> lab;
            std::vector<double> val;
            engine_last(lab, val);
            if (eqlab.empty()) {                                  // the label sets from the first evaluation (the base's)
                std::vector<double> cs;
                static const std::regex num("[0-9]+\\.[0-9]{6}");
                for (const auto& l : lab) {
                    if (l.rfind("eq ", 0) == 0 && !eqdrop.count(l) && !std::any_of(eqdrop.begin(), eqdrop.end(), [&](const std::string& d) {
                        return !d.empty() && d.back() == '*' && l.rfind(d.substr(0, d.size() - 1), 0) == 0; }))   // (round 1079: 'PREFIX*' released)
                        eqlab.push_back(l);
                    if (l.rfind("mk ", 0) == 0) mklab.push_back(l);
                    for (auto it = std::sregex_iterator(l.begin(), l.end(), num); it != std::sregex_iterator(); ++it) cs.push_back(std::stod(it->str()));
                }
                std::sort(cs.begin(), cs.end());
                cs.erase(std::unique(cs.begin(), cs.end()), cs.end());
                cents0 = cs;
                if (static_cast<int>(mklab.size()) != NMK) throw std::runtime_error("mellin: " + std::to_string(mklab.size()) + " mk outputs vs NMK " + std::to_string(NMK));
            }
            select(lab, val, H, M);
            tau = 0;                                              // the rows' measured level (n3b_linthroat.tau_measured)
            for (size_t i = 0; i < lab.size(); i++) if (lab[i].rfind("monrow ", 0) == 0) tau = std::max(tau, std::fabs(val[i]));
            nE = static_cast<int>(H.size());
            nM = static_cast<int>(M.size());
        }
        MPI_Bcast(&nE, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nM, 1, MPI_INT, 0, MPI_COMM_WORLD);
        H.resize(nE); M.resize(nM);
        MPI_Bcast(H.data(), nE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(M.data(), nM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&tau, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
        if (uj.empty() && rk < 0 && load_cache(u)) {
            tjac += MPI_Wtime() - t0;
            return;
        }
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
        // (CUT rulec, the J step's first Jacobian: (q, +2), (q, -2) too -- Richardson's eps_J)
        const bool rich = rule && rk < 0;
        std::vector<std::pair<int, int>> tasks;
        for (int q = 0; q < NU; q++) {
            if (kind[q] == "t") { tasks.push_back({q, 1}); tasks.push_back({q, -1}); }
            else tasks.push_back({q, 0});
        }
        if (rich)
            for (int q = 0; q < NU; q++)
                if (kind[q] == "t") { tasks.push_back({q, 2}); tasks.push_back({q, -2}); }
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
                if (dv[uname[q]].size() != lab.size())                // (a failed direction writes an empty dv: tangent.json's 'error')
                    throw std::runtime_error("mellin: the tangent for " + uname[q] + " has " + std::to_string(dv[uname[q]].size()) + " outputs, the base " +
                                             std::to_string(lab.size()) + " (" + work + "/" + tag + "/run/tangent.json)");
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
        std::vector<double> plus(static_cast<size_t>(NU) * no, 0.0), minus = plus, plus2 = plus, minus2 = plus;
        JH.assign(static_cast<size_t>(ne) * NU, 0.0);
        JM.assign(static_cast<size_t>(nm) * NU, 0.0);
        for (size_t a = 0; a < all.size(); a += 2 + static_cast<size_t>(no)) {
            const int q = static_cast<int>(all[a]), s = static_cast<int>(all[a + 1]);
            const double* v = &all[a + 2];
            if (s == 1) std::memcpy(&plus[static_cast<size_t>(q) * no], v, sizeof(double) * no);
            else if (s == -1) std::memcpy(&minus[static_cast<size_t>(q) * no], v, sizeof(double) * no);
            else if (s == 2) std::memcpy(&plus2[static_cast<size_t>(q) * no], v, sizeof(double) * no);
            else if (s == -2) std::memcpy(&minus2[static_cast<size_t>(q) * no], v, sizeof(double) * no);
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
        if (rich) {                                               // eps_J (n3b_linthroat.eps_J) and the rule's cut
            std::vector<double> Dc = col_scale();
            std::vector<double> A(static_cast<size_t>(ne) * NU), Ef = A, Ec = A;
            for (int r = 0; r < ne; r++)
                for (int q = 0; q < NU; q++) {
                    const size_t i = static_cast<size_t>(r) * NU + q;
                    A[i] = JH[i] / Dc[q];
                    if (kind[q] != "t") continue;
                    const double c2 = (plus2[static_cast<size_t>(q) * no + r] - minus2[static_cast<size_t>(q) * no + r]) / (4 * h[q]);
                    const double fw = (plus[static_cast<size_t>(q) * no + r] - H[r]) / h[q];
                    Ec[i] = (c2 - JH[i]) / Dc[q] / 3;
                    Ef[i] = (fw - JH[i]) / Dc[q];
                }
            const double s0 = smax(A, ne, NU);
            eps_c = smax(Ec, ne, NU) / s0;
            eps_f = smax(Ef, ne, NU) / s0;
            cut = std::max({10 * tau, 10 * eps_c, 10 * eps_tan});
            if (mpirank == 0)
                std::cout << "#  mellin: eps_J central " << eps_c << ", forward " << eps_f << "; tau " << tau << ", eps_tan " << eps_tan
                          << " -> cut " << cut << "\n";
        }
        cut_basis();
        uj = u;
        tjac += MPI_Wtime() - t0;
        if (rich && mpirank == 0) save_cache(u);
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

    // the column scales of write_lin_full: max(|d eq / du_q|, |d mk / du_q|)
    std::vector<double> col_scale() const
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
        return Dc;
    }

    // the largest singular value of the n x m row-major matrix A (one-sided Jacobi, as cut_basis)
    static double smax(const std::vector<double>& A, int n, int m)
    {
        std::vector<double> At(A);
        std::vector<double> V;
        std::vector<std::pair<double, int>> sv;
        jacobi(At, m, n, V, sv, false);
        return sv.empty() ? 0.0 : sv[0].first;
    }

    // one-sided Jacobi on At (n columns of length m: the rows of A); sv = (singular value, column) descending; V (n x n) the rotations
    static void jacobi(std::vector<double>& At, int m, int n, std::vector<double>& V, std::vector<std::pair<double, int>>& sv, bool wantV)
    {
        if (wantV) {
            V.assign(static_cast<size_t>(n) * n, 0.0);
            for (int i = 0; i < n; i++) V[static_cast<size_t>(i) * n + i] = 1.0;
        }
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
                    if (!wantV) continue;
                    double* vi = &V[static_cast<size_t>(i) * n];
                    double* vj = &V[static_cast<size_t>(j) * n];
                    for (int k = 0; k < n; k++) { double x = vi[k], y = vj[k]; vi[k] = cs * x - sn * y; vj[k] = sn * x + cs * y; }
                }
            if (off < 1e-15) break;
        }
        sv.clear();
        for (int i = 0; i < n; i++) {
            double a = 0;
            for (int k = 0; k < m; k++) a += At[static_cast<size_t>(i) * m + k] * At[static_cast<size_t>(i) * m + k];
            sv.push_back({std::sqrt(a), i});
        }
        std::sort(sv.begin(), sv.end(), [](const std::pair<double, int>& x, const std::pair<double, int>& y) { return x.first > y.first; });
    }

    // the first Jacobian's cache (CUT rulec): written by rank 0, read by every rank when u0 matches bitwise
    void save_cache(const std::vector<double>& u) const
    {
        std::ofstream f(work + "/jac_cache.txt");
        f.precision(17);
        f << "MELLINCACHE 1\n" << NU << " " << neq << " " << mklab.size() << " " << rk << " " << cut << " " << eps_c << " " << eps_f << " " << tau << "\n";
        for (double v : u) f << v << "\n";
        f << cents0.size() << "\n";
        for (double c : cents0) f << c << "\n";
        for (int r = 0; r < neq; r++) f << eqlab[r] << "\t" << H[r] << "\n";
        for (size_t r = 0; r < mklab.size(); r++) f << mklab[r] << "\t" << M[r] << "\n";
        for (double v : JH) f << v << "\n";
        for (double v : JM) f << v << "\n";
    }

    bool load_cache(const std::vector<double>& u)
    {
        std::ifstream f(work + "/jac_cache.txt");
        if (!f) return false;
        std::string tag, l;
        int ver = 0, nu = 0, ne = 0, nm = 0, r = 0;
        double c = 0, ec = 0, ef = 0, ta = 0;
        f >> tag >> ver >> nu >> ne >> nm >> r >> c >> ec >> ef >> ta;
        if (tag != "MELLINCACHE" || nu != NU) return false;
        std::vector<double> uc(NU);
        for (auto& v : uc) f >> v;
        if (uc != u) return false;                                // (bitwise: %.17g round-trips)
        size_t nc = 0;
        f >> nc;
        std::vector<double> cs(nc);
        for (auto& v : cs) f >> v;
        std::getline(f, l);
        std::vector<std::string> el(ne), ml(nm);
        std::vector<double> h_(ne), m_(nm);
        for (int i = 0; i < ne; i++) { std::getline(f, l); size_t t = l.rfind('\t'); el[i] = l.substr(0, t); h_[i] = std::stod(l.substr(t + 1)); }
        for (int i = 0; i < nm; i++) { std::getline(f, l); size_t t = l.rfind('\t'); ml[i] = l.substr(0, t); m_[i] = std::stod(l.substr(t + 1)); }
        std::vector<double> jh(static_cast<size_t>(ne) * NU), jm(static_cast<size_t>(nm) * NU);
        for (auto& v : jh) f >> v;
        for (auto& v : jm) f >> v;
        if (!f) throw std::runtime_error("mellin: truncated " + work + "/jac_cache.txt");
        eqlab = el; mklab = ml; cents0 = cs; H = h_; M = m_; JH = jh; JM = jm;
        neq = ne; rk = r; cut = c; eps_c = ec; eps_f = ef; tau = ta;
        uv = u;
        cut_basis();
        uj = u;
        if (mpirank == 0)
            std::cout << "#  mellin: the first Jacobian from " << work << "/jac_cache.txt (rank " << rk << ", cut " << cut << ", eps_J central "
                      << eps_c << ")\n";
        return true;
    }

    // the column-scaled SVD of d eq / du (write_lin_full's): U_r, r frozen at the first (or the config's)
    void cut_basis()
    {
        const int ne = neq;
        std::vector<double> Dc = col_scale();
        // the SVD of the ne x NU matrix by the eigen-decomposition of A A^T is ill-conditioned; use one-sided Jacobi on A^T (NU x ne)
        const int m = NU, n = ne;
        std::vector<double> At(static_cast<size_t>(m) * n);                 // column j of A^T = row j of A (scaled)
        for (int r = 0; r < n; r++)
            for (int q = 0; q < m; q++) At[static_cast<size_t>(r) * m + q] = JH[static_cast<size_t>(r) * NU + q] / Dc[q];
        std::vector<double> V;
        std::vector<std::pair<double, int>> sv;
        jacobi(At, m, n, V, sv, true);
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
        if (Ur.empty()) precompute_jacobian(u);                 // the first call: the rank and the basis (or the cache's)
        if (uv != u) forward(u);
        rv.assign(static_cast<size_t>(rk + ntik), 0.0);
        for (int c = 0; c < rk; c++) {
            double acc = 0;
            for (int k = 0; k < neq; k++) acc += Ur[static_cast<size_t>(c) * neq + k] * H[k];
            rv[c] = acc;
        }
        for (int k = 0; k < ntik; k++) rv[rk + k] = sw * (u[tik_i[k]] - tik_c[k]) / tik_s[k];   // (sw: the square-root weight, write_lin_full's tik_sw)
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

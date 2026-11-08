// Mellin throat recursion, M2f / M3: the standalone driver.
//   usage: mk_engine BUNDLE_DIR OUT_DIR [TANGENT_OPTIONS]      (TANGENT_OPTIONS: 'key=value;...', mk_run_bundle_tan's; the forward run then
//          the tangents, e.g. 'which=all;tol=1e-13;passes=30;no_stall=1;top_local=1;threads=40')
//          mk_engine --opbuild BUNDLE_DIR OUT_DIR [OPTIONS]   (M3.5: the operator about the bundle's cylinder, mk_op_build's options)
//          mk_engine --mk-dn BUNDLE_DIR OUT_DIR ['h=1e-5']   (G3 (i): the matching keys' n-derivative at the final series, with its FD check)
//          mk_engine --opbuild-dtheta BUNDLE_DIR OUT_DIR 'levels_in=FILE;thetas=u0,c,n,...'   (G1: the operator's derivative in the cylinder)
//          mk_engine --bundle-at BASE_BUNDLE CYL.json K OUT_DIR [OPTIONS]   (M4a: the bundle at a cylinder, the operator rebuilt about it)
//          mk_engine --values BASE_BUNDLE CYL.json K PARAMS OUT_DIR [TANGENT_OPTIONS [OPBUILD_OPTIONS]]   (M4b: the throat at [theta; p] -- the bundle at the
//                    cylinder, the parameters written in, the forward and the fit's outputs: OUT_DIR/run/fit.json; default 'which=none')
//          mk_engine --column BASE_BUNDLE CYL.json K PARAMS KEY DELTA OUT_DIR   (M4c: --values with one grade-0 parameter moved by DELTA)
#include <cstdio>
#include <string>
extern "C" int mk_run_bundle(const char* dir, const char* out, char* err, int errcap);
extern "C" int mk_run_bundle_tan(const char* dir, const char* out, const char* opts, char* err, int errcap);
extern "C" int mk_op_build(const char* dir, const char* out, const char* opts, char* err, int errcap);
extern "C" int mk_mk_dn(const char* dir, const char* out, const char* opts, char* err, int errcap);
extern "C" int mk_op_build_dtheta(const char* dir, const char* out, const char* opts, char* err, int errcap);
extern "C" int mk_set_params(const char* dir, const char* params, char* err, int errcap);
extern "C" int mk_bundle_at(const char* base, const char* cyl, double K, const char* out, const char* opts, char* err, int errcap);
int main(int argc, char** argv) {
    if (argc >= 4 && std::string(argv[1]) == "--opbuild") {        // M3.5: mk_engine --opbuild BUNDLE_DIR OUT_DIR ['e0=..;rows=..;lkdiv=..']
        char err[1024] = {0};
        int rc = mk_op_build(argv[2], argv[3], argc > 4 ? argv[4] : "", err, sizeof err);
        if (rc) std::fprintf(stderr, "mk_engine: %s\n", err);
        return rc;
    }
    if (argc >= 4 && std::string(argv[1]) == "--mk-dn") {          // G3 (i): mk_engine --mk-dn BUNDLE_DIR OUT_DIR ['h=1e-5']
        char err[1024] = {0};
        int rc = mk_mk_dn(argv[2], argv[3], argc > 4 ? argv[4] : "", err, sizeof err);
        if (rc) std::fprintf(stderr, "mk_engine: %s\n", err);
        return rc;
    }
    if (argc >= 5 && std::string(argv[1]) == "--opbuild-dtheta") {  // G1: mk_engine --opbuild-dtheta BUNDLE_DIR OUT_DIR 'levels_in=..;thetas=..'
        char err[1024] = {0};
        int rc = mk_op_build_dtheta(argv[2], argv[3], argv[4], err, sizeof err);
        if (rc) std::fprintf(stderr, "mk_engine: %s\n", err);
        return rc;
    }
    if (argc >= 9 && std::string(argv[1]) == "--column") {         // M4c: mk_engine --column BASE CYL.json K PARAMS KEY DELTA OUT ['opbuild options']
        char err[1024] = {0};
        std::string out(argv[8]), pert = std::string("perturb=") + argv[6] + ":" + argv[7] + (argc > 9 ? std::string(";") + argv[9] : std::string());
        int rc = mk_bundle_at(argv[2], argv[3], std::stod(argv[4]), argv[8], pert.c_str(), err, sizeof err);
        if (!rc) rc = mk_set_params(argv[8], argv[5], err, sizeof err);
        if (!rc) rc = mk_run_bundle_tan(argv[8], (out + "/run").c_str(), "which=none", err, sizeof err);
        if (rc) std::fprintf(stderr, "mk_engine: %s\n", err);
        return rc;
    }
    if (argc >= 7 && std::string(argv[1]) == "--values") {         // M4b: mk_engine --values BASE CYL.json K PARAMS OUT ['tangent options']
        char err[1024] = {0};
        std::string out(argv[6]);
        int rc = mk_bundle_at(argv[2], argv[3], std::stod(argv[4]), argv[6], argc > 8 ? argv[8] : "", err, sizeof err);
        if (!rc) rc = mk_set_params(argv[6], argv[5], err, sizeof err);
        if (!rc) rc = mk_run_bundle_tan(argv[6], (out + "/run").c_str(), argc > 7 ? argv[7] : "which=none", err, sizeof err);
        if (rc) std::fprintf(stderr, "mk_engine: %s\n", err);
        return rc;
    }
    if (argc >= 6 && std::string(argv[1]) == "--bundle-at") {      // M4a: mk_engine --bundle-at BASE CYL.json K OUT ['opbuild options']
        char err[1024] = {0};
        int rc = mk_bundle_at(argv[2], argv[3], std::stod(argv[4]), argv[5], argc > 6 ? argv[6] : "", err, sizeof err);
        if (rc) std::fprintf(stderr, "mk_engine: %s\n", err);
        return rc;
    }
    if (argc != 3 && argc != 4) { std::fprintf(stderr, "usage: %s BUNDLE_DIR OUT_DIR [TANGENT_OPTIONS]\n", argv[0]); return 2; }
    char err[1024] = {0};
    int rc = argc == 4 ? mk_run_bundle_tan(argv[1], argv[2], argv[3], err, sizeof err) : mk_run_bundle(argv[1], argv[2], err, sizeof err);
    if (rc) std::fprintf(stderr, "mk_engine: %s\n", err);
    return rc;
}

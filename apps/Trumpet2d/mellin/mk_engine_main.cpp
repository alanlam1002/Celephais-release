// Mellin throat recursion, M2f / M3: the standalone driver.
//   usage: mk_engine BUNDLE_DIR OUT_DIR [TANGENT_OPTIONS]      (TANGENT_OPTIONS: 'key=value;...', mk_run_bundle_tan's; the forward run then
//          the tangents, e.g. 'which=all;tol=1e-13;passes=30;no_stall=1;top_local=1;threads=40')
//          mk_engine --opbuild BUNDLE_DIR OUT_DIR [OPTIONS]   (M3.5: the operator about the bundle's cylinder, mk_op_build's options)
#include <cstdio>
#include <string>
extern "C" int mk_run_bundle(const char* dir, const char* out, char* err, int errcap);
extern "C" int mk_run_bundle_tan(const char* dir, const char* out, const char* opts, char* err, int errcap);
extern "C" int mk_op_build(const char* dir, const char* out, const char* opts, char* err, int errcap);
int main(int argc, char** argv) {
    if (argc >= 4 && std::string(argv[1]) == "--opbuild") {        // M3.5: mk_engine --opbuild BUNDLE_DIR OUT_DIR ['e0=..;rows=..;lkdiv=..']
        char err[1024] = {0};
        int rc = mk_op_build(argv[2], argv[3], argc > 4 ? argv[4] : "", err, sizeof err);
        if (rc) std::fprintf(stderr, "mk_engine: %s\n", err);
        return rc;
    }
    if (argc != 3 && argc != 4) { std::fprintf(stderr, "usage: %s BUNDLE_DIR OUT_DIR [TANGENT_OPTIONS]\n", argv[0]); return 2; }
    char err[1024] = {0};
    int rc = argc == 4 ? mk_run_bundle_tan(argv[1], argv[2], argv[3], err, sizeof err) : mk_run_bundle(argv[1], argv[2], err, sizeof err);
    if (rc) std::fprintf(stderr, "mk_engine: %s\n", err);
    return rc;
}

// Mellin throat recursion, M2f: the standalone driver.   usage: mk_engine BUNDLE_DIR OUT_DIR
#include <cstdio>
extern "C" int mk_run_bundle(const char* dir, const char* out, char* err, int errcap);
int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s BUNDLE_DIR OUT_DIR\n", argv[0]); return 2; }
    char err[1024] = {0};
    int rc = mk_run_bundle(argv[1], argv[2], err, sizeof err);
    if (rc) std::fprintf(stderr, "mk_engine: %s\n", err);
    return rc;
}

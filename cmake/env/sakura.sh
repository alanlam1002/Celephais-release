#!/bin/bash
# Module environment for Sakura HPC (MPCDF)
# Sourced by compile.sh and apps/submission_scripts/sakura.sh
module purge
module load git/2.50
# gcc/13 (not gcc/14) + impi/2021.17 (not impi/2021.11): must match the GCC
# toolchain/libstdc++ that AthenaK's icpx-driven final link routes through
# (--gcc-toolchain=.../gcc/13.1.0) and its own impi/2021.17. Building
# libcelephais.a with gcc/14 previously produced an ABI mismatch at AthenaK's
# link step (`undefined reference to __cxa_call_terminate`).
module load gcc/13
module load ninja/1.11
module load gsl/2.7
module load impi/2021.17
module load mkl/2025.2

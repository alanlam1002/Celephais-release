# Mellin throat kernel (code round 463 M1, research round 1013): a STANDALONE project -- production targets are untouched (apps/Trumpet2d's
# CMakeLists does not include this directory).  Builds libmellin_kernel.so (C ABI, used by spinning_trumpet/scripts/mk_binding.py).
#   cmake -S apps/Trumpet2d/mellin -B build-mellin -DCMAKE_CXX_COMPILER=g++ (GCC >= 9) && cmake --build build-mellin
cmake_minimum_required(VERSION 3.10)
project(mellin_kernel CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# no -ffast-math: the port must reproduce the Python engine's IEEE arithmetic to ~1e-12
# -fcx-fortran-rules: the plain complex product (numpy's), not C99 Annex G's NaN-recovery call (__muldc3); division keeps its range reduction
# -O3 -march=cascadelake: the cluster's Xeon Gold 6248 (login and p.sakura nodes alike); FMA contraction changes rounding at ~1e-16 only
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=cascadelake -fcx-fortran-rules ${MK_EXTRA_FLAGS}")
add_library(mellin_kernel SHARED mk_kernel.cpp mk_contour.cpp mk_data.cpp)
# M2b onwards: numpy's loops do not fuse multiply-adds, and the recursion's linear algebra is compared with the oracle BITWISE
set_source_files_properties(mk_contour.cpp mk_data.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
target_compile_options(mellin_kernel PRIVATE -Wall -Wextra)
# the GCC 13 runtime linked in: the library loads in the system Python without GCC 13's lib64 on the path
target_link_options(mellin_kernel PRIVATE -static-libstdc++ -static-libgcc)
# M2 (research rounds 1019-1020): the ONE linear-algebra library, MKL LAPACK through LAPACKE.  MK_MKL=numpy (default): the MKL numpy runs here
# (anaconda 2021.05, MKL 2021.2; headers oneAPI 2021.3) -- the oracle's kernels; MK_MKL=kadath: Kadath's oneAPI 2025.2 (the two-library check, M4)
set(MK_MKL "numpy" CACHE STRING "numpy | kadath")
if(MK_MKL STREQUAL "kadath")
  set(MK_MKL_INC /mpcdf/soft/SLE_15/packages/x86_64/intel_oneapi/2025.2/mkl/latest/include)
  set(MK_MKL_LIB /mpcdf/soft/SLE_15/packages/x86_64/intel_oneapi/2025.2/mkl/latest/lib)
else()
  set(MK_MKL_INC /mpcdf/soft/SLE_15/packages/x86_64/intel_oneapi/2021.3/mkl/2021.3.0/include)
  set(MK_MKL_LIB /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2021.05/lib)
endif()
target_include_directories(mellin_kernel PRIVATE ${MK_MKL_INC})
target_link_libraries(mellin_kernel PRIVATE ${MK_MKL_LIB}/libmkl_rt.so)
set_target_properties(mellin_kernel PROPERTIES BUILD_RPATH ${MK_MKL_LIB} INSTALL_RPATH ${MK_MKL_LIB})

#!/bin/bash
# Build the kernel and install it ATOMICALLY: each build copied to a unique file, the symlink build-mellin/install/libmellin_kernel.so swapped
# by rename -- processes that already loaded a library keep their mapping (the build tree's .so is rewritten in place by the linker and must
# never be loaded directly).   usage: apps/Trumpet2d/mellin/install.sh   (from the clone's root)
set -eu
G=${MK_GCC:-/mpcdf/soft/SLE_15/packages/x86_64/gcc/13.1.0/bin}
cmake -S apps/Trumpet2d/mellin -B build-mellin -DCMAKE_CXX_COMPILER=$G/g++ > /dev/null
cmake --build build-mellin
mkdir -p build-mellin/install
h=$(sha256sum build-mellin/libmellin_kernel.so | cut -c1-12)
cp build-mellin/libmellin_kernel.so build-mellin/install/libmellin_kernel.$h.so
ln -sfn libmellin_kernel.$h.so build-mellin/install/.tmp_link && mv -T build-mellin/install/.tmp_link build-mellin/install/libmellin_kernel.so
echo "installed libmellin_kernel.$h.so"

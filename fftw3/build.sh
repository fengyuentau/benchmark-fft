#!/bin/bash

#option (BUILD_SHARED_LIBS "Build shared libraries" ON)
#option (BUILD_TESTS "Build tests" ON)
#
#option (ENABLE_OPENMP "Use OpenMP for multithreading" OFF)
#option (ENABLE_THREADS "Use pthread for multithreading" OFF)
#option (WITH_COMBINED_THREADS "Merge thread library" OFF)
#
#option (ENABLE_FLOAT "single-precision" OFF)
#option (ENABLE_LONG_DOUBLE "long-double precision" OFF)
#option (ENABLE_QUAD_PRECISION "quadruple-precision" OFF)
#
#option (ENABLE_SSE "Compile with SSE instruction set support" OFF)
#option (ENABLE_SSE2 "Compile with SSE2 instruction set support" OFF)
#option (ENABLE_AVX "Compile with AVX instruction set support" OFF)
#option (ENABLE_AVX2 "Compile with AVX2 instruction set support" OFF)
#
#option (DISABLE_FORTRAN "Disable Fortran wrapper routines" OFF)

cmake -G Ninja \
      -B build \
      -D CMAKE_INSTALL_PREFIX=build/install \
      -D CMAKE_BUILD_TYPE=Release \
      -D BUILD_TESTS=OFF \
      fftw-3.3.10

cmake --build build --target install -j4


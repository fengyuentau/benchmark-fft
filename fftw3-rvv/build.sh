#!/bin/bash

#cmake -G Ninja \
#      -B build \
#      -D CMAKE_INSTALL_PREFIX=build/install \
#      -D CMAKE_BUILD_TYPE=Release \
#      -D BUILD_TESTS=OFF \
#      fftw-3.3.10
#
#cmake --build build --target install -j4

if [ ! -d "build" ]; then
    mkdir build
fi

cd fftw-3.3.10
./configure --prefix=/home/opencv/workspace/fytao/benchmark-fft/fftw3-rvv/build --enable-shared --enable-r5v --enable-fma CFLAGS="-O3 -march=rv64gcv" CXXFLAGS="-O3 -march=rv64gcv" FFLAGS="-O3 -march=rv64gcv"
make -j4
make install


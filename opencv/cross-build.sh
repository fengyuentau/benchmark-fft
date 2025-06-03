WORKSPACE=/home/tao/workspace/fytao/k1
#SDK_DIR=$WORKSPACE/spacemit-toolchain-linux-glibc-x86_64-v1.0.4
SDK_DIR=$WORKSPACE/spacemit-toolchain-linux-glibc-x86_64-v1.0.5

TOOLCHAIN_FILE_GCC=$(pwd)/opencv/platforms/linux/riscv64-gcc.toolchain.cmake
TOOLCHAIN_FILE_CLANG=$(pwd)/opencv/platforms/linux/riscv64-clang.toolchain.cmake

#TARGETS_TEST="opencv_test_core opencv_test_calib3d opencv_test_features2d opencv_test_imgproc"
#TARGETS_PERF="opencv_perf_core opencv_perf_calib3d opencv_perf_features2d opencv_perf_imgproc"
TARGETS_TEST="install"
TARGETS_PERF=""

BUILD_TYPE=Release
#BUILD_TYPE=RELWITHDEBINFO
#BUILD_TYPE=Debug

# GCC
cmake -G Ninja -B cross-build-gcc \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_C_COMPILER=$SDK_DIR/bin/riscv64-unknown-linux-gnu-gcc \
      -DCMAKE_CXX_COMPILER=$SDK_DIR/bin/riscv64-unknown-linux-gnu-g++ \
      -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_GCC \
      -DCPU_BASELINE=RVV -DCPU_BASELINE_REQUIRE=RVV -DRISCV_RVV_SCALABLE=ON -DWITH_OPENCL=OFF opencv
# Clang
cmake -G Ninja -B cross-build-clang \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DRISCV_CLANG_BUILD_ROOT=$SDK_DIR \
      -DRISCV_GCC_INSTALL_ROOT=$SDK_DIR \
      -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_CLANG \
      -DCPU_BASELINE=RVV -DCPU_BASELINE_REQUIRE=RVV -DRISCV_RVV_SCALABLE=ON -DWITH_OPENCL=OFF opencv

cmake --build cross-build-gcc --target ${TARGETS_TEST} ${TARGETS_PERF} -j10
cmake --build cross-build-clang --target ${TARGETS_TEST} ${TARGETS_PERF} -j10


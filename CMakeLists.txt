cmake_minimum_required(VERSION 3.24.0)
project(benchmark-fft)

set(KISSFFT_INSTALL_PATH "/home/opencv/workspace/fytao/benchmark-fft/kissfft/build/install" CACHE PATH "Where to look for KissFFT installation")
set(OPENCV_INSTALL_PATH "/home/opencv/workspace/fytao/benchmark-fft/opencv/build/install" CACHE PATH "Where to look for OpenCV installation")
#set(FFTW3_INSTALL_PATH "/home/opencv/workspace/fytao/benchmark-fft/fftw3/build/install" CACHE PATH "Where to look for FFTW3 (official source) installation")
set(FFTW3_INSTALL_PATH "/home/opencv/workspace/fytao/benchmark-fft/fftw3-rvv/build" CACHE PATH "Where to look for FFTW3 (RVV) installation")
set(PFFFT_INSTALL_PATH "/home/opencv/workspace/fytao/benchmark-fft/pffft/pffft" CACHE PATH "Where to look for PFFFT installation")


# Find KISSFFT
#find_package(kissfft CONFIG REQUIRED HINTS ${KISSFFT_INSTALL_PATH})
# Find OpenCV
#find_package(OpenCV REQUIRED HINTS ${OPENCV_INSTALL_PATH})
#find_library(FFTW_LIB fftw3 REQUIRED HINTS ${FFTW3_INSTALL_PATH})
#message(STATUS "fftw found=${FFTW_LIB}, lib=${FFTW_LIB}")


include_directories(${KISSFFT_INSTALL_PATH}/include)
include_directories(${OPENCV_INSTALL_PATH}/include/opencv4)
include_directories(${FFTW3_INSTALL_PATH}/include)
include_directories(${PFFFT_INSTALL_PATH}/)

message(STATUS "${FFTW3_INSTALL_PATH}/lib/libfftw3.so")

add_executable(bench2d 2d.cpp)
add_executable(bench1d 1d.cpp)

target_link_libraries(bench2d ${KISSFFT_INSTALL_PATH}/lib/libkissfft-float.so)
target_link_libraries(bench2d ${OPENCV_INSTALL_PATH}/lib/libopencv_core.so)
target_link_libraries(bench2d ${FFTW3_INSTALL_PATH}/lib/libfftw3.so m)

target_link_libraries(bench1d ${KISSFFT_INSTALL_PATH}/lib/libkissfft-float.so)
target_link_libraries(bench1d ${OPENCV_INSTALL_PATH}/lib/libopencv_core.so)
target_link_libraries(bench1d ${FFTW3_INSTALL_PATH}/lib/libfftw3.so m)
target_link_libraries(bench1d ${PFFFT_INSTALL_PATH}/libpffft.so)


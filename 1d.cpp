#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <complex>
#include <string>

#include "kissfft/kiss_fft.h"
#include "opencv2/opencv.hpp"
#include "fftw3.h"
#include "pffft.h"

constexpr int rows = 1;
constexpr int cols = 2048*3*5;
constexpr int N = rows * cols;
constexpr int loops = 30;

void benchmark_kissfft(const float *input, double *elapsed_time) {
    std::vector<kiss_fft_cpx> input_data(N);
    std::vector<kiss_fft_cpx> output_data(N);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            input_data[r * cols + c].r = input[r * 2 * cols + c];
            input_data[r * cols + c].i = input[r * 2 * cols + c + 1];
        }
    }

    kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, nullptr, nullptr);
    kiss_fft(cfg, input_data.data(), output_data.data());

    for (int i = 0; i < loops; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        kiss_fft(cfg, input_data.data(), output_data.data());
        auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> ms = end - start;
        elapsed_time[i] = ms.count();
    }
}

void benchmark_opencv(const float *input, double *elapsed_time) {
    using namespace cv;

    Mat input_mat(rows, cols, CV_32FC2);
    Mat output_mat(rows, cols, CV_32FC2);

    float* input_data = input_mat.ptr<float>();
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            input_data[r * 2 * cols + c] = input[r * 2 * cols + c];
            input_data[r * 2 * cols + c + 1] = input[r * 2 * cols + c + 1];
        }
    }

    int flags = DFT_ROWS; // DFT_COMPLEX_INPUT
    dft(input_mat, output_mat, flags);

    for (int i = 0; i < loops; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        dft(input_mat, output_mat, flags);
        auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> ms = end - start;
        elapsed_time[i] = ms.count();
    }
}

void benchmark_fftw3(const float *input, double *elapsed_time) {
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            in[r * cols + c][0] = input[r * 2 * cols + c];
            in[r * cols + c][1] = input[r * 2 * cols + c + 1];
        }
    }

    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    for (int i = 0; i < loops; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        fftw_execute(plan);
        auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> ms = end - start;
        elapsed_time[i] = ms.count();
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}

void benchmark_pffft(const float *input, double *elapsed_time) {
    float* in   = (float*)pffft_aligned_malloc(sizeof(float) * N);
    float* out  = (float*)pffft_aligned_malloc(sizeof(float) * N);
    float* work = (float*)pffft_aligned_malloc(sizeof(float) * N);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            in[r * 2 * cols + c]     = input[r * 2 * cols + c];
            in[r * 2 * cols + c + 1] = input[r * 2 * cols + c + 1];
        }
    }

    PFFFT_Setup *setup = pffft_new_setup(N, PFFFT_REAL);
    pffft_transform_ordered(setup, in, out, work, PFFFT_FORWARD);

    for (int i = 0; i < loops; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        pffft_transform_ordered(setup, in, out, work, PFFFT_FORWARD);
        auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> ms = end - start;
        elapsed_time[i] = ms.count();
    }

    pffft_destroy_setup(setup);
    pffft_aligned_free(in);
    pffft_aligned_free(out);
    pffft_aligned_free(work);
}

int main() {
    float input[N*2];

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float v = std::sin(2 * M_PI * r / rows) + std::cos(2 * M_PI * c / cols);
            int idx = r * 2 * cols + c;
            input[idx] = v;
            input[idx+1] = 0.0f;
        }
    }

    // using Transpose2dFunc = void (*)(const uchar*, size_t, uchar*, size_t, int, int);
    using BenchmarkFunc = void (*)(const float *, double *);
    constexpr int nfunc = 4;
    BenchmarkFunc tab[nfunc] = {
        (BenchmarkFunc)benchmark_kissfft,
        (BenchmarkFunc)benchmark_opencv,
        (BenchmarkFunc)benchmark_fftw3,
        (BenchmarkFunc)benchmark_pffft,
    };
    std::vector<std::string> fname_tab {
        "kissfft",
        "opencv",
        "fftw3",
        "pffft",
    };


    for (int i = 0; i < nfunc; i++) {
        BenchmarkFunc func = tab[i];
        if (func == nullptr) { continue; }

        double elapsed_time[loops];
        func(input, elapsed_time);
        double mean = std::accumulate(elapsed_time, elapsed_time + loops, 0.f) / loops;
        printf("%s, mean=%.2fms\n", fname_tab[i].c_str(), mean);
    }

    return 0;
}


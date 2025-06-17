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
constexpr int cols = 1024*3*5; //2048; - fine // 2048*3; - fine // 2048*5; - fine // 2048*3*5 - wrong // 1024*3*5 - wrong // 512*3*5 - correct
constexpr int N = rows * cols;
constexpr int loops = 30; //30;

void benchmark_kissfft(const float *input, float *output, double *elapsed_time) {
    std::vector<kiss_fft_cpx> input_data(N);
    std::vector<kiss_fft_cpx> output_data(N);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            input_data[r * cols + c].r = input[r * 2 * cols + 2*c];
            input_data[r * cols + c].i = input[r * 2 * cols + 2*c + 1];
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

    // copy output
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            output[r * 2 * cols + 2*c]     = output_data[r * cols + c].r;
            output[r * 2 * cols + 2*c + 1] = output_data[r * cols + c].i;
        }
    }
}

void benchmark_opencv(const float *input, float *output, double *elapsed_time) {
    using namespace cv;

    Mat input_mat(rows, cols, CV_32FC2);
    Mat output_mat(rows, cols, CV_32FC2);

    float* input_data = input_mat.ptr<float>();
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            input_data[r * 2 * cols + 2*c]     = input[r * 2 * cols + 2*c];
            input_data[r * 2 * cols + 2*c + 1] = input[r * 2 * cols + 2*c + 1];
        }
    }

    int flags = DFT_COMPLEX_INPUT;
    dft(input_mat, output_mat, flags);

    for (int i = 0; i < loops; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        dft(input_mat, output_mat, flags);
        auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> ms = end - start;
        elapsed_time[i] = ms.count();
    }

    float* output_data = output_mat.ptr<float>();
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            output[r * 2 * cols + 2*c]     = output_data[r * 2 * cols + 2*c];
            output[r * 2 * cols + 2*c + 1] = output_data[r * 2 * cols + 2*c + 1];
        }
    }
}

void benchmark_fftw3(const float *input, float *output, double *elapsed_time) {
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            in[r * cols + c][0] = input[r * 2 * cols + 2*c];
            in[r * cols + c][1] = input[r * 2 * cols + 2*c + 1];
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

    // copy output
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            output[r * 2 * cols + 2*c]     = out[r * cols + c][0];
            output[r * 2 * cols + 2*c + 1] = out[r * cols + c][1];
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}

void benchmark_pffft(const float *input, float *output, double *elapsed_time) {
    float* in   = (float*)pffft_aligned_malloc(sizeof(float) * N * 2);
    float* out  = (float*)pffft_aligned_malloc(sizeof(float) * N * 2);
    float* work = (float*)pffft_aligned_malloc(sizeof(float) * N * 2);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            in[r * 2 * cols + 2*c]     = input[r * 2 * cols + 2*c];
            in[r * 2 * cols + 2*c + 1] = input[r * 2 * cols + 2*c + 1];
        }
    }

    PFFFT_Setup *setup = pffft_new_setup(N, PFFFT_COMPLEX);
    pffft_transform_ordered(setup, in, out, work, PFFFT_FORWARD);

    for (int i = 0; i < loops; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        pffft_transform_ordered(setup, in, out, work, PFFFT_FORWARD);
        auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> ms = end - start;
        elapsed_time[i] = ms.count();
    }

    // copy output
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            output[r * 2 * cols + 2*c]     = out[r * 2 * cols + 2*c];
            output[r * 2 * cols + 2*c + 1] = out[r * 2 * cols + 2*c + 1];
        }
    }

    pffft_destroy_setup(setup);
    pffft_aligned_free(in);
    pffft_aligned_free(out);
    pffft_aligned_free(work);
}

int main() {
    float input[N*2];
    float output[N*2] = {0};
    float output_ocv[N*2] = {0};

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float v = std::sin(2 * M_PI * r / rows) + std::cos(2 * M_PI * c / cols);
            int idx = r * 2 * cols + 2*c;
            input[idx] = v;
            input[idx+1] = 0.0f;
        }
    }

    // using Transpose2dFunc = void (*)(const uchar*, size_t, uchar*, size_t, int, int);
    using BenchmarkFunc = void (*)(const float *, float *,  double *);
    constexpr int nfunc = 4;
    BenchmarkFunc tab[nfunc] = {
        (BenchmarkFunc)benchmark_opencv,
        (BenchmarkFunc)benchmark_kissfft,
        (BenchmarkFunc)benchmark_fftw3,
        (BenchmarkFunc)benchmark_pffft,
    };
    std::vector<std::string> fname_tab {
        "opencv",
        "kissfft",
        "fftw3",
        "pffft",
    };

    // Run OpenCV's FFT first
    BenchmarkFunc func = tab[0];
    double elapsed_time1[loops];
    func(input, output_ocv, elapsed_time1);
    double mean1 = std::accumulate(elapsed_time1, elapsed_time1 + loops, 0.f) / loops;
    printf("%s, mean=%.2fms\n", fname_tab[0].c_str(), mean1);
    //printf("opencv results:");
    //for (int i = 0; i < 2 * N; i++) {
    //    printf(" %f", output_ocv[i]);
    //}
    //printf("\n");

    for (int i = 1; i < nfunc; i++) {
        BenchmarkFunc func = tab[i];
        if (func == nullptr) { continue; }

        memset(output, 0, sizeof(float) * N * 2);
        double elapsed_time[loops];
        func(input, output, elapsed_time);
        double mean = std::accumulate(elapsed_time, elapsed_time + loops, 0.f) / loops;

        // check correctness
        bool correctness = true;
        int index = 0;
        for (; index < 2 * N; index++) {
            if (fabs(output_ocv[index] - output[index]) > 1e-3) {
                correctness = false;
                break;
            }
        }
        printf("%s, mean=%.2fms, correctness=%d", fname_tab[i].c_str(), mean, correctness);
        if (correctness) {
            printf("\n");
        } else {
            printf("\n\tfailed at %d", index);
            index = index / 2;
            printf("\topencv: %f %f\n", output_ocv[index], output_ocv[index+1]);
            printf("\t%s: %f %f\n", fname_tab[i].c_str(), output[index], output[index+1]);
        }
        //printf("%s results:", fname_tab[i].c_str());
        //for (int j = 0; j < 2 * N; j++) {
        //    printf(" %f", output[j]);
        //}
        //printf("\n");
    }

    return 0;
}


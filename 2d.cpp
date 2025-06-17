#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <complex>
#include <string>

#include "kissfft/kiss_fftnd.h"
#include "opencv2/opencv.hpp"
#include "fftw3.h"

constexpr int rows = 4;
constexpr int cols = 4;
constexpr int N = rows * cols;
constexpr int loops = 30;

void benchmark_kissfft(const float *input, float *output, double *elapsed_time) {
    int dims[2] = {rows, cols};
    std::vector<kiss_fft_cpx> input_data(N);
    std::vector<kiss_fft_cpx> output_data(N);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            input_data[r * cols + c].r = input[r * 2 * cols + 2*c];
            input_data[r * cols + c].i = input[r * 2 * cols + 2*c + 1];
        }
    }

    kiss_fftnd_cfg cfg = kiss_fftnd_alloc(dims, 2, 0, 0, 0);
    kiss_fftnd(cfg, input_data.data(), output_data.data());

    for (int i = 0; i < loops; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        kiss_fftnd(cfg, input_data.data(), output_data.data());
        auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> ms = end - start;
        elapsed_time[i] = ms.count();
    }

    // copy output
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
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

    //dft(input_mat, output_mat, DFT_COMPLEX_INPUT);
    dft(input_mat, output_mat);

    for (int i = 0; i < loops; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        //dft(input_mat, output_mat, DFT_COMPLEX_INPUT);
        dft(input_mat, output_mat, DFT_COMPLEX_OUTPUT);
        auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> ms = end - start;
        elapsed_time[i] = ms.count();
    }

    // copy output
    float* output_data = output_mat.ptr<float>();
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
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

    fftw_plan plan = fftw_plan_dft_2d(rows, cols, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    for (int i = 0; i < loops; i++) { // loops = 30
        auto start = std::chrono::high_resolution_clock::now();
        fftw_execute(plan);
        auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> ms = end - start;
        elapsed_time[i] = ms.count();
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    // copy output
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            output[r * 2 * cols + 2*c]     = out[r * cols + c][0];
            output[r * 2 * cols + 2*c + 1] = out[r * cols + c][1];
        }
    }

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

    using BenchmarkFunc = void (*)(const float *, float*,  double *);
    constexpr int nfunc = 3;
    BenchmarkFunc tab[nfunc] = {
        (BenchmarkFunc)benchmark_opencv,
        (BenchmarkFunc)benchmark_kissfft,
        (BenchmarkFunc)benchmark_fftw3,
    };
    std::vector<std::string> fname_tab {
        "opencv",
        "kissfft",
        "fftw3",
    };

    BenchmarkFunc func = tab[0];
    double elapsed_time1[loops];
    func(input, output_ocv, elapsed_time1);
    double mean1 = std::accumulate(elapsed_time1, elapsed_time1 + loops, 0.f) / loops;
    printf("%s, mean=%.2fms\n", fname_tab[0].c_str(), mean1);
    printf("ocv:");
    for (int i = 0; i < N*2; i++) {
        printf(" %f", output_ocv[i]);
    }
    printf("\n");

    for (int i = 1; i < nfunc; i++) {
        BenchmarkFunc func = tab[i];
        if (func == nullptr) { continue; }

        double elapsed_time[loops];
        func(input, output, elapsed_time);
        double mean = std::accumulate(elapsed_time, elapsed_time + loops, 0.f) / loops;

        // verify correctness
        bool correctness = true;
        int j = 0;
        for (; j < N*2; j++) {
            if (fabs(output_ocv[j] - output[j]) > 1e-6) {
                correctness = false;
                break;
            }
        }
        printf("%s, mean=%.2fms, correctness=%d\n", fname_tab[i].c_str(), mean, correctness);
        printf("%s:", fname_tab[i].c_str());
        for (int k = 0; k < N*2; k++) {
            printf(" %f", output[k]);
        }
        printf("\n");
    }

    return 0;
}


#include <iostream>
#include <opencv2/opencv.hpp>
#include <fftw3.h>

int main() {
    const int rows = 4;
    const int cols = 4;

    // Create real input
    cv::Mat input = (cv::Mat_<float>(rows, cols) <<
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16);

    // OpenCV DFT (complex output)
    cv::Mat opencv_input, opencv_dft;
    cv::Mat planes[] = {input, cv::Mat::zeros(rows, cols, CV_32F)};
    cv::merge(planes, 2, opencv_input);
    
    // opencv input data
    const float *p = opencv_input.ptr<const float>();
    //const float *p = input.ptr<const float>();
    printf("opencv input:\n");
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            printf(" %f %f", p[r * cols * 2 + 2*c], p[r*cols*2+2*c+1]);
            //printf(" %f", p[r * cols + c]);
        }
        printf("\n");
    }
    printf("\n");

    cv::dft(opencv_input, opencv_dft, cv::DFT_COMPLEX_OUTPUT);



    // FFTW3 DFT
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);

    // Fill FFTW input with same data
    for (int i = 0; i < rows * cols; ++i) {
        in[i][0] = input.at<float>(i / cols, i % cols); // real
        in[i][1] = 0.0f; // imag
    }

    // FFTW input
    printf("fftw input:\n");
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            printf(" %f %f", in[r*cols+c][0], in[r*cols+c][1]);
        }
        printf("\n");
    }
    printf("\n");


    fftw_plan plan = fftw_plan_dft_2d(rows, cols, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Compare
    std::cout << "Comparing OpenCV DFT and FFTW3 DFT results:\n";
    bool match = true;
    for (int i = 0; i < rows * cols; ++i) {
        cv::Vec2f val = opencv_dft.at<cv::Vec2f>(i / cols, i % cols);
        float opencv_real = val[0];
        float opencv_imag = val[1];

        float fftw_real = out[i][0];
        float fftw_imag = out[i][1];

        float tol = 1e-3;
        if (std::abs(opencv_real - fftw_real) > tol || std::abs(opencv_imag - fftw_imag) > tol) {
            std::cout << "Mismatch at (" << i / cols << ", " << i % cols << "): "
                      << "OpenCV = (" << opencv_real << ", " << opencv_imag << "), "
                      << "FFTW = (" << fftw_real << ", " << fftw_imag << ")\n";
            match = false;
        }
    }

    if (match)
        std::cout << "✅ Results match within tolerance.\n";

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return 0;
}


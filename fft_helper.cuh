//
// Created by dynamic on 6/1/19.
//

#ifndef NUFFT_FFT_HELPER_HPP
#define NUFFT_FFT_HELPER_HPP
#include <vector>
#include <complex>
#include <cufft.h>

typedef std::complex<float> Complex;

const float PI = std::acos(-1);
const Complex J(0, 1);

// Parameters used for the convolution based fast NUFFT.
struct Param {
    int Msp; // Half of gridding kernel's size.
    int Mr; // Oversampled grid size
    float tau; // Normalization width for the gaussian.
};

std::vector<float> getFreq(float df, int M);

struct Param computeGridParams(int M, float eps);

std::vector<Complex> fftGpu(std::vector<float> inp, const int iflag);

std::vector<Complex> fftCpu(std::vector<float> inp, int iflag);

void callCufft(cufftComplex *dev_in, cufftComplex *dev_out, int n, int iflag);
#endif //NUFFT_FFT_HELPER_HPP

//
// Created by dynamic on 6/1/19.
//

#ifndef NUFFT_FFT_HELPER_HPP
#define NUFFT_FFT_HELPER_HPP
#include <vector>
#include <complex>

typedef std::complex<float> Complex;

const float PI = std::acos(-1);
const Complex J(0, 1);

// Parameters used for the convolution based fast NUFFT.
struct Param {
    int Msp;
    int Mr;
    float tau;
};

std::vector<float> getFreq(float df, int M);

std::vector<Complex> fftGpu(std::vector<float> inp, const int iflag);

std::vector<Complex> fftCpu(std::vector<float> inp, int iflag);
#endif //NUFFT_FFT_HELPER_HPP

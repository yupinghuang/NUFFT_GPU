#ifndef  CPU_NUFFT_HPP
#define CPU_NUFFT_HPP

#include "fft_helper.cuh"

//
// Parameters:
// vector<float> x, vector<float> y: the input x and y
// int M: number of frequencies
// bool gpuFFT: whether to use the GPU for FFT
// float df: frequency spacing
// float eps: gridding accuracy. Translate to gridding kernel size.
//
std::vector<Complex> nufftCpu(std::vector<float> x, std::vector<float> y, int M, bool gpuFFT=false,
        float df=1.0, float eps=1e-15, int iflag=-1);

//
// Parameters:
// vector<float> x, vector<float> y: the input x and y
// int M: number of frequencies
// float df: frequency spacing
//
std::vector<Complex> nudft(std::vector<float> x, std::vector<float> y, int M, float df);
#endif
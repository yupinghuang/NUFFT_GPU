#ifndef  CPU_NUFFT_HPP
#define CPU_NUFFT_HPP

#include "fft_helper.cuh"

std::vector<Complex> nufftCpu(std::vector<float> x, std::vector<float> y, int M, bool gpuFFT=false,
        float df=1.0, float eps=1e-15, int iflag=-1);

std::vector<Complex> nudft(std::vector<float> x, std::vector<float> y, int M, float df);
#endif
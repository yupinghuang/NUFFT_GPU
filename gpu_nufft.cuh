#ifndef NUFFT_GPU_NUFFT_CUH
#define NUFFT_GPU_NUFFT_CUH

#include "fft_helper.cuh"

void queryGpus();

std::vector<Complex> nufftGpu(std::vector<float> x, std::vector<float> y, int M,
                         float df=1.0, float eps=1e-15, int iflag=-1);

#endif //NUFFT_GPU_NUFFT_CUH

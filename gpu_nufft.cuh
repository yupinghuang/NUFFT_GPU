#ifndef NUFFT_GPU_NUFFT_CUH
#define NUFFT_GPU_NUFFT_CUH

#include "fft_helper.cuh"

void cudaCallProdScaleKernel(uint blocks, uint threadsPerBlock);

void queryGpus();

#endif //NUFFT_GPU_NUFFT_CUH

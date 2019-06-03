#ifndef NUFFT_GPU_NUFFT_CUH
#define NUFFT_GPU_NUFFT_CUH

#include "fft_helper.cuh"

#define MAX_THREADS 1024
#define MAX_SHMEM 49152

// For testing the #pragma unroll macro
#define KERNEL_SIZE 29

void queryGpus();

enum GriddingImplementation { NAIVE, PARALLEL, SHMEM, ILP };

std::vector<Complex> nufftGpu(std::vector<float> x, std::vector<float> y, int M, GriddingImplementation type,
                         float df=1.0, float eps=1e-15, int iflag=-1);

#endif //NUFFT_GPU_NUFFT_CUH

#ifndef NUFFT_GPU_NUFFT_CUH
#define NUFFT_GPU_NUFFT_CUH

#include "fft_helper.cuh"

#define MAX_THREADS 1024

// For testing the #pragma unroll macro
#define KERNEL_SIZE 29

void queryGpus();

enum GriddingImplementation { NAIVE, PARALLEL, SHMEM, ILP };

//
// Parameters:
// vector<float> x, vector<float> y: the input x and y
// int M: number of frequencies
// GriddingImplementation type: type of gridding implementation desired
// float df: frequency spacing
// float eps: gridding accuracy. Translate to gridding kernel size.
// int iflag: the sign in front of the i in the complex exponential. -1 is a forward transform. +1 is not implemented.
//
std::vector<Complex> nufftGpu(std::vector<float> x, std::vector<float> y, int M, GriddingImplementation type,
                         float df=1.0, float eps=1e-15, int iflag=-1);

#endif //NUFFT_GPU_NUFFT_CUH

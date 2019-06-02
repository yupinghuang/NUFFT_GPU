#include <cuda_runtime.h>
#include <cufft.h>
#include <math_constants.h>
#include <iostream>
#include <stdio.h>

#include "helper_cuda.h"
#include "fft_helper.cuh"
#include "gpu_nufft.cuh"

using std::vector;

int MAX_THREADS = 1024;

__global__
void
naiveGriddingKernel(float *dev_x, float *dev_y, cufftComplex *dev_ftau, float df, float tau,
                    int N, int Mr, int kernelSize) {
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    float hx = 2 * CUDART_PI_F / Mr;
    // Calculate the kernel
    if (threadId < N) {
        float xi = fmodf(dev_x[threadId] * df, 2.0 * CUDART_PI_F);
        int m = 1 + ((int) (xi / hx));
        for (int j = 0; j < kernelSize; ++j) {
            int mmj = -(kernelSize / 2) + j;
            // TODO try __expf
            float kj = expf(-0.25f * powf(xi - hx * (m + mmj), 2) / tau);
            // Assuming Mr > Msp i.e. grid size greater than half of the kernel size
            // TODO modulo instructions are apparently expensive.
            // TODO use reduction for this step instead of atomicAdd.
            int index = (m + mmj + Mr) % Mr;
            atomicAdd(&(dev_ftau[index].x), dev_y[threadId] * kj);
        }
    }
}


__global__
void postProcessingKernel(cufftComplex *dev_Ftau, cufftComplex *dev_yt, float* dev_kvec,
        int N, int M, int Mr, float tau) {
    // M/2 threads are needed here.
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    float t1x, t1y, t2x, t2y;
    if (threadId < M/2) {
        t1x = dev_Ftau[threadId - M/2 + Mr].x / Mr;
        t1y = dev_Ftau[threadId - M/2 + Mr].y / Mr;
        t2x = dev_Ftau[threadId].x / Mr;
        t2y = dev_Ftau[threadId].y / Mr;
        dev_yt[threadId].x = (1.0 / N) * sqrtf(CUDART_PI_F/tau) * expf(tau * powf(dev_kvec[threadId], 2.0)) * t1x;
        dev_yt[threadId].y = (1.0 / N) * sqrtf(CUDART_PI_F/tau) * expf(tau * powf(dev_kvec[threadId], 2.0)) * t1y;
        dev_yt[threadId + M/2].x = (1.0 / N) * sqrtf(CUDART_PI_F/tau) * expf(tau * powf(dev_kvec[threadId + M/2], 2.0)) * t2x;
        dev_yt[threadId + M/2].y = (1.0 / N) * sqrtf(CUDART_PI_F/tau) * expf(tau * powf(dev_kvec[threadId + M/2], 2.0)) * t2y;
    }
}


vector<Complex> nufftGpu(const vector<float> x, const vector<float> y, const int M,
                         const float df, const float eps, const int iflag) {
    std::cout << "Starting GPU NUFFT...\n";
    struct Param param = computeGridParams(M, eps);
    int N = x.size();
    int kernelSize = 2 * param.Msp + 1;
    std::cout << "Gridding kernel has size " << kernelSize << "; grid has size " << param.Mr << ".\n";

    float *dev_x;
    float *dev_y;
    float *dev_kvec;
    cudaEvent_t start1, stop1, start2, stop2;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cufftComplex *dev_ftau;
    cufftComplex *dev_Ftau;
    cufftComplex *dev_yt;

    // Allocate memory
    CUDA_CALL(cudaMalloc((void **) &dev_x, N * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &dev_y, N * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &dev_ftau, param.Mr * sizeof(cufftComplex)));

    // Copy & initialize
    CUDA_CALL(cudaMemset(dev_ftau, 0, param.Mr * sizeof(cufftComplex)));
    CUDA_CALL(cudaMemcpy(dev_x, x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_y, y.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Construct the convolved grid
    int blockSize = 1024;
    int gridSize = N / blockSize + 1;
    cudaEventRecord(start1);
    naiveGriddingKernel<<<gridSize, blockSize>>>(dev_x, dev_y, dev_ftau, df, param.tau, N, param.Mr, kernelSize);
    cudaEventRecord(stop1);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaFree(dev_x));
    CUDA_CALL(cudaFree(dev_y));


    // FFT step
    CUDA_CALL(cudaMalloc((void **) &dev_Ftau, param.Mr * sizeof(cufftComplex)));
    callCufft(dev_ftau, dev_Ftau, param.Mr, iflag);
    CUDA_CALL(cudaFree(dev_ftau));

    // Reordering and de-gridding.
    CUDA_CALL(cudaMalloc((void **) &dev_yt, M * sizeof(cufftComplex)));
    CUDA_CALL(cudaMalloc((void **) &dev_kvec, M * sizeof(cufftComplex)));
    vector<float> k = getFreq(df, M);
    CUDA_CALL(cudaMemcpy(dev_kvec, k.data(), M * sizeof(float), cudaMemcpyHostToDevice));


    gridSize = (M / 2 / MAX_THREADS) + 1;
    blockSize = gridSize == 1 ? (M / 2) : MAX_THREADS;
    cudaEventRecord(start2);
    postProcessingKernel<<<gridSize, blockSize>>>(dev_Ftau, dev_yt, dev_kvec, N, M, param.Mr, param.tau);
    cudaEventRecord(stop2);
    CUDA_CALL(cudaPeekAtLastError());
    cudaEventSynchronize(stop1);
    cudaEventSynchronize(stop2);
    float ms1, ms2;
    cudaEventElapsedTime(&ms1, start1, stop1);
    cudaEventElapsedTime(&ms2, start2, stop2);
    std::cout << "Gridding kernel took " << ms1 <<  "ms; Post-processing kernel took " << ms2 << "ms.\n";
    vector<Complex> yt = vector<Complex>(M);
    CUDA_CALL(cudaMemcpy(yt.data(), dev_yt, M * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(dev_Ftau));
    CUDA_CALL(cudaFree(dev_yt));
    CUDA_CALL(cudaFree(dev_kvec));
    std::cout << "GPU NUFFT Completed\n";
    return yt;
}


// CUDA GPU Device Query code from the course website.
void queryGpus() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Major revision number: %d\n", prop.major);
        printf("  Minor revision number: %d\n", prop.minor);
        printf("  Total shared memory per block (Bytes): %u\n",  prop.sharedMemPerBlock);
        printf("  Total registers per block: %d\n",  prop.regsPerBlock);
        printf("  Warp size: %d\n",  prop.warpSize);
        printf("  Maximum threads per block: %d\n",  prop.maxThreadsPerBlock);
        printf("  Clock rate (KHz): %d\n",  prop.clockRate);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Total VRAM (Bytes): %u\n",  prop.totalGlobalMem);
        printf("  Total constant memory (Bytes): %u\n",  prop.totalConstMem);
        printf("  Number of SMs: %d\n", prop.multiProcessorCount);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Concurrent copy and execution: %s\n",  (prop.deviceOverlap ? "Yes" : "No"));
        printf("  Concurrent kernels: %s\n",  (prop.concurrentKernels ? "Yes" : "No"));
        printf("  Kernel execution timeout: %s\n",  (prop.kernelExecTimeoutEnabled ? "Yes" : "No"));
    }

}

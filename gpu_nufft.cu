#include <cuda_runtime.h>
#include <cufft.h>
#include <math_constants.h>
#include <iostream>
#include <stdio.h>

#include "helper_cuda.h"
#include "fft_helper.cuh"
#include "gpu_nufft.cuh"

using std::vector;


__global__
void
naiveGriddingKernel(float *dev_x, float *dev_y, cufftComplex *dev_ftau, float df, float tau,
                    int N, int Mr, int kernelSize) {
    // Parallelization of the CPU code.
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    float hx = 2 * CUDART_PI_F / Mr;
    if (threadId < N) {
        float xi = fmodf(dev_x[threadId] * df, 2.0 * CUDART_PI_F);
        int m = 1 + ((int) (xi / hx));
        for (int j = 0; j < kernelSize; ++j) {
            int mmj = -(kernelSize / 2) + j;
            float kj = expf(-0.25f * powf(xi - hx * (m + mmj), 2) / tau);
            // Assuming Mr > Msp i.e. grid size greater than half of the kernel size which is v reasonable
            int index = (m + mmj + Mr) % Mr;
            atomicAdd(&(dev_ftau[index].x), dev_y[threadId] * kj);
        }
    }
}


__global__
void
instructionOptimizedGriddingKernel(float *dev_x, float *dev_y, cufftComplex *dev_ftau, float df, float tau,
                    int N, int Mr, int kernelSize) {
    // same as NAIVE but with intrinsics and ILP for speeding up. The limitation is that I hard-coded the kernel-size.
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    float hx = 2 * CUDART_PI_F / Mr;
    // Hopefully this read is coalesced?
    float x = dev_x[threadId];
    float y = dev_y[threadId];
    float xi = fmodf(x * df, 2.0 * CUDART_PI_F);
    int m = 1 + ((int) (xi / hx));
    int kernelSizeOverTwo = KERNEL_SIZE / 2;
    // Loop unrolling.
#pragma unroll (29)
    for (int j = 0; j < KERNEL_SIZE; ++j) {
        if (threadId < N) {
            // Removed instruction dependency; used instricsics __powf, __expf.
            atomicAdd(&(dev_ftau[(m - kernelSizeOverTwo + j + Mr) % Mr].x), y * (
                    __expf(-0.25f * __powf(xi - hx * (m + j - (kernelSizeOverTwo)), 2) / tau)));
        }
    }
}


__global__
void
shmemGriddingKernel(float *dev_x, float *dev_y, cufftComplex *dev_ftau, float df, float tau,
                                int N, int Mr, int kernelSize) {
    // Store the grid in shared memory and then sum the grid back to device memory.
    extern __shared__ float shmem[];
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    float x = dev_x[threadId];
    float y = dev_y[threadId];

    while (tid < Mr) {
        shmem[tid] = 0.0;
        tid += blockDim.x;
    }

    float hx = 2 * CUDART_PI_F / Mr;
    if (threadId < N) {
        float xi = fmodf(x * df, 2.0 * CUDART_PI_F);
        int m = 1 + ((int) (xi / hx));
        for (int j = 0; j < kernelSize; ++j) {
            int mmj = -(kernelSize / 2) + j;
            float kj = expf(-0.25f * powf(xi - hx * (m + mmj), 2) / tau);
            // Assuming Mr > Msp i.e. grid size greater than half of the kernel size which is v reasonable
            int index = (m + mmj + Mr) % Mr;
            // Accumulate on shared memory
            atomicAdd(&(shmem[index]), y * kj);
        }
    }
    __syncthreads();
    tid = threadIdx.x;
    while (tid < Mr) {
        atomicAdd(&(dev_ftau[tid].x), shmem[tid]);
        tid += blockDim.x;
    }
}


__global__
void
triedToBeParallelGriddingKernel(float *dev_x, float *dev_y, cufftComplex *dev_ftau, float df, float tau,
                                int N, int Mr, int kernelSize) {
    // blockIdx.y is the index to dev_ftau that this set of blocks is responsible for accumulating.
    // The hope is that the threads would run in parallel and we can speed up by throwing away work.
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float sumReal[];
    sumReal[tid] = 0.0;
    __syncthreads();
    float hx = 2 * CUDART_PI_F / Mr;
    float x = dev_x[threadId];
    float y = dev_y[threadId];
    if (threadId < N) {
        float xi = fmodf(x * df, 2.0 * CUDART_PI_F);
        int m = 1 + ((int) (xi / hx));
        // TODO can do some math so that we only have at most three iterations here.
        for (int j = 0; j < kernelSize; ++j) {
            int mmj = -(kernelSize / 2) + j;
            int index = (m + mmj + Mr) % Mr;
            if (index == blockIdx.y) {
                float kj = expf(-0.25f * powf(xi - hx * (m + mmj), 2) / tau);
                sumReal[tid] = y * kj;
            }
        }
    }
    __syncthreads();

    // sum with reduction
    for (uint s=blockDim.x/2; s>0; s >>= 1) {
        if (tid < s) {
            sumReal[tid] += sumReal[tid+s];
        }
        __syncthreads();
    }

    // Write back to dev_ftau
    if (tid == 0) {
        atomicAdd(&dev_ftau[blockIdx.y].x, sumReal[tid]);
    }
}


__global__
void postProcessingKernel(cufftComplex *dev_Ftau, cufftComplex *dev_yt, float* dev_kvec,
        int N, int M, int Mr, float tau) {
    // M/2 threads are needed here.
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    float t1x = dev_Ftau[threadId - M/2 + Mr].x / Mr;
    float t1y = dev_Ftau[threadId - M/2 + Mr].y / Mr;
    float t2x = dev_Ftau[threadId].x / Mr;
    float t2y = dev_Ftau[threadId].y / Mr;
    if (threadId < M/2) {
        // Undo the gridding kernel. Also normalize by 1/N
        dev_yt[threadId].x = (1.0 / N) * sqrtf(CUDART_PI_F/tau) * expf(tau * powf(dev_kvec[threadId], 2.0)) * t1x;
        dev_yt[threadId].y = (1.0 / N) * sqrtf(CUDART_PI_F/tau) * expf(tau * powf(dev_kvec[threadId], 2.0)) * t1y;
        dev_yt[threadId + M/2].x = (1.0 / N) * sqrtf(CUDART_PI_F/tau) *
                expf(tau * powf(dev_kvec[threadId + M/2], 2.0)) * t2x;
        dev_yt[threadId + M/2].y = (1.0 / N) * sqrtf(CUDART_PI_F/tau) *
                expf(tau * powf(dev_kvec[threadId + M/2], 2.0)) * t2y;
    }
}


vector<Complex> nufftGpu(const vector<float> x, const vector<float> y, const int M, const GriddingImplementation type,
                         const float df, const float eps, const int iflag) {
    std::cout << "Starting GPU NUFFT...\n";
    struct Param param = computeGridParams(M, eps);
    int N = x.size();
    int kernelSize = 2 * param.Msp + 1;
    std::cout << "Gridding kernel has size " << kernelSize << "; grid has size " << param.Mr << ".\n";

    float *dev_x;
    float *dev_y;
    float *dev_kvec;
    cufftComplex *dev_ftau;
    cufftComplex *dev_Ftau;
    cufftComplex *dev_yt;

    // For timing the gridding kernel and the postProcessing kernel.
    cudaEvent_t start1, stop1, start2, stop2;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    CUDA_CALL(cudaMalloc((void **) &dev_x, N * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &dev_y, N * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &dev_ftau, param.Mr * sizeof(cufftComplex)));

    CUDA_CALL(cudaMemset(dev_ftau, 0, param.Mr * sizeof(cufftComplex)));
    CUDA_CALL(cudaMemcpy(dev_x, x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_y, y.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Construct the convolved grid
    int blockSize = 1024;
    int gridSize = N / blockSize + 1;
    cudaEventRecord(start1);
    if (type == PARALLEL) {
        std::cout << "Using PARALLEL gridding scheme (this one is slower than NAIVE).\n";
        std::cout << "!!!!!!!!!!!WARNING: UNBEARABLY SLOW!!!!!!!!!!!!!!!\n";
        dim3 grid(gridSize, param.Mr);
        printf("For gridding we have %d x %d blocks. Using the extra blockDim to parallelized sum for the grid\n",
                grid.x, grid.y);
        triedToBeParallelGriddingKernel <<<grid, blockSize, blockSize * sizeof(float)>>>(dev_x, dev_y, dev_ftau,
                df, param.tau, N, param.Mr, kernelSize);
    } else if (type == NAIVE) {
        std::cout << "Using NAIVE gridding scheme.\n";
        std::cout << "For gridding we have " << gridSize << " blocks.\n";
        naiveGriddingKernel<<<gridSize, blockSize>>>(dev_x, dev_y, dev_ftau, df, param.tau, N, param.Mr, kernelSize);
    } else if (type == SHMEM) {
        std::cout << "Using SHMEM gridding scheme.\n";
        std::cout << "For gridding we have " << gridSize << " blocks.\n";
        shmemGriddingKernel<<<gridSize, blockSize, param.Mr * sizeof(float)>>>(dev_x, dev_y, dev_ftau, df,
                param.tau, N, param.Mr, kernelSize);

    } else if (type == ILP) {
        std::cout << "Using ILP gridding scheme i.e. naive with instruction optimizations and loop unrolling.\n";
        std::cout << "!!!!!!!!!!!Note that the kernel size 29 is hard-coded!!!!!!!!!!!!!!!\n";
        std::cout << "For gridding we have " << gridSize << " blocks.\n";
        instructionOptimizedGriddingKernel<<<gridSize, blockSize>>>(dev_x, dev_y, dev_ftau, df,
                param.tau, N, param.Mr, kernelSize);
    }

    cudaEventRecord(stop1);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaFree(dev_x));
    CUDA_CALL(cudaFree(dev_y));


    // FFT step
    CUDA_CALL(cudaMalloc((void **) &dev_Ftau, param.Mr * sizeof(cufftComplex)));
    callCufft(dev_ftau, dev_Ftau, param.Mr, iflag);
    CUDA_CALL(cudaFree(dev_ftau));

    // Reordering and de-gridding (post-processing).
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
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
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
    printf("Printing device information just for my sake...\n");
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Major revision number: %d\n", prop.major);
        printf("  Minor revision number: %d\n", prop.minor);
        // Added this
        printf("  Maximum number of blocks: %d\n", prop.maxGridSize[0]);
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

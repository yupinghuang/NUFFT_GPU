#include <iostream>
#include <cstdlib>
#include <chrono>
#include <math.h>
#include <vector>
#include <complex>
#include <fftw3.h>

#include "ta_utilities.hpp"
#include "cpu_nufft.hpp"
#include "gpu_nufft.cuh"

using std::vector;

void checkArgs(int argc, char **argv) {
   if (argc != 3) {
       std::cerr << "Arguments: <testcase> \n";
       exit(EXIT_FAILURE);
   }
}

void compareComplexVectors(vector<Complex> v1, vector<Complex> v2, float eps) {
    bool correct = true;
    std::cout << "Comparing results...\n";
    for(int i=0; i < v1.size(); ++i) {
        Complex delta = v1[i] - v2[i];
        if (delta.real() > eps || delta.imag() > eps) {
            std::cerr << "Different results at " << i << " " << v1[i] << " " << v2[i] << "\n";
            correct = false;
        }
    }
    if (correct) std::cout << "The two results are the same.\n";
}


//
// Compare CPU and GPU implementation results.
//
void cpuGpu1DTest() {
}

//
// Compare CPU 1D Direct Fourier Transform and Fast-Fourier Transform with Gaussian interpolation
//
void runCpu1DTest(int testSize, int M) {
    if (M % 2 == 1) {
        std::cerr << "I haven't figured out how to do odd number of frequency bins yet.\n";
        exit(1);
    }
    // Compare CPU 1D NUDFT and NUFFT implementations.
    std::cout << "Comparing 1D CPU Non-uniform FFT and DFT with " << testSize <<" samples and " << M <<
              " frequency bins...\n";
    std::cout << "The signal contains 3 different frequencies and a bit of noise.\n";
    vector<float> x(testSize);
    vector<float> y(testSize);
    for (int i=0; i< testSize; ++i) {
        x[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(50)));;
        y[i] = sin(x[i]) + sin(2.0f * x[i]) + sin(7.0f * x[i]) +
                static_cast<float>(rand()) / static_cast<float>(RAND_MAX/(50));
    }
    auto start = std::chrono::high_resolution_clock::now();
    vector<Complex> ans = nudft(x, y, M, 1.0);
    auto stop1 = std::chrono::high_resolution_clock::now();
    vector<Complex> ans2 = nufftCpu(x, y, M);
    auto stop2 = std::chrono::high_resolution_clock::now();
    compareComplexVectors(ans, ans2, 1e-4);
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start);
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - stop1);
    std::cout << "CPU NUDFT completed in " << duration1.count() << " ms\n";
    std::cout << "CPU NUFFT completed in " << duration2.count() << " ms\n";
}


//
// Compare CPU 1D Direct Fourier Transform and Fast-Fourier Transform with Gaussian interpolation
//
void runCpuHybridTest(int testSize, int M) {
    if (M % 2 == 1) {
        std::cerr << "I haven't figured out how to do odd number of frequency bins yet.\n";
        exit(1);
    }
    // Compare CPU 1D NUDFT and NUFFT implementations.
    std::cout << "Comparing 1D CPU and hybrid Non-uniform FFT with " << testSize <<" samples and " << M <<
              " frequency bins...\n";
    std::cout << "The signal contains 3 different frequencies and a bit of noise.\n";
    vector<float> x(testSize);
    vector<float> y(testSize);
    for (int i=0; i< testSize; ++i) {
        x[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(50)));
        y[i] = sin(x[i]) + sin(2.0f * x[i]) + sin(7.0f * x[i]) +
               static_cast<float>(rand()) / static_cast<float>(RAND_MAX/(50));
    }
    auto start = std::chrono::high_resolution_clock::now();
    vector<Complex> ans = nufftCpu(x, y, M);
    auto stop1 = std::chrono::high_resolution_clock::now();
    vector<Complex> ans2 = nufftCpu(x, y, M, true);
    auto stop2 = std::chrono::high_resolution_clock::now();
    compareComplexVectors(ans, ans2, 1e-4);
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start);
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - stop1);
    std::cout << "Pure CPU NUFFT completed in " << duration1.count() << " ms\n";
    std::cout << "CPU Gridding with GPU FFT completed in " << duration2.count() << " ms\n";
}

void printDividingLine() { std::cout << "=========================================================================\n"; }

void runTest(int argc, char **argv) {
    if (argv[1] == "cpu-1d") {
        runCpu1DTest(10000, 20);
    }
}


int main(int argc, char **argv) {
    //TA_Utilities::select_coldest_GPU();
    //int max_time_allowed_in_seconds = 300;
    //TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);
    queryGpus();
    srand (time(NULL));
    runCpu1DTest(100000, 100);
    printDividingLine();
    runCpuHybridTest(50000000, 100);
    return 0;
}

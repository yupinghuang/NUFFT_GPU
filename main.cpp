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
using std::string;

void printHelpAndExit() {
    std::cerr << "Arguments: <testcase> <numSamples> <numFreqBin>\n";
    std::cerr << "numFreqBin must be an even number. Parameters are ignored when testcase is demo\n";
    std::cerr << "Test cases are: demo, cpu, hybrid, gpu-naive, gpu-slow.\n";
    exit(EXIT_FAILURE);
}

string griddingImplementationToString(GriddingImplementation type) {
    string ans;
    switch (type) {
        case NAIVE: ans = "NAIVE"; break;
        case ILP: ans = "ILP"; break;
        case PARALLEL: ans = "PARALLEL"; break;
        default: ans= "";
    }
    return ans;
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
// Compare CPU and GPU NUFFT with Gaussian gridding.
//
void runCpuGpu1DTest(int testSize, int M, GriddingImplementation type) {
    std::cout << "Comparing 1D CPU and GPU " << griddingImplementationToString(type) <<
              " Non-uniform FFT with " << testSize <<" samples and " << M <<
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
    vector<Complex> ans2 = nufftGpu(x, y, M, type);
    auto stop2 = std::chrono::high_resolution_clock::now();
    compareComplexVectors(ans, ans2, 1e-4);
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start);
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - stop1);
    std::cout << "CPU NUFFT completed in " << duration1.count() << " ms\n";
    std::cout << "GPU " << griddingImplementationToString(type) << " NUFFT completed in "
            << duration2.count() << " ms\n";
}

//
// Compare CPU 1D Direct Fourier Transform and Fast-Fourier Transform with Gaussian interpolation
//
void runCpu1DTest(int testSize, int M) {
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
    vector<Complex> ans2 = nufftCpu(x, y, M, false, 1.0);
    auto stop2 = std::chrono::high_resolution_clock::now();
    compareComplexVectors(ans, ans2, 1e-4);
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start);
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - stop1);
    std::cout << "CPU NUDFT completed in " << duration1.count() << " ms\n";
    std::cout << "CPU NUFFT completed in " << duration2.count() << " ms\n";
}


//
// Compare CPU NUFFT with the FFT step done with CUFFT and FFTW.
//
void runCpuHybridTest(int testSize, int M) {
    std::cout << "Comparing 1D CPU and hybrid Non-uniform FFT with " << testSize <<" samples and " << M <<
              " frequency bins...\n";
    std::cout << "The signal contains 3 different frequencies and a bit of noise.\n";
    vector<float> x(testSize);
    vector<float> y(testSize);
    for (int i=0; i< testSize; ++i) {
        x[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(9)));
        y[i] = sin(x[i]) + sin(2.0f * x[i]) + sin(7.0f * x[i]) +
               static_cast<float>(rand()) / static_cast<float>(RAND_MAX/(9));
    }
    auto start = std::chrono::high_resolution_clock::now();
    vector<Complex> ans = nufftCpu(x, y, M, false, 1.0);
    auto stop1 = std::chrono::high_resolution_clock::now();
    vector<Complex> ans2 = nufftCpu(x, y, M, true, 1.0);
    auto stop2 = std::chrono::high_resolution_clock::now();
    compareComplexVectors(ans, ans2, 1e-4);
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start);
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - stop1);
    std::cout << "Pure CPU NUFFT completed in " << duration1.count() << " ms\n";
    std::cout << "CPU Gridding with GPU FFT completed in " << duration2.count() << " ms\n";
}

// Test various optimization schemes on GPU against the NAIVE implementation
void runGpu1DTest(int testSize, int M, GriddingImplementation type) {
    std::cout << "Comparing 1D GPU NAIVE and GPU " << griddingImplementationToString(type) <<
              " Non-uniform FFT with " << testSize <<" samples and " << M <<
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
    vector<Complex> ans = nufftGpu(x, y, M, NAIVE);
    auto stop1 = std::chrono::high_resolution_clock::now();
    vector<Complex> ans2 = nufftGpu(x, y, M, type);
    auto stop2 = std::chrono::high_resolution_clock::now();
    compareComplexVectors(ans, ans2, 1e-4);
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start);
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - stop1);
    std::cout << "GPU NAIVE NUFFT completed in " << duration1.count() << " ms\n";
    std::cout << "GPU" << griddingImplementationToString(type) << " NUFFT completed in "
              << duration2.count() << " ms\n";
}

void printDividingLine() { std::cout << "=========================================================================\n"; }


void runTest(int argc, char *argv[]) {
    if (argc < 2) printHelpAndExit();
    std::string testCase = argv[1];
    if (testCase!="demo" && argc!=4) {
        printHelpAndExit();
    }
    if (testCase == "demo") {
        runCpu1DTest(100000, 500);
        printDividingLine();
        runCpuHybridTest(10000000, 5000);
        printDividingLine();
        runCpuGpu1DTest(50000000, 100000, NAIVE);
        printDividingLine();
        runGpu1DTest(50000000, 50000000, ILP);
        return;
    }
    int testSize = atoi(argv[2]);
    int M = atoi(argv[3]);
    if (M % 2 == 1) {
        std::cerr << "numFreqBin must be an even number.\n";
        exit(EXIT_FAILURE);
    }

    if (testCase == "cpu") {
        runCpu1DTest(testSize, M);
    } else if (testCase == "hybrid") {
        runCpuHybridTest(testSize, M);
    } else if (testCase == "gpu-naive") {
        runCpuGpu1DTest(testSize, M, NAIVE);
    } else {
        std::cerr << "Invalid test case.\n";
        printHelpAndExit();
    }
}


int main(int argc, char *argv[]) {
    TA_Utilities::select_coldest_GPU();
    int max_time_allowed_in_seconds = 1200;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);
    srand (time(NULL));
    runTest(argc, argv);
    return 0;
}

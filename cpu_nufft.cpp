#include "cpu_nufft.hpp"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>


#include "fft_helper.cuh"

using std::vector;

// DFT with non-uniform input data. This computes the normalized transform.
// Start by computing the uniform frequency spacing and then use DFT to compute
// the Fourier transformation
// Based on http://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/
// Tested against Python code to make sure that it is right.
vector<Complex> nudft(const vector<float> x, const vector<float> y, const int M, const float df) {
    std::cout << "Starting CPU NUDFT.\n";
    vector<float> freq = getFreq(df, M);
    vector<Complex> yt(freq.size(), Complex(0.0f, 0.0f));
    for (int k=0; k < freq.size(); ++k) {
        for (int j=0; j < x.size(); ++j) {
            yt[k] += (y[j] * std::exp(-J * freq[k] * x[j])) / static_cast<float>(x.size());
        }
    }
    std::cout << "Done with CPU NUDFT.\n";
    return yt;
}

// //////////////////////////////////////////////////////////////////////////////
// Fast NUFFT implementation. Gridding with a Gaussian kernel and FFT with FFTW3 or CUFFT.
// /////////////////////////////////////////////////////////////////////////////


// CPU Implementation of a Non-Uniform Fast-Fourier Transform by interpolation
// with a reasonably sized Gaussian kernel (Dutt & Rokhlin, 1993) and then FFT
// with FFTW. And finally undo the effect of the Gaussian kernel by dividing the
// Fourier transform of the kernel in the frequency domain.
// Also see http://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/
vector<Complex> nufftCpu(const vector<float> x, const vector<float> y, const int M, const bool gpuFFT,
      const float df, const float eps, const int iflag) {

    std::string type = gpuFFT ? "Hybrid" : "CPU";

    std::cout << "Starting " << type <<" NUFFT\n";
    struct Param param = computeGridParams(M, eps);
    int N = x.size();

    // the grid in time domain
    vector<float> ftau(param.Mr, 0.0);
    // the grid in frequency domain
    vector<Complex> Ftau(param.Mr);
    // the output
    vector<Complex> yt(M);

    // minimum spacing in x
    float hx = 2 * PI / param.Mr;
    // All the wave numbers
    vector<int> mm(2 * param.Msp + 1);
    std::iota(mm.begin(), mm.end(), -param.Msp);
    // spread ~ xi * kernel
    vector<float> spread(mm.size(), 0);

    std::cout << "Gridding kernel has size " << mm.size() << "; grid has size " << param.Mr << ".\n";

    for (int i = 0; i < N; ++i) {
        float xi = fmodf((x[i] * df), (2.0f * PI));
        int m = 1 + static_cast<int>(xi / hx);

        // Convolution
        for (int j = 0; j < spread.size(); j++) {
            spread[j] = expf(-0.25f * powf(xi - hx * (m + mm[j]), 2) / param.tau);
        }
        // Sum spread back to where it belongs.
        for (int j = 0; j < spread.size(); j++) {
            int index = (m + mm[j]) % param.Mr;
            // So that we end up with a positive modulo always.
            if (index < 0) { index += param.Mr; };
            ftau[index] += y[i] * spread[j];
        }
    }

    // Compute FFT on the convolved grid.
    if (gpuFFT) {
       Ftau = fftGpu(ftau, iflag);
       std::cout << "FFT with GPU\n";
    } else {
        std::cout << "FFT with CPU\n";
        Ftau = fftCpu(ftau, iflag);
    }

    // Picking the frequencies that we want from the oversampled frequency grid (and normalize).
    for (int i = (Ftau.size() - M / 2); i < Ftau.size(); ++i) {
        yt[i - static_cast<int>(Ftau.size()) + M / 2] = Ftau[i] / static_cast<float>(param.Mr);
    }
    for (int i = 0; i < (M / 2 + M % 2); ++i) {
        yt[i + M / 2] = Ftau[i] / static_cast<float>(param.Mr);
    }

    // Undoing the Gaussian gridding kernel.
    vector<float> k = getFreq(df, M);
    for (int i = 0; i < yt.size(); i++) {
        yt[i] = (1.0f / N) * sqrtf(PI / param.tau) * expf(param.tau * powf(k[i], 2)) * yt[i];
    }
    std::cout << "Done with " << type << " NUFFT \n";
    return yt;
}
#include "cpu_nufft.hpp"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <fftw3.h>

using std::vector;


vector<float> getFreq(float df, int M) {
    vector<float> freq(M);
    for (int i=0; i < freq.size(); i++) {
        freq[i] = df * (-M/2.0 +i);
    }
    return freq;
}



// DFT with non-uniform input data. This computes the unnormalized transform.
// Start by computing the uniform frequency spacing and then use DFT to compute
// the Fourier transformation
// Based on http://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/
// Tested against Python code to make sure that it is right.
vector<Complex> nudft(vector<float> x, vector<float> y, int M, float df) {
    vector<float> freq = getFreq(df, M);
    vector<Complex> yt(freq.size(), Complex(0.0f, 0.0f));
    for (int k=0; k < freq.size(); ++k) {
        for (int j=0; j < x.size(); ++j) {
            yt[k] += (y[j] * std::exp(J * freq[k] * x[j]));
        }
    }
    return yt;
}

// //////////////////////////////////////////////////////////////////////////////
// Fast NUFFT implementation. Gridding with a Gaussian kernel and FFT with FFTW3.
// /////////////////////////////////////////////////////////////////////////////

struct Param computeGridParams(const int M, const float eps) {
    // Choosing the interpolation Gaussian kernel parameters based on Dutt & Rohklin (1993)
    struct Param param;
    param.tau = 0.0f;
    // Oversampling ratio. ratio=3 gives higher accuracy.
    int ratio = 3;
    param.Msp = static_cast<int>(-std::log(eps) / (PI * (ratio - 1) / (ratio - 0.5f)) + 0.5f);
    param.Mr = std::max(2 * param.Msp, ratio * M);
    param.tau = PI * (param.Msp/ (ratio * (ratio - 0.5f))) / powf(M, 2.0f);
    return param;
}

// CPU FFT on a regular grid using FFTW
vector<Complex> fftCpu(vector<float> inp, const int iflag) {
    int n = inp.size();
    vector<Complex> out(n);
    fftwf_plan p;
    fftwf_complex *inCopied;
    fftwf_complex *outTemp;
    inCopied = (fftwf_complex*) fftw_malloc(sizeof(fftwf_complex) * n);
    outTemp = (fftwf_complex*) fftw_malloc(sizeof(fftwf_complex) * n);
    for (int i=0; i< n; i++) {
        inCopied[i][0] = inp[i];
        inCopied[i][1] = 0.0f;
    }

    if (iflag < 0) {
        p =  fftwf_plan_dft_1d(n, inCopied, outTemp, FFTW_FORWARD, FFTW_ESTIMATE);
    } else {
        // Cast inp in fftw_complex and use regular fftw
        p =  fftwf_plan_dft_1d(n, inCopied, outTemp, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    fftwf_execute(p);

    for (int i=0; i < n/2+1; i++) {
        out[i] = Complex(outTemp[i][0], outTemp[i][1]);
    }

    fftw_free(inCopied);
    fftw_free(outTemp);
    fftwf_destroy_plan(p);
    return out;
}


vector<Complex> nufftCpu(const vector<float> x, const vector<float> y, const int M,
        const float df, const float eps, const int iflag) {

    struct Param param = computeGridParams(M, eps);
    int N = x.size();

    // Construct the convolved grid
    vector<float> ftau(param.Mr, 0.0);
    vector<Complex> Ftau(param.Mr);

    float hx = 2 * PI / param.Mr;
    vector<int> mm(2 * param.Msp + 1);
    std::iota(mm.begin(), mm.end(), -param.Msp);
    vector<float> kernel(mm.size(), 0);

    for (int i=0; i< N; ++i) {
        float xi = fmodf((x[i] * df), (2.0f * PI));
        int m = 1 + static_cast<int>(xi / hx);
        for(int j=0; j< kernel.size(); j++) {
            kernel[j] = expf(-0.25f * powf(xi - hx * (m + mm[j]), 2)/param.tau);
        }
        // Convolution
        for(int j=0; j< kernel.size(); j++) {
            int index = (m + mm[j]) % param.Mr;
            if (index < 0) index += param.Mr;
            ftau[index] += y[i] * kernel[j];
        }
    }

    // Compute FFT on the convolved grid.
    // TODO Ftau = np.concatenate([Ftau[-(M//2):], Ftau[:M//2 + M % 2]])
     Ftau = fftCpu(ftau, iflag);

    // Deconvolve the Gaussian gridding kernel used.
    vector<float> k = getFreq(df, M);
    for (int i=0; i< Ftau.size(); i++) {
        Ftau[i] = (1.0f/N) * sqrtf(PI/param.tau) * expf(param.tau * powf(k[i], 2)) * Ftau[i];
    }
    std::cout << " Done...\n";
    return Ftau;
};
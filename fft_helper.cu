#include <cuda_runtime.h>
#include <cufft.h>
#include <fftw3.h>

#include "helper_cuda.h"

#include "fft_helper.cuh"

using std::vector;


// Utility method for computing the array of frequencies given the number of frequency bins M.
vector<float> getFreq(float df, int M) {
    vector<float> freq(M);
    for (int i=0; i < freq.size(); i++) {
        freq[i] = df * (-M/2.0f +i);
    }
    return freq;
}


// Compute parameters for gridding.
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

    for (int i=0; i < n; i++) {
        out[i] = Complex(outTemp[i][0], outTemp[i][1]);
    }

    fftw_free(inCopied);
    fftw_free(outTemp);
    fftwf_destroy_plan(p);
    return out;
}


void callCufft(cufftComplex *dev_in, cufftComplex *dev_out, int n, const int iflag) {
    cufftHandle plan;
    int batch = 1;
    CUFFT_CALL(cufftPlan1d(&plan, n, CUFFT_C2C, batch));
    if (iflag < 0) {
        CUFFT_CALL(cufftExecC2C(plan, dev_in, dev_out, CUFFT_FORWARD));
    } else {
        CUFFT_CALL(cufftExecC2C(plan, dev_in, dev_out, CUFFT_INVERSE));
    }
    CUFFT_CALL(cufftDestroy(plan));
}

// GPU FFT for testing CUFFT against FFTW
vector<Complex> fftGpu(vector<float> inp, const int iflag) {
    int n = inp.size();
    vector<Complex> out(n);
    cufftComplex complex_in[n];
    cufftComplex *dev_in;
    cufftComplex *dev_out;
    CUDA_CALL(cudaMalloc((void **) &dev_in, n * sizeof(cufftComplex)));
    CUDA_CALL(cudaMalloc((void **) &dev_out, n * sizeof(cufftComplex)));

    for (int i=0;i<inp.size(); ++i) {
        complex_in[i].x = inp[i];
        complex_in[i].y = 0.0f;
    }

    CUDA_CALL(cudaMemcpy(dev_in, complex_in, n * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    callCufft(dev_in, dev_out, n, iflag);
    CUDA_CALL(cudaMemcpy(out.data(), dev_out, n * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(dev_in));
    CUDA_CALL(cudaFree(dev_out));
    return out;
}
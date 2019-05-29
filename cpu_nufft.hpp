#ifndef  CPU_NUFFT_HPP
#define CPU_NUFFT_HPP
#include <vector>
#include <complex>

typedef std::complex<float> Complex;

const float PI = std::acos(-1);
const Complex J(0, 1);
namespace CpuNufft
{
    std::vector<std::complex<float>> something();
    std::vector<std::complex<float>> somethingElse();
}

// Parameters used for the convolution based fast NUFFT.
struct Param {
    int Msp;
    int Mr;
    float tau;
};

std::vector<float> getFreq(float df, int M);

std::vector<Complex> nufftCpu(std::vector<float> x, std::vector<float> y, int M,
        float df=1.0, float eps=1e-15, int iflag=-1);

std::vector<Complex> nudft(std::vector<float> x, std::vector<float> y, int M, float df);
#endif
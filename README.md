This is my final project for CS 179 GPU Programming.
## Build and Run Instruction
To build
```
mkdir build
cd build
cmake ..
make -j4
```
Then executing `./NUFFT` should print the help message and how each test case can be run. As it stands now, I am
only doing forward FFT with real inputs. I am not dealing with odd number of frequency bins since I do not
understand how various FFT routines handle it and I got inconsistent results with my DFT code. A easy workaround
is to use (odd number + 1) as the frequency bin number and choose the frequency bins desired out of the output.

For demo, `./NUFFT demo` would run through all algorithms and compare them. We're only dealing with FFT forward
transform with real input here. The demo script runs through
the following cases

- CPU Direct Fourier Transformation against CPU NUFFT (to establish the correctness of NUFFT implementation)
- CPU NUFFT against CPU Gridding with CUFFT FFT (test CUFFT code)
- CPU NUFFT against GPU NAIVE (see sections below. The baseline GPU algorithm)
- GPU NAIVE against GPU SHMEM (only with frequency bin of 4000; see sections below)
- GPU NAIVE against GPU ILP (GPU ILP is the best I have. This has the target input/output dimension of 5e7)
- GPU NAIVE against GPU PARALLEL (the PARALLEL algorithm turns out to be extremely slow)


## Introduction
Fast Fourier Transform requires the input to be on a uniform grid. In a lot of applications of FFT, like medical
imaging/MRI and radio astronomy. Interpolation of non-uniform data onto the uniform grid (known as “gridding”), tends to
be the bottleneck of coherence-based imaging. The goal of this project is to investigate the Gaussian gridding algorithm
NUFFT (Non-uniform Fast-Fourier Transform), as detailed in Dutt & Rokhlin (1993) and illustrated in python by
http://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/. 

The algorithms implemented here are all in 1-dimension. However, the eventual use case I have in mind is for
high resolution medical and radio astronomy images. So the number of samples (input x,y pair) can be arbitrarily
large and the number of output bins would be on the order of 5000.
Ideally I would do 5000<sup>2</sup> for the frequency bin count, but that corresponds to a Nyquist frequency that is
way lower than the precision offered by single precision mantissa, so verification of the GPU algorithm against the
CPU algorithm cannot be done.

The number 5 * 10<sup>7</sup> is chosen to be my target input size for the GPU implementation. And with my not very
elaborate mechanism for generating test data, it seems like I can have up to 300000 frequency bins without losing
numerical stability, which is two orders of magnitude lower than what I am targeting.

## Overview of Algorithm
Suppose we want to transform N non-uniformly spaced (x,y) pair onto M frequencies. The x<sub>i</sub>s are not assumed
to be ordered.The NUFFT algorithm relies on the
clever construction of a Gaussian kernel of size K (it ends up being 29 for all of my test cases). It then interpolates
the input data onto a grid of M\*r uniformly spaced x<sub>i</sub>s, where r is oversampling ratio. r only depends on how
close you want to be to the Direct Fourier Transform (DFT) result. r=3 is chosen in this case and the authors claim that
it gives 1e-14 accuracy. The Gridding step has a O(N\*K) cost. 

After gridding, we FFT the oversampled grid to obtain an oversampled frequency grid. And we just take the head and tail
of the frequency grid corresponding to the frequencies that we are interested in. The FFT, as usual, would have a
runtime of O(M\*r log (M\*r)).

Finally, we undo the gridding kernel in the frequency domain by diving out its Fourier transform which is also a
Gaussian. This has a runtime of O(M).

## GPU Implementation
The GPU implementation has three steps: gridding, FFT, and post-processing. Gridding interpolates the data onto the
oversampled grid. FFT calls CUFFT to do the FFT. Post-processing does the undoing gridding kernel and choosing the right
frequencies out of the oversampled frequency grid, as well as doing the 1/N scaling.

The runtime at low number of frequency bins (~10000) is dominated by FFT. However, at high (~50000000) number of
frequency bins, it is dominated by gridding. Here I introduce the gridders implemented and my attempts to make it
faster. The main difficulty is that the grid would not fit in shared memory. It is worth noting that one can
harness the Hermitian structure of the input/output and reduce the FFT runtime (which I did not have time to try
out here).

### Naive Gridder
The naive gridder has N number of threads, each doing its own share of multiplication with the kernel and then
atomicAdd to the device memory. Testing by deleting parts of the code showed that the atomicAdd took up ~95% of the
runtime of the kernel. It is still pretty fast (1353ms for 5e7 input and frequency bins) but there's room for
improvement. Note that the input x are completely randomized, so this probably helps atomicAdd by not having hot
grid points.

### ILP Gridder
This is basically the naive gridder with some assumptions. The kernel size only depends on the precision needed and is
29 for all of the reasonable test cases. So I used #pragma unroll 29 before the loop so that nvcc would unroll the loop.
Also I removed a couple assignment statements to remove instruction dependencies. Finally I substitute instrinsics for
the math functions (e.g. \_\_expf)whenever possible. This has a 20% improvement over the NAIVE gridder in the largest
testcase (5e7 input and 5e7 output).

### PARALLEL Gridder
The thought was that if I introduce another blockDim with size that of the number of grid points (M * r), and just
have each column of blocks sum to one single single grid point with reduction, I would reduce the number of atomicAdd
call to device memory. But this didn't quite work out. For 5e7 samples and 1e5 frequency bins, 
reading the data takes ~43s; the main computation took 250s; the reduction took another 30s. It seems like there's not
much latency to hide here and therefore I just ended up with 10<sup>12</sup> threads waiting for SM resources.

### SHMEM Gridder
I thought about sorting the input x<sub>i</sub>s (or rather x<sub>i</sub> * df % (2pi))first and then divide the grid
into shared-memory sized sub-grids. Then I can do a subset of the input for a subset of the grid on a given block.
But sorting on GPU can be expensive too so I thought I'd do a proof on concept with a grid size of 12000 (4000
frequency bins). This is the SHMEM gridder. It accumulates the sum for each grid point onto shared memory and then
sum them back to device memory. The input is still not sorted; so I don't get any parallel shared memory access.
With 5e7 input and 4000 frequency bins, I only save 3ms out of 82ms for the gridding kernel.
Perhaps if the input is sorted and we can index the threads cleverly to enable bank-parallel memory access, it
would be much faster.

## Code Structure
main.cpp is the entry point of the program that contains the main() method and the test cases.

cpu_nufft.cpp/cpu_nufft.hpp contains the CPU implementations: CPU DFT and CPU NUFFT.

gpu_nufft.cpp/gpu_nufft.cuh contains the GPU implementation with all the different gridders.

fft_helper.cu/fft_helper.cuh contains the FFT code with FFTW on the CPU and CUFFT on the GPU, as well as common
methods for forming the oversampled grid.


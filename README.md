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

### Overview of Algorithm
Suppose we want to transform N non-uniformly spaced (x,y) pair onto M frequencies. The NUFFT algorithm relies on the
clever construction of a Gaussian kernel of size K (it ends up being 29 for all of my test cases). It then interpolates
the input data onto a grid of M*r uniformly spaced x<sub>i</sub>s, where r is oversampling ratio. r only depends on how
close you want to be to the Direct Fourier Transform (DFT) result. r=3 is chosen in this case and the authors claim that
it gives 1e-14 accuracy. 

### GPU Implementation
The run time is dominated by the FFT step. But the gridding complexity scales with the size of the gridding kernel, so it
is important to optimize the gridder too.
#### Naive Gridder

#### With Reduction
Tried substituting \_\_expf for expf but no significant improvement was seen.

without atomicAdd the gridder took about 2.4ms. With the atomicAdd it took about 82ms with 5000000 samples
and kernel size of 29 and grid size 15000. So the atomicAdd is our biggest performance bottleneck outside
of the actual FFT (which we can probably also optimize by using the Hermitian structure).

Tried #pragma unroll

Blowing up the thread dimension slows things down significantly. Reading the data takes ~43s. It seems like
the extra factor of ~15000 threads aren't really run in parallel. The main computation took 250s. Reduction took
another 30s.

## Build and Run Instruction
To build
```
mkdir build
cd build
cmake ..
make -j 4
```
Then execute `./NUFFT` should run the executable.

## Output on Titan
```
yuping@titan:~/NUFFT/build$ ./NUFFT
Comparing 1D CPU Non-uniform FFT and DFT with 100000 samples and 50 frequency bins...
The signal contains 3 different frequencies and a bit of noise.Starting CPU NUDFT.
Done with CPU NUDFT.
Starting CPU NUFFT
Done with CPU NUFFT
Comparing results...
The two results are the same.
CPU NUDFT completed in 912 ms
CPU NUFFT completed in 168 ms
```

## Introduction
motivation
### Overview of algorithm

### GPU Implementation
The run time is dominated by the FFT step. But the gridding complexity scales with the size of the gridding kernel, so it
is important to optimize the gridder too.
#### Naive Gridder
#### With Reduction
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

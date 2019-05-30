### CPU Demo Instruction
To build
```
mkdir build
cd build
cmake ..
```
Then execute `./NUFFT` should run the current executable.

#### Output on Titan
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

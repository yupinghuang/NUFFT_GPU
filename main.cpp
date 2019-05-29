#include <iostream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <complex>
#include <fftw3.h>

#include "ta_utilities.hpp"
#include "cpu_nufft.hpp"

using std::vector;

void checkArgs(int argc, char **argv) {
   if (argc != 3) {
       std::cerr << "Arguments: <testcase> \n";
       exit(EXIT_FAILURE);
   }
}



void runCpu1DTest() {
    int testSize = 1000;
    vector<float> x(testSize);
    vector<float> y(testSize);
    for (int i=0; i< testSize; ++i) {
        x[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10));;
        y[i] = sin(x[i]);
    }
    vector<Complex> ans = nudft(x, y, 100, 1.0);
    std::cout << "Finished computing stuff" << std::endl;
    std::cout<<ans[ans.size()/2];
    vector<Complex> ans2 = nufftCpu(x, y, 100);
    // std::cout<<ans2[ans2.size()/2];
}

int runTest(int argc, char **argv) {
    if (argv[1] == "cpu-1d") {
        runCpu1DTest();
    }
    return 0;
}

void test_fftw() {
    float  in1[] = { 0.00000, 0.12467, 0.24740, 0.36627,
                     0.47943, 0.58510, 0.68164, 0.76754
    };

    int N = 8;
    float  in2[N];

    fftwf_complex out[N / 2 + 1];
    fftwf_plan    p1, p2;

    p1 = fftwf_plan_dft_r2c_1d(N, in1, out, FFTW_ESTIMATE);
    p2 = fftwf_plan_dft_c2r_1d(N, out, in2, FFTW_ESTIMATE);

    fftwf_execute(p1);
    fftwf_execute(p2);

    for (int i = 0; i < N; i++) {
        printf("%2d %15.10f %15.10f\n", i, in1[i], in2[i] / N);
    }

    fftwf_destroy_plan(p1);
    fftwf_destroy_plan(p2);
}


int main(int argc, char **argv) {
    //TA_Utilities::select_coldest_GPU();
    //int max_time_allowed_in_seconds = 300;
    //TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);
    // srand (time(NULL));
    runCpu1DTest();
    // test_fftw();
    return 0;
}
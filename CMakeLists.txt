cmake_minimum_required(VERSION 3.10)
project(NUFFT LANGUAGES CXX)
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

set(CXX_FLAGS "-g -O3 -Wall -D_REENTRANT")
set(NVCC_GENCODES "-gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61")

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} -O3 -D_REENTRANT ${NVCC_GENCODES}" )


add_executable(NUFFT main.cpp cpu_nufft.cpp cpu_nufft.hpp gpu_nufft.cu gpu_nufft.cuh fft_helper.cu fft_helper.cuh ta_utilities.cpp ta_utilities.hpp)
set_target_properties(NUFFT PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(NUFFT fftw3f)
target_link_libraries(NUFFT fftw3)
target_link_libraries(NUFFT cufft)

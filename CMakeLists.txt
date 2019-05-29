cmake_minimum_required(VERSION 3.12)
project(NUFFT)

set(CMAKE_CXX_STANDARD 11)

add_executable(NUFFT main.cpp cpu_nufft.cpp cpu_nufft.hpp)
target_link_libraries(NUFFT fftw3f)
target_link_libraries(NUFFT fftw3)

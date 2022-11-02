cmake_minimum_required(VERSION 3.17)
project(cufft_plans LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
find_package(CUDAToolkit)

message(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
message(${CUDAToolkit_LIBRARY_DIR})

add_executable(c2c c2c.cu)
target_include_directories(c2c PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(c2c PRIVATE CUDA::cufft CUDA::cudart)

add_executable(cufft_plans_with_cache cufft_plans_2d_with_cache.cc)
target_include_directories(cufft_plans_with_cache PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cufft_plans_with_cache PRIVATE CUDA::cufft CUDA::cudart)
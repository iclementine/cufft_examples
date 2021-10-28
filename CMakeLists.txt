cmake_minimum_required(VERSION 3.10)
project(cmake_and_cuda LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

add_executable(c2c c2c.cu)
target_include_directories(c2c PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(c2c PRIVATE cufft)

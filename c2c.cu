#include "cufft_utility.h"
#include "dynload.h"
#include <algorithm>
#include <array>
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <thrust/device_vector.h>
#include <vector>

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::complex<T> &x) {
  os << x.real() << "+" << x.imag() << "j";
  return os;
}

std::ostream &operator<<(std::ostream &os, const cufftComplex &x) {
  os << x.x << "+" << x.y << "j";
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  os << "{";
  for (size_t i = 0; i < v.size(); i++) {
    if (i)
      os << ", ";
    os << v[i];
  }
  os << "}";
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const thrust::device_vector<T> v) {
  os << "{";
  for (size_t i = 0; i < v.size(); i++) {
    if (i)
      os << ", ";
    os << v[i];
  }
  os << "}";
  return os;
}

int main() {
  int version;
  dyn::cufftGetVersion(&version);
  std::cout << "cufft version: " << version << std::endl;
  std::vector<CuFFTHandle> plans(10);
  CuFFTHandle &a = plans[0];

  std::array<long long int, 4> r_shape{1, 4, 28, 107};
  int r_numel = std::accumulate(r_shape.begin(), r_shape.end(), 1,
                                std::multiplies<int>());
  std::array<long long int, 4> c_shape{1, 4, 28, 54};
  int c_numel = std::accumulate(c_shape.begin(), c_shape.end(), 1,
                                std::multiplies<int>());

  size_t work_size = 0;
  std::vector<float> x(r_numel);
  std::random_device rd{};
  std::mt19937 gen{rd()};

  // values near the mean are the most likely
  // standard deviation affects the dispersion of generated values from the mean
  std::normal_distribution<float> d{0, 1};
  for (auto &v : x) {
    v = d(gen);
  }
  thrust::device_vector<cufftComplex> dx(r_numel);
  cudaMemcpy(thrust::raw_pointer_cast(dx.data()), x.data(),
             r_numel * sizeof(float), cudaMemcpyHostToDevice);

  dyn::cufftXtMakePlanMany(a.get(), 2, r_shape.data() + 2, nullptr, 1, 1,
                           CUDA_R_32F, nullptr, 1, 1, CUDA_C_32F, 4, &work_size,
                           CUDA_C_32F);
  dyn::cufftSetAutoAllocation(a.get(), 0);

  for (int i = 0; i < 10; i++) {
    thrust::device_vector<cufftComplex> dy(c_numel);
    void *ws;
    cudaMalloc(&ws, work_size);
    dyn::cufftSetWorkArea(a.get(), ws);
    cudaStream_t s;
    cudaStreamCreate(&s);
    dyn::cufftSetStream(a.get(), s);
    dyn::cufftXtExec(a.get(), thrust::raw_pointer_cast(dx.data()),
                     thrust::raw_pointer_cast(dy.data()), CUFFT_FORWARD);

    std::vector<std::complex<float>> y(c_numel);
    cudaMemcpy(y.data(), thrust::raw_pointer_cast(dy.data()),
               c_numel * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    // std::cout << y << std::endl;
    cudaFree(ws);
  }
  return 0;
}
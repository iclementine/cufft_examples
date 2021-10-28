#include "cuffthandle.h"
#include <cufft.h>
#include <cufftXt.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <complex>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

template <typename T>
std::ostream& operator<< (std::ostream& os, const std::complex<T>& x) {
    os << x.real() << "+" << x.imag() << "j";
    return os;
}

std::ostream& operator<< (std::ostream& os, const cufftComplex& x) {
    os << x.x << "+" << x.y << "j";
    return os;
}

template <typename T>
std::ostream& operator<< (std::ostream& os, const std::vector<T>& v) {
    os << "{";
    for (size_t i = 0; i < v.size(); i++){
        if (i) os << ", ";
        os << v[i];
    }
    os << "}";
    return os;
}

template <typename T>
std::ostream& operator<< (std::ostream& os, const thrust::device_vector<T> v) {
    os << "{";
    for (size_t i = 0; i < v.size(); i++){
        if (i) os << ", ";
        os << v[i];
    }
    os << "}";
    return os;
}

int main(){
    CuFFTHandle b;
    CuFFTHandle a(std::move(b));
    long long int N = 16;
    size_t work_size = 0;
    std::vector<std::complex<float>> x(N);
    std::iota(x.begin(), x.end(), 10.0);
    thrust::device_vector<cufftComplex> dx(N);
    cudaMemcpy(
        thrust::raw_pointer_cast(dx.data()), 
        x.data(), 
        N * sizeof(cufftComplex), 
        cudaMemcpyHostToDevice);

    cufftXtMakePlanMany(
        a.get(), 1, &N, 
        nullptr, 1, 1, CUDA_C_32F, 
        nullptr, 1, 1, CUDA_C_32F, 
        1, &work_size, CUDA_C_32F);
    
    thrust::device_vector<cufftComplex> dy(N);
    cufftXtExec(a.get(), 
                thrust::raw_pointer_cast(dx.data()), 
                thrust::raw_pointer_cast(dy.data()),
                CUFFT_FORWARD);

    std::vector<std::complex<float>> y(N);
    cudaMemcpy(
        y.data(), 
        thrust::raw_pointer_cast(dy.data()),
        N * sizeof(cufftComplex), 
        cudaMemcpyDeviceToHost);
    std::cout << y << std::endl;
    return 0;
}
#include "cuda_runtime.h"
#include "cuda_utility.h"
#include "cufft.h"
#include "cufftXt.h"
#include "cufft_utility.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include <complex>
#include <iostream>

int main() {
  // =======================cufft plan================================
  std::vector<int64_t> input_sizes {192, 28, 107};
  FFTConfig config(input_sizes, FFTTransformType::R2C, DataType::f4);

  for (int i = 0; i < 10; i++) {
    // input in host
    Eigen::Tensor<float, 3> y(input_sizes[0], input_sizes[1], input_sizes[2]);
    y.setRandom();
    std::cout << "total elements: " << y.size() << std::endl;
    size_t size_in_bytes = y.size() * sizeof(float);

    // copy to device
    float *y_on_device = nullptr;
    CUDA_CHECK(cudaMalloc(&y_on_device, size_in_bytes));
    CUDA_CHECK(cudaMemcpy(y_on_device, y.data(), size_in_bytes,
                          cudaMemcpyHostToDevice));

    // output on device and output for host
    Eigen::Tensor<std::complex<float>, 3> D(input_sizes[0], input_sizes[1],
                                            input_sizes[2] / 2 + 1);
    D.setZero();
    size_t D_size_in_bytes = D.size() * sizeof(std::complex<float>);
    void *D_on_device = nullptr;
    CUDA_CHECK(cudaMalloc(&D_on_device, D_size_in_bytes));

    // renew workspace
    void *workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, config.workspace_size()));
    CUFFT_CHECK(dyn::cufftSetWorkArea(config.plan(), workspace));

    // ============================Execution==============================
    CUFFT_CHECK(
        dyn::cufftXtExec(config.plan(), y_on_device, D_on_device, CUFFT_FORWARD));

    // free input & output & workspace, copy output back to host
    CUDA_CHECK(cudaMemcpy(D.data(), D_on_device, D_size_in_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(y_on_device));
    CUDA_CHECK(cudaFree(D_on_device));
    CUDA_CHECK(cudaFree(workspace));
    std::cout << "Done once: " << std::endl;
  }
  return 0;
}

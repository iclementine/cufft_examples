#include "cuda_runtime.h"
#include "cuda_utility.h"
#include "cufft.h"
#include "cufftXt.h"
#include "cufft_utility.h"
#include "lru_cache.h"
#include <array>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

int main() {
  // =======================cufft plan================================
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(1.0, 2.0);

  for (int i = 0; i < 10; i++) {
    // ---------make fft config key----------
    std::vector<int64_t> input_sizes{2, 3, 107};
    std::vector<int64_t> output_sizes{input_sizes[0], input_sizes[1],
                                      input_sizes[2] / 2 + 1};
    std::vector<int64_t> fft_sizes{input_sizes[1], input_sizes[2]};
    FFTConfigKey key(input_sizes, output_sizes, fft_sizes,
                     FFTTransformType::R2C, DataType::f4);

    // ---------get a config either by creating or using cached config----------
    FFTConfig *config = nullptr;
    std::unique_ptr<FFTConfig> config_ =
        nullptr; // to hold it if not using cache because it cannot be copied or
                 // assigned
    if (true) {
      FFTConfigCache &cache = get_fft_plan_cache(/*device_id*/ 0);
      std::unique_lock<std::mutex> guard(cache.mutex, std::defer_lock);
      guard.lock();
      config = &cache.lookup(key);
    } else {
      config_ = std::make_unique<FFTConfig>(key);
      config = config_.get();
    }

    // input in host
    std::vector<float> y(input_sizes[0] * input_sizes[1] * input_sizes[2], 0.0);
    for (auto &item : y) {
      item = dis(gen);
    }
    std::cout << "total elements: " << y.size() << std::endl;
    size_t size_in_bytes = y.size() * sizeof(float);

    // copy to device
    float *y_on_device = nullptr;
    CUDA_CHECK(cudaMalloc(&y_on_device, size_in_bytes));
    CUDA_CHECK(cudaMemcpy(y_on_device, y.data(), size_in_bytes,
                          cudaMemcpyHostToDevice));

    // output on device and output for host
    std::vector<std::complex<float>> D(
        output_sizes[0] * output_sizes[1] * output_sizes[2], 0.0);
    size_t D_size_in_bytes = D.size() * sizeof(std::complex<float>);
    void *D_on_device = nullptr;
    CUDA_CHECK(cudaMalloc(&D_on_device, D_size_in_bytes));

    // renew workspace
    void *workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, config->workspace_size()));
    CUFFT_CHECK(dyn::cufftSetWorkArea(config->plan(), workspace));

    // ============================Execution==============================
    CUFFT_CHECK(dyn::cufftXtExec(config->plan(), y_on_device, D_on_device,
                                 CUFFT_FORWARD));

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

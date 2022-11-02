#pragma once
#include "cuda_utility.h"
#include "cufft.h"
#include "cufftXt.h"
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

class CuFFTHandle {
private:
  cufftHandle raw_handle;

public:
  CuFFTHandle() {
    CUFFT_CHECK(cufftCreate(&raw_handle));
    std::cout << "Created cufftHandle " << raw_handle << std::endl;
  }

  CuFFTHandle(const CuFFTHandle &) = delete;
  CuFFTHandle &operator=(const CuFFTHandle &) = delete;
  CuFFTHandle(CuFFTHandle &&other) = delete;
  CuFFTHandle &operator=(CuFFTHandle &&other) = delete;

  ~CuFFTHandle() {
    std::cout << "Destroying cufftHandle " << raw_handle << std::endl;
    CUFFT_CHECK(cufftDestroy(raw_handle));
  }

  cufftHandle &get() { return raw_handle; }

  const cufftHandle &get() const { return raw_handle; }
};

// -----------------fft config key-------------------
constexpr int64_t kMaxFFTNdim = 3;
constexpr int64_t kMaxDataNdim = kMaxFFTNdim + 1;

enum class FFTTransformType : int8_t {
  C2C = 0, // Complex-to-complex
  R2C,     // Real-to-complex
  C2R,     // Complex-to-real
};

enum class DataType : int8_t {
  f4 = 0,
  f8,
  c8,
  c16,
};

struct FFTConfigKey {
  int signal_ndim_; // 1 <= signal_ndim <= kMaxFFTNdim
  // These include additional batch dimension as well.
  int64_t sizes_[kMaxDataNdim];
  int64_t input_shape_[kMaxDataNdim];
  int64_t output_shape_[kMaxDataNdim];
  FFTTransformType fft_type_;
  DataType value_type_;

  using shape_t = std::vector<int64_t>;
  FFTConfigKey() = default;

  FFTConfigKey(const shape_t &in_shape, const shape_t &out_shape,
               const shape_t &signal_size, FFTTransformType fft_type,
               DataType value_type) {
    // Padding bits must be zeroed for hashing
    memset(this, 0, sizeof(*this));
    signal_ndim_ = signal_size.size() - 1;
    fft_type_ = fft_type;
    value_type_ = value_type;
    std::copy(signal_size.cbegin(), signal_size.cend(), sizes_);
    std::copy(in_shape.cbegin(), in_shape.cend(), input_shape_);
    std::copy(out_shape.cbegin(), out_shape.cend(), output_shape_);
  }
};

template <typename Key> struct KeyHash {
  // Key must be a POD because we read out its memory
  // contenst as char* when hashing
  static_assert(std::is_pod<Key>::value, "Key must be plain old data type");

  size_t operator()(const Key &params) const {
    auto ptr = reinterpret_cast<const uint8_t *>(&params);
    uint32_t value = 0x811C9DC5;
    for (int i = 0; i < static_cast<int>(sizeof(Key)); ++i) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return static_cast<size_t>(value);
  }
};

template <typename Key> struct KeyEqual {
  // Key must be a POD because we read out its memory
  // contenst as char* when comparing
  static_assert(std::is_pod<Key>::value, "Key must be plain old data type");

  bool operator()(const Key &a, const Key &b) const {
    auto ptr1 = reinterpret_cast<const uint8_t *>(&a);
    auto ptr2 = reinterpret_cast<const uint8_t *>(&b);
    return memcmp(ptr1, ptr2, sizeof(Key)) == 0;
  }
};

// Returns true if the transform type has complex input
inline bool has_complex_input(FFTTransformType type) {
  switch (type) {
  case FFTTransformType::C2C:
  case FFTTransformType::C2R:
    return true;

  case FFTTransformType::R2C:
    return false;
  }
  return false;
}

// Returns true if the transform type has complex output
inline bool has_complex_output(FFTTransformType type) {
  switch (type) {
  case FFTTransformType::C2C:
  case FFTTransformType::R2C:
    return true;

  case FFTTransformType::C2R:
    return false;
  }
  return false;
}

class FFTConfig {
public:
  using plan_size_type = long long int; // NOLINT (be consistent with cufft)
  explicit FFTConfig(const FFTConfigKey &key)
      : FFTConfig(
            std::vector<int64_t>(key.sizes_, key.sizes_ + key.signal_ndim_ + 1),
            key.fft_type_, key.value_type_) {}
  // sizes are full signal, including batch size and always two-sided
  FFTConfig(const std::vector<int64_t> &sizes, FFTTransformType fft_type,
            DataType precison)
      : fft_type_(fft_type), precision_(precison) {
    const auto batch_size = static_cast<plan_size_type>(sizes[0]);
    std::vector<plan_size_type> signal_sizes(sizes.cbegin() + 1, sizes.cend());
    const int signal_ndim = sizes.size() - 1;

    cudaDataType itype, otype, exec_type;
    const bool complex_input = has_complex_input(fft_type);
    const bool complex_output = has_complex_output(fft_type);
    if (precison == DataType::f4) {
      itype = complex_input ? CUDA_C_32F : CUDA_R_32F;
      otype = complex_output ? CUDA_C_32F : CUDA_R_32F;
      exec_type = CUDA_C_32F;
    } else {
      itype = complex_input ? CUDA_C_64F : CUDA_R_64F;
      otype = complex_output ? CUDA_C_64F : CUDA_R_64F;
      exec_type = CUDA_C_64F;
    }

    // disable auto allocation of workspace to use allocator from the framework
    CUFFT_CHECK(cufftSetAutoAllocation(plan(), /* autoAllocate */ 0));
    CUFFT_CHECK(cufftXtMakePlanMany(plan(), signal_ndim, signal_sizes.data(),
                                    /* inembed */ nullptr,
                                    /* base_istride */ 1L,
                                    /* idist */ 1L, itype,
                                    /* onembed */ nullptr,
                                    /* base_ostride */ 1L,
                                    /* odist */ 1L, otype, batch_size,
                                    &ws_size_, exec_type));
  }

  FFTConfig(const FFTConfig &other) = delete;
  FFTConfig &operator=(const FFTConfig &other) = delete;

  FFTConfig(FFTConfig &&other) = delete;
  FFTConfig &operator=(FFTConfig &&other) = delete;

  const cufftHandle &plan() const { return plan_.get(); }
  FFTTransformType transform_type() const { return fft_type_; }
  DataType data_type() const { return precision_; }
  size_t workspace_size() const { return ws_size_; }

private:
  CuFFTHandle plan_;
  size_t ws_size_; // workspace size in bytes
  FFTTransformType fft_type_;
  DataType precision_;
};

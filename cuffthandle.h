#pragma once
#include <iostream>

#include "cufft.h"
#include "cufftXt.h"

class CuFFTHandle {
 private:
  cufftHandle raw_handle;

 public:
  CuFFTHandle() {
    cufftCreate(&raw_handle);
    std::cout << "Created cufftHandle " << raw_handle << std::endl;
  }

  CuFFTHandle(const CuFFTHandle &) = delete;
  CuFFTHandle &operator=(const CuFFTHandle &) = delete;

  CuFFTHandle(CuFFTHandle &&other) : raw_handle(other.raw_handle) {
    // 0 is a magic number for cufftHandle, if is the null value of cufftHandle
    // which is always safe to be cufftDestroyed
    other.raw_handle = 0;
  }

  CuFFTHandle &operator=(CuFFTHandle &&other) {
    std::cout << "Destroying cufftHandle " << raw_handle << std::endl;
    cufftDestroy(raw_handle);
    raw_handle = other.raw_handle;
    other.raw_handle = 0;
    return *this;
  };

  ~CuFFTHandle() {
    std::cout << "Destroying cufftHandle " << raw_handle << std::endl;
    cufftDestroy(raw_handle);
  }

  cufftHandle &get() { return raw_handle; }

  const cufftHandle &get() const { return raw_handle; }
};
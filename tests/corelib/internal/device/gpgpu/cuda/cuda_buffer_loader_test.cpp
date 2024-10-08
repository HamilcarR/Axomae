#include "internal/common/math/math_random.h"
#include "internal/device/gpgpu/cuda/CudaDevice.h"
#include "internal/device/gpgpu/device_transfer_interface.h"
#include <gtest/gtest.h>

namespace gpu = device::gpgpu;

const int BUFFER_SIZE = 200;
TEST(cuda_buffer_loader_test, allocate_buffer) {
  gpu::GPU_query_result result = gpu::allocate_buffer(BUFFER_SIZE);
  ASSERT_NE(result.device_ptr, nullptr);
  gpu::deallocate_buffer(result.device_ptr);
}

template<class T>
static void check_array_equal(const std::array<T, BUFFER_SIZE> &test, const std::array<T, BUFFER_SIZE> &expected) {
  for (int i = 0; i < BUFFER_SIZE; i++)
    ASSERT_EQ(test[i], expected[i]);
}

TEST(cuda_buffer_loader_test, copy_buffer) {
  std::array<int, BUFFER_SIZE> array_test = {0};
  for (int i = 0; i < BUFFER_SIZE; i++)
    array_test[i] = math::random::nrandi(0, BUFFER_SIZE);
  gpu::GPU_query_result result = gpu::allocate_buffer(BUFFER_SIZE);
  ASSERT_NE(result.device_ptr, nullptr);
  void *device_buffer = result.device_ptr;
  gpu::copy_buffer(array_test.data(), device_buffer, BUFFER_SIZE * sizeof(int), gpu::HOST_DEVICE);
  std::array<int, BUFFER_SIZE> array_result = {0};
  gpu::copy_buffer(array_result.data(), device_buffer, BUFFER_SIZE * sizeof(int), gpu::DEVICE_HOST);
  check_array_equal(array_test, array_result);
  gpu::deallocate_buffer(result.device_ptr);
}
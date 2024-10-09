#include "internal/common/math/math_random.h"
#include "internal/device/gpgpu/device_transfer_interface.h"
#include <gtest/gtest.h>

#define DEVICE_GTEST_ASSERT(ans) \
  { \
    if (!ans.error_status.isOk()) { \
      DEVICE_ERROR_CHECK(ans.error_status); \
      ASSERT_TRUE(false); \
    } \
  }

namespace gpu = device::gpgpu;

const int BUFFER_SIZE = 200;
TEST(device_resource_loader_test, allocate_buffer) {
  gpu::GPU_query_result result = gpu::allocate_buffer(BUFFER_SIZE);
  ASSERT_NE(result.device_ptr, nullptr);
  DEVICE_GTEST_ASSERT(gpu::deallocate_buffer(result.device_ptr));
}

template<class T>
static void check_array_equal(const std::array<T, BUFFER_SIZE> &test, const std::array<T, BUFFER_SIZE> &expected) {
  for (int i = 0; i < BUFFER_SIZE; i++)
    ASSERT_EQ(test[i], expected[i]);
}

static void generate_random_arrayi(std::array<int, BUFFER_SIZE> &array) {
  for (int i = 0; i < BUFFER_SIZE; i++)
    array[i] = math::random::nrandi(0, BUFFER_SIZE);
}

TEST(device_resource_loader_test, copy_buffer) {
  std::array<int, BUFFER_SIZE> array_test = {0};
  generate_random_arrayi(array_test);
  gpu::GPU_query_result result = gpu::allocate_buffer(BUFFER_SIZE * sizeof(int));
  ASSERT_NE(result.device_ptr, nullptr);
  DEVICE_GTEST_ASSERT(gpu::copy_buffer(array_test.data(), result.device_ptr, BUFFER_SIZE * sizeof(int), gpu::HOST_DEVICE));
  std::array<int, BUFFER_SIZE> array_result = {0};
  DEVICE_GTEST_ASSERT(gpu::copy_buffer(result.device_ptr, array_result.data(), BUFFER_SIZE * sizeof(int), gpu::DEVICE_HOST));
  check_array_equal(array_test, array_result);
  DEVICE_GTEST_ASSERT(gpu::deallocate_buffer(result.device_ptr));
}

/**
 * 1) Creates array on host with rand values.
 * 2) Pin the memory of that buffer.
 * 3) Allocate memory on the device
 * 4) Retrieves device pointer corresponding to the pinned memory
 * 5) Copy pinned mem through its device pointer to allocated memory.
 * 6) Copy it on a host buffer.
 * 7) Compare.
 */

TEST(device_resource_loader_test, pin_host_memory) {
  std::array<int, BUFFER_SIZE> array_test = {0};
  generate_random_arrayi(array_test);
  gpu::GPU_query_result pin_result = gpu::pin_host_memory(array_test.data(), BUFFER_SIZE * sizeof(int), gpu::PIN_MODE_DEFAULT);
  DEVICE_GTEST_ASSERT(pin_result);
  gpu::GPU_query_result device_buffer_result = gpu::allocate_buffer(BUFFER_SIZE * sizeof(int));
  DEVICE_GTEST_ASSERT(device_buffer_result);
  gpu::GPU_query_result pinned_mem_dptr = gpu::get_pinned_memory_dptr(array_test.data());
  DEVICE_GTEST_ASSERT(pinned_mem_dptr);
  gpu::GPU_query_result copy_result_D2D = gpu::copy_buffer(
      pinned_mem_dptr.device_ptr, device_buffer_result.device_ptr, BUFFER_SIZE * sizeof(int), gpu::DEVICE_DEVICE);
  DEVICE_GTEST_ASSERT(copy_result_D2D);
  std::array<int, BUFFER_SIZE> result_test = {0};
  gpu::GPU_query_result copy_result_D2H = gpu::copy_buffer(
      device_buffer_result.device_ptr, result_test.data(), BUFFER_SIZE * sizeof(int), gpu::DEVICE_HOST);
  DEVICE_GTEST_ASSERT(copy_result_D2H);
  check_array_equal(array_test, result_test);
}
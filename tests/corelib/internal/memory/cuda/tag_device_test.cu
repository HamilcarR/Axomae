#include "Test.h"
#include "internal/device/gpgpu/kernel_launch_interface.h"
#include "tag_device_test.cuh"
#include <internal/device/gpgpu/device_transfer_interface.h>

/* Tests the correctness of the dynamic dispatch implementation on the device side */

ax_kernel void test_kernel(DispatchDeviceTest *device_data, char *response, int size) {
  int idx = ax_device_thread_idx_x;
  if (idx < size) {
    char c = device_data[idx].getDescription();
    response[idx] = c;
  }
}

static void check_tag_dispatch(const char *const test_data, int size) {
  for (int i = 0; i < size; i++)
    EXPECT_EQ(test_data[i], i + 48);
}

void tag_ptr_kernel_test() {

  std::vector<DispatchDeviceTest> collection;
  DeviceTest0 test1;
  DeviceTest1 test2;
  DeviceTest2 test3;
  DispatchDeviceTest dtest1 = &test1;
  DispatchDeviceTest dtest2 = &test2;
  DispatchDeviceTest dtest3 = &test3;
  collection.push_back(dtest1);
  collection.push_back(dtest2);
  collection.push_back(dtest3);

  kernel_argpack_t argpack;
  argpack.block_size = AX_GPU_WARP_SIZE;
  argpack.num_blocks = 1;
  device::gpgpu::pin_host_memory(&test1, sizeof(DeviceTest0), device::gpgpu::PIN_MODE_DEFAULT);
  device::gpgpu::pin_host_memory(&test2, sizeof(DeviceTest1), device::gpgpu::PIN_MODE_DEFAULT);
  device::gpgpu::pin_host_memory(&test3, sizeof(DeviceTest2), device::gpgpu::PIN_MODE_DEFAULT);
  device::gpgpu::pin_host_memory(collection.data(), sizeof(DispatchDeviceTest) * collection.size(), device::gpgpu::PIN_MODE_DEFAULT);
  char response[3] = {0};
  device::gpgpu::pin_host_memory(response, sizeof(response), device::gpgpu::PIN_MODE_DEFAULT);
  exec_kernel(argpack, test_kernel, collection.data(), response, 3);
  device::gpgpu::unpin_host_memory(&test1);
  device::gpgpu::unpin_host_memory(&test2);
  device::gpgpu::unpin_host_memory(&test3);
  device::gpgpu::unpin_host_memory(collection.data());
  device::gpgpu::unpin_host_memory(response);
  check_tag_dispatch(response, 3);
}
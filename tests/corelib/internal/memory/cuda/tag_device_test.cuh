#ifndef TAG_DEVICE_TEST_CUH
#define TAG_DEVICE_TEST_CUH
#include "internal/device/gpgpu/device_utils.h"
#include "internal/macro/project_macros.h"
#include "internal/memory/tag_ptr.h"
class DeviceTest0 {
 public:
  CLASS_DCM(DeviceTest0)
  ax_device_callable char getDescription() { return '0'; }
};

class DeviceTest1 {
 public:
  CLASS_DCM(DeviceTest1)
  ax_device_callable char getDescription() { return '1'; }
};

class DeviceTest2 {
 public:
  CLASS_DCM(DeviceTest2)
  ax_device_callable char getDescription() { return '2'; }
};

class DispatchDeviceTest : public core::tag_ptr<DeviceTest0, DeviceTest1, DeviceTest2> {
 public:
  using tag_ptr::tag_ptr;
  ax_device_callable char getDescription() {
    auto d = [&](auto *d_test) { return d_test->getDescription(); };
    return dispatch(d);
  }
};

void tag_ptr_kernel_test();

#endif  // TAG_DEVICE_TEST_CUH

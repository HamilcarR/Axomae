#include "../device_acceleration_interface.h"

#ifdef AXOMAE_USE_CUDA
#  include "optix_includes.h"
#endif

namespace nova::aggregate {

  std::unique_ptr<DeviceAcceleratorInterface> DeviceAcceleratorInterface::make() {
    std::unique_ptr<DeviceAcceleratorInterface> accelerator = nullptr;
#ifdef AXOMAE_USE_CUDA
    accelerator = std::make_unique<BackendOptix>();
#endif
    return accelerator;
  }
}  // namespace nova::aggregate
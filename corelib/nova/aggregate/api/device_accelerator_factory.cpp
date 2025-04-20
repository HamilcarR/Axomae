#include "../device_acceleration_interface.h"
#include <internal/debug/Logger.h>
#if defined(AXOMAE_USE_CUDA) && defined(AXOMAE_USE_OPTIX)
#  include "optix/internal.h"
#endif

namespace nova::aggregate {

  std::unique_ptr<DeviceAcceleratorInterface> DeviceAcceleratorInterface::make() {
    std::unique_ptr<DeviceAcceleratorInterface> accelerator = nullptr;
#if defined(AXOMAE_USE_CUDA) && defined(AXOMAE_USE_OPTIX)
    accelerator = std::make_unique<BackendOptix>();
#else
    LOG("Build doesn't provide any valid GPU API for acceleration structures.", LogLevel::ERROR);
#endif
    return accelerator;
  }
}  // namespace nova::aggregate

#ifndef GPU_LAUNCHER_H
#define GPU_LAUNCHER_H
#include "aggregate/device_acceleration_interface.h"
#include "gpu/nova_gpu.h"
#include "manager/ManagerInternalStructs.h"
#include <internal/common/axstd/span.h>

namespace nova {
  namespace gputils {
    struct gpu_util_structures_t;
  }
  struct gpu_shared_data_t {
    std::vector<axstd::span<uint8_t>> buffers;
  };

  void device_start_integrator(const device_traversal_param_s &traversal_parameters, nova_eng_internals &nova_internals);
}  // namespace nova

#endif

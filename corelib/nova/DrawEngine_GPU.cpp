#include "DrawEngine.h"
#include "integrator/Integrator.h"
#include "manager/ManagerInternalStructs.h"
#include <internal/debug/Logger.h>

#if defined(AXOMAE_USE_CUDA)
#  include "gpu/nova_gpu.h"
#  include "integrator/gpu_launcher.h"
#endif

namespace nova {

  void gpu_draw(const device_traversal_param_s &traversal_params, nova_eng_internals &internals) {
#if defined(AXOMAE_USE_CUDA)
    device_start_integrator(traversal_params, internals);
#else
    LOG("Application built without CUDA. Enable AXOMAE_USE_CUDA in build if GPU is compatible.", LogLevel::ERROR);
#endif
  }
}  // namespace nova

#include "DrawEngine.h"
#include "Logger.h"
#include "PerformanceLogger.h"
#include "device_utils.h"
#include "engine/nova_exception.h"
#include "integrator/Integrator.h"
#include "manager/NovaResourceManager.h"

#if defined(AXOMAE_USE_CUDA)
#  include "integrator/GPUIntegrator.cuh"
#endif

namespace nova {

  void gpu_draw(HdrBufferStruct *buffers,
                unsigned width_resolution,
                unsigned height_resolution,
                NovaRenderEngineInterface *engine_interface,
                nova::nova_eng_internals &nova_internal_structs) {
#if defined(AXOMAE_USE_CUDA)
    launch_gpu_kernel(buffers, width_resolution, height_resolution, engine_interface, nova_internal_structs);
#else
    LOG("Application built without CUDA. Enable AXOMAE_USE_CUDA in build if GPU is compatible.", LogLevel::ERROR);
#endif
  }
}  // namespace nova

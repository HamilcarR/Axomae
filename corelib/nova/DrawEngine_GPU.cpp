#include "DrawEngine.h"
#include "PerformanceLogger.h"
#include "device_utils.h"
#include "engine/nova_exception.h"
#include "integrator/Integrator.h"
#include "manager/NovaResourceManager.h"
#include "Logger.h"
#if defined(AXOMAE_USE_CUDA)
#include "gpu/gpu.cuh"
#endif

namespace nova {



  void gpu_draw(HdrBufferStruct *buffers,
                unsigned width_resolution,
                unsigned height_resolution,
                NovaRenderEngineInterface *engine_interface,
                const NovaResourceManager *nova_resources_manager) {
#if defined(AXOMAE_USE_CUDA)
    launch_gpu_kernel(buffers, width_resolution, height_resolution, engine_interface, nova_resources_manager);
#else
    LOG("Application built without CUDA. Enable AXOMAE_USE_CUDA in build if GPU is compatible.", LogLevel::ERROR);
#endif
  }
}




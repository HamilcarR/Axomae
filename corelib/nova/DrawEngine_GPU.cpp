#include "DrawEngine.h"
#include "integrator/Integrator.h"
#include "internal/debug/Logger.h"
#include "manager/NovaResourceManager.h"

#if defined(AXOMAE_USE_CUDA)
#  include "integrator/GPUIntegrator.cuh"
#endif

namespace nova {

  void gpu_draw(HdrBufferStruct *buffers,
                unsigned width_resolution,
                unsigned height_resolution,
                nova::nova_eng_internals &nova_internal_structs,
                gputils::gpu_util_structures_t &gpu_structures) {
#if defined(AXOMAE_USE_CUDA)
    launch_gpu_kernel(buffers, width_resolution, height_resolution, nova_internal_structs, gpu_structures);
#else
    LOG("Application built without CUDA. Enable AXOMAE_USE_CUDA in build if GPU is compatible.", LogLevel::ERROR);
#endif
  }
}  // namespace nova

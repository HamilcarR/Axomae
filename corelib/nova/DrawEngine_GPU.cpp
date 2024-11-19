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
                NovaRenderEngineInterface *engine_interface,
                nova::nova_eng_internals &nova_internal_structs,
                const device_shared_caches_t &shared_buffer_collection) {
#if defined(AXOMAE_USE_CUDA)
    internal_gpu_integrator_shared_host_mem_t shared_buffers;
    shared_buffers.buffers = shared_buffer_collection.contiguous_caches;
    launch_gpu_kernel(buffers, width_resolution, height_resolution, engine_interface, nova_internal_structs, shared_buffers);
#else
    LOG("Application built without CUDA. Enable AXOMAE_USE_CUDA in build if GPU is compatible.", LogLevel::ERROR);
#endif
  }
}  // namespace nova

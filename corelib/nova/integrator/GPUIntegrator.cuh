#ifndef GPU_CUH
#define GPU_CUH
#include "internal/common/axstd/span.h"
namespace nova {

  struct internal_gpu_integrator_shared_host_mem_t {
    std::vector<axstd::span<uint8_t>> buffers;
  };
  void launch_gpu_kernel(HdrBufferStruct *buffers,
                         unsigned screen_width,
                         unsigned screen_height,
                         NovaRenderEngineInterface *engine_interface,
                         nova::nova_eng_internals &nova_internals,
                         const internal_gpu_integrator_shared_host_mem_t &shared_buffer_collection);
}  // namespace nova

#endif

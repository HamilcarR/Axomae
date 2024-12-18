#ifndef GPU_CUH
#define GPU_CUH
#include "internal/common/axstd/span.h"
namespace nova {
  namespace gputils {
    struct gpu_util_structures_t;
  }
  struct gpu_shared_data_t {
    std::vector<axstd::span<uint8_t>> buffers;
  };

  void launch_gpu_kernel(HdrBufferStruct *buffers,
                         unsigned screen_width,
                         unsigned screen_height,
                         NovaRenderEngineInterface *engine_interface,
                         nova_eng_internals &nova_internals,
                         gputils::gpu_util_structures_t &gpu_structures,
                         const gpu_shared_data_t &shared_buffer_collection);
}  // namespace nova

#endif

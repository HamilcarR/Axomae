#ifndef GPU_CUH
#define GPU_CUH
#include "device_utils.h"

namespace nova {

  void launch_gpu_kernel(HdrBufferStruct *buffers,
                         unsigned width_resolution,
                         unsigned height_resolution,
                         NovaRenderEngineInterface *engine_interface,
                         nova::nova_eng_internals &nova_internals);
}  // namespace nova

#endif

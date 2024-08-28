#ifndef GPU_CUH
#define GPU_CUH
#include "device_utils.h"


namespace nova {

  class NovaResourceManager;
  namespace gpu {
    AX_KERNEL void test_func(float *ptr, unsigned width, unsigned height, const NovaResourceManager *nova_resource_manager);
  }

  void launch_gpu_kernel(HdrBufferStruct *buffers,
                         unsigned width_resolution,
                         unsigned height_resolution,
                         NovaRenderEngineInterface *engine_interface,
                         const NovaResourceManager *nova_resources_manager);
}

#endif

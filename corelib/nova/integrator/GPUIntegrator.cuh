#ifndef GPU_CUH
#define GPU_CUH

namespace nova {

  void launch_gpu_kernel(HdrBufferStruct *buffers,
                         unsigned screen_width,
                         unsigned screen_height,
                         NovaRenderEngineInterface *engine_interface,
                         nova::nova_eng_internals &nova_internals);
}  // namespace nova

#endif

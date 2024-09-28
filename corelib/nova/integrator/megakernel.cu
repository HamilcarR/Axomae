#include "DrawEngine.h"
#include "GPUIntegrator.cuh"
#include "Integrator.h"
#include "engine/nova_exception.h"
#include "gpu/ResourceLoader.h"
#include "internal/debug/Logger.h"
#include "internal/debug/PerformanceLogger.h"
#include "internal/device/cuda/CudaDevice.h"
#include "internal/device/device_utils.h"
#include "internal/device/kernel_launch_interface.h"
#include "manager/NovaResourceManager.h"
#include "math/math_texturing.h"

/* Serves only as a baseline for performance to compare against */
namespace nova {
  namespace gpu {
    AX_KERNEL void test_func(float *ptr, const float *host_texture, unsigned width, unsigned height, int i_width, int i_height) {
      unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
      unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
      unsigned int offset = (y * width + x) * 4;
      double u = math::texture::pixelToUv(x, width);
      double v = math::texture::pixelToUv(y, height);
      unsigned h_x = math::texture::uvToPixel(u, i_width);
      unsigned h_y = math::texture::uvToPixel(v, i_height);
      unsigned h_offset = (h_y * i_width + h_x) * 3;
      if (offset < width * height * 4) {
        ptr[offset] = host_texture[h_offset];
        ptr[offset + 1] = host_texture[h_offset + 1];
        ptr[offset + 2] = host_texture[h_offset + 2];
        ptr[offset + 3] = 1.f;
      }
    }
  }  // namespace gpu
  void __attribute((optimize("O0"))) launch_gpu_kernel(HdrBufferStruct *buffers,
                                                       unsigned screen_width,
                                                       unsigned screen_height,
                                                       NovaRenderEngineInterface *engine_interface,
                                                       nova::nova_eng_internals &nova_internals) {
    if (ax_cuda::utils::cuda_info_device().empty()) {
      LOGS("No suitable gpu detected.");
      nova_internals.exception_manager->addError(nova::exception::GENERAL_GPU_ERROR);
      return;
    }
    namespace resrc = nova::gpu::resrc;
    const NovaResourceManager *resource_manager = nova_internals.resource_manager;
    const texturing::TextureRawData image_texture = resource_manager->getEnvmapData();
    std::size_t alloc_size = image_texture.width * image_texture.height * image_texture.channels * sizeof(float);
    std::size_t screen_size = screen_width * screen_height * buffers->channels * sizeof(float);
    resrc::GPUResource resource = resrc::allocate(alloc_size);
    AXCUDA_ERROR_CHECK(resource.error_status);
    resrc::GPUResource draw_buffer = resrc::allocate(screen_size);
    AXCUDA_ERROR_CHECK(draw_buffer.error_status);
    if (resource.device_ptr) {
      AXCUDA_ERROR_CHECK(resrc::copy(image_texture.raw_data, resource.device_ptr, alloc_size, 0).error_status);
    }
    kernel_argpack_t argpack;
    argpack.num_blocks = {screen_width, screen_height, 1};
    argpack.block_size = {32, 32, 1};

    exec_kernel(argpack,
                gpu::test_func,
                (float *)draw_buffer.device_ptr,
                (const float *)resource.device_ptr,
                screen_width,
                screen_height,
                image_texture.width,
                image_texture.height);
    AXCUDA_ERROR_CHECK(resrc::copy(draw_buffer.device_ptr, buffers->partial_buffer, screen_size, 1).error_status);
    AXCUDA_ERROR_CHECK(resrc::deallocate(resource.device_ptr).error_status);
    AXCUDA_ERROR_CHECK(resrc::deallocate(draw_buffer.device_ptr).error_status);
  }
}  // namespace nova

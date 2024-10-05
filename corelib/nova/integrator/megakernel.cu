#include "DrawEngine.h"
#include "GPUIntegrator.cuh"
#include "Integrator.h"
#include "engine/nova_exception.h"
#include "internal/common/math/math_texturing.h"
#include "internal/debug/Logger.h"
#include "internal/device/gpgpu/cuda/CudaDevice.h"
#include "internal/device/gpgpu/device_transfer_interface.h"
#include "internal/device/gpgpu/device_utils.h"
#include "internal/device/gpgpu/kernel_launch_interface.h"
#include "manager/NovaResourceManager.h"

namespace resrc = device::gpgpu;

/* Serves only as a baseline for performance to compare against */
namespace nova {
  namespace gpu {
    AX_KERNEL void test_func(float *ptr, cudaTextureObject_t host_texture, unsigned width, unsigned height, int i_width, int i_height) {
      unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
      unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
      unsigned int offset = (y * width + x) * 4;

      if (offset < width * height * 4) {
        float u = math::texture::pixelToUv(x, width);
        float v = math::texture::pixelToUv(y, height);

        float4 sample{};
        sample = tex2D<float4>(host_texture, u, v);
        ptr[offset] = sample.x;
        ptr[offset + 1] = sample.y;
        ptr[offset + 2] = sample.z;
        ptr[offset + 3] = 1.f;
      }
    }
  }  // namespace gpu

  static void setup_descriptors(resrc::texture_descriptor &tex_desc, resrc::resource_descriptor &resrc_desc) {
    resrc::channel_format &ch_format = tex_desc.channel_descriptor;
    ch_format.bits_size_x = 32;
    ch_format.bits_size_y = 32;
    ch_format.bits_size_z = 32;
    ch_format.bits_size_a = 32;
    ch_format.format_type = resrc::FLOAT;

    tex_desc.filter_mode = resrc::FILTER_LINEAR;
    tex_desc.read_mode = resrc::READ_ELEMENT_TYPE;
    tex_desc.address_mode[0] = tex_desc.address_mode[1] = resrc::ADDRESS_BORDER;
    tex_desc.normalized_coords = true;
    resrc_desc.resource_buffer_descriptors.res.array.array = nullptr;
    resrc_desc.type = resrc::RESOURCE_ARRAY;
  }

  void launch_gpu_kernel(HdrBufferStruct *buffers,
                         unsigned screen_width,
                         unsigned screen_height,
                         NovaRenderEngineInterface *engine_interface,
                         nova::nova_eng_internals &nova_internals) {

    if (ax_cuda::utils::cuda_info_device().empty()) {
      LOGS("No suitable gpu detected.");
      nova_internals.exception_manager->addError(nova::exception::GENERAL_GPU_ERROR);
      return;
    }

    const NovaResourceManager *resource_manager = nova_internals.resource_manager;
    const texturing::TextureRawData image_texture = resource_manager->getEnvmapData();
    std::size_t screen_size = screen_width * screen_height * buffers->channels * sizeof(float);

    resrc::texture_descriptor tex_desc{};
    resrc::resource_descriptor res_desc{};
    setup_descriptors(tex_desc, res_desc);
    resrc::GPU_texture texture_resrc = resrc::create_texture(
        (const void *)image_texture.raw_data, image_texture.width, image_texture.height, tex_desc, res_desc);
    resrc::GPU_query_result draw_buffer = resrc::allocate_buffer(screen_size);
    AXCUDA_ERROR_CHECK(draw_buffer.error_status);

    kernel_argpack_t argpack;
    argpack.num_blocks = {screen_width / 32, screen_height, 1};
    argpack.block_size = {32, 1, 1};
    exec_kernel(argpack,
                gpu::test_func,
                (float *)draw_buffer.device_ptr,
                std::any_cast<cudaTextureObject_t>(texture_resrc.texture_object),
                screen_width,
                screen_height,
                image_texture.width,
                image_texture.height);
    AXCUDA_ERROR_CHECK(resrc::copy_buffer(draw_buffer.device_ptr, buffers->partial_buffer, screen_size, 1).error_status);
    AXCUDA_ERROR_CHECK(resrc::deallocate_buffer(draw_buffer.device_ptr).error_status);
    resrc::destroy_texture(texture_resrc);
  }
}  // namespace nova

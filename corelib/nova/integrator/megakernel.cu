#include "DrawEngine.h"
#include "GPUIntegrator.cuh"
#include "Integrator.h"
#include "engine/nova_exception.h"
#include "gpu/GPURandomGenerator.h"
#include "internal/common/math/math_texturing.h"
#include "internal/debug/Logger.h"
#include "internal/device/gpgpu/cuda/CudaDevice.h"
#include "internal/device/gpgpu/device_transfer_interface.h"
#include "internal/device/gpgpu/device_utils.h"
#include "internal/device/gpgpu/kernel_launch_interface.h"
#include "manager/NovaResourceManager.h"

#include <curand_kernel.h>

namespace resrc = device::gpgpu;

/* Serves only as a baseline for performance to compare against */
namespace nova {
  namespace gpu {
    ax_kernel void test_func(float *ptr,
                             cudaTextureObject_t host_texture,
                             unsigned width,
                             unsigned height,
                             int i_width,
                             int i_height,
                             texturing::NovaTextureInterface other_tex,
                             curandStateScrambledSobol32 *rand_array,
                             unsigned seed) {
      unsigned int x = ax_device_thread_idx_x;
      unsigned int y = ax_device_thread_idx_y;
      unsigned int offset = (y * width + x) * 4;

      if (offset >= width * height * 4)
        return;
      float u = math::texture::pixelToUv(x, width);
      float v = math::texture::pixelToUv(y, height);

#ifdef __CUDA_ARCH__
      int px = 0, py = 0;
      sampler::SobolSampler sampler = sampler::SobolSampler(rand_array, seed);
      glm::vec3 sample = sampler.sample() * 0.5f + 0.5f;
      px = math::texture::uvToPixel(sample.x, width);
      py = math::texture::uvToPixel(sample.y, height);
      offset = (py * width + px) * 4;
      ptr[offset] = 1.f;
      ptr[offset + 1] = 1.f;
      ptr[offset + 2] = 1.f;
      ptr[offset + 3] = 1.f;
#endif
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
                         nova::nova_eng_internals &nova_internals,
                         const internal_gpu_integrator_shared_host_mem_t &shared_host_caches) {

    if (ax_cuda::utils::cuda_info_device().empty()) {
      LOGS("No suitable gpu detected.");
      nova_internals.exception_manager->addError(nova::exception::GENERAL_GPU_ERROR);
      return;
    }

    const NovaResourceManager *resource_manager = nova_internals.resource_manager;
    const texturing::TextureRawData image_texture = resource_manager->getEnvmapData();
    std::size_t screen_size = screen_width * screen_height * buffers->channels * sizeof(float);
    texturing::NovaTextureInterface some_image = resource_manager->getTexturesData().get_textures()[0];
    resrc::texture_descriptor tex_desc{};
    resrc::resource_descriptor res_desc{};
    setup_descriptors(tex_desc, res_desc);
    resrc::GPU_texture texture_resrc = resrc::create_texture(
        (const void *)image_texture.raw_data, image_texture.width, image_texture.height, tex_desc, res_desc);
    resrc::GPU_query_result draw_buffer = resrc::allocate_buffer(screen_size);
    DEVICE_ERROR_CHECK(draw_buffer.error_status);
    kernel_argpack_t argpack;
    argpack.num_blocks = {screen_width / 32, screen_height, 1};
    argpack.block_size = {32, 1, 1};
    math::random::qrand_alloc_result_t alloc = math::random::GPUQuasiRandomGenerator::init(argpack, 5);

    exec_kernel(argpack,
                gpu::test_func,
                (float *)draw_buffer.device_ptr,
                std::any_cast<cudaTextureObject_t>(texture_resrc.texture_object),
                screen_width,
                screen_height,
                image_texture.width,
                image_texture.height,
                some_image,
                alloc.states_array,
                alloc.dimension);

    DEVICE_ERROR_CHECK(resrc::copy_buffer(draw_buffer.device_ptr, buffers->partial_buffer, screen_size, resrc::DEVICE_HOST).error_status);
    DEVICE_ERROR_CHECK(resrc::deallocate_buffer(draw_buffer.device_ptr).error_status);
    resrc::destroy_texture(texture_resrc);
    math::random::GPUQuasiRandomGenerator::cleanStates(alloc.states_array);
  }
}  // namespace nova

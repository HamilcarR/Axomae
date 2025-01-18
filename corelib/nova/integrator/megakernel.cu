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
#include <curand_kernel.h>
#include <internal/common/math/gpu/math_random_gpu.h>

/* Serves only as a baseline for performance to compare against */

namespace resrc = device::gpgpu;
namespace utils = nova::gputils;

namespace nova {
  namespace gpu {

    struct render_buffer_t {
      float *render_target;
      unsigned width;
      unsigned height;
    };

    ax_device_only void shade(render_buffer_t &render_buffer, float u, float v, glm::vec4 color) {
      unsigned px = math::texture::uvToPixel(u, render_buffer.width);
      unsigned py = math::texture::uvToPixel(v, render_buffer.height);
      unsigned offset = (py * render_buffer.width + px) * 4;
      render_buffer.render_target[offset] = color.r;
      render_buffer.render_target[offset + 1] = color.g;
      render_buffer.render_target[offset + 2] = color.b;
      render_buffer.render_target[offset + 3] = color.a;
    }
    ax_device_only void shade(render_buffer_t &render_buffer, unsigned x, unsigned y, glm::vec4 color) {
      unsigned offset = (y * render_buffer.width + x) * 4;
      render_buffer.render_target[offset] = color.r;
      render_buffer.render_target[offset + 1] = color.g;
      render_buffer.render_target[offset + 2] = color.b;
      render_buffer.render_target[offset + 3] = color.a;
    }

    ax_device_only void shade(render_buffer_t &render_buffer, unsigned offset, glm::vec4 color) {
      render_buffer.render_target[offset] = color.r;
      render_buffer.render_target[offset + 1] = color.g;
      render_buffer.render_target[offset + 2] = color.b;
      render_buffer.render_target[offset + 3] = color.a;
    }

    ax_kernel void test_func(
        render_buffer_t render_buffer, cudaTextureObject_t host_texture, int i_width, int i_height, utils::gpu_random_generator_t generator) {
      unsigned int x = ax_device_thread_idx_x;
      unsigned int y = ax_device_thread_idx_y;
      if (!AX_GPU_IN_BOUNDS_2D(x, y, render_buffer.width, render_buffer.height))
        return;
      sampler::SobolSampler sobol_sampler = sampler::SobolSampler(generator.sobol);
      sampler::RandomSampler random_sampler = sampler::RandomSampler(generator.xorshift);
      glm::vec3 sobol{}, random{};
      sobol = sobol_sampler.sample(0, 1);
      random = random_sampler.sample(0, 1);
      nova::shape::Triangle tri;

      const Object3D *mesh = tri.getMesh();
      if (ax_device_linearRM3D_idx < mesh->uv.size())
        printf("warp id:%u ::: lane id:%u ::: i:%d  ::: value:%f\n",
               ax_device_warp_id,
               ax_device_lane_id,
               ax_device_linearRM3D_idx,
               mesh->uv[ax_device_linearRM3D_idx]);
      float u = (float)math::texture::pixelToUv(x, render_buffer.width - 1);
      float v = (float)math::texture::pixelToUv(y, render_buffer.height - 1);
      float4 tex = tex2D<float4>(host_texture, u, v);
      shade(render_buffer, sobol.y, sobol.z, glm::vec4(tex.x, tex.y, tex.z, 1.f));
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
    tex_desc.address_mode[0] = tex_desc.address_mode[1] = resrc::ADDRESS_WRAP;
    tex_desc.normalized_coords = true;
    resrc_desc.resource_buffer_descriptors.res.array.array = nullptr;
    resrc_desc.type = resrc::RESOURCE_ARRAY;
  }

  void launch_gpu_kernel(HdrBufferStruct *buffers,
                         unsigned screen_width,
                         unsigned screen_height,
                         nova::nova_eng_internals &nova_internals,
                         gputils::gpu_util_structures_t &gpu_structures) {

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
    resrc::GPU_texture texture_resrc = resrc::create_texture(image_texture.raw_data, image_texture.width, image_texture.height, tex_desc, res_desc);

    resrc::GPU_query_result draw_buffer = resrc::allocate_buffer(screen_size);  // TODO: Bottleneck. Generate only 1 buffer for the whole app life
    DEVICE_ERROR_CHECK(draw_buffer.error_status);
    gpu::render_buffer_t render_buffer{static_cast<float *>(draw_buffer.device_ptr), screen_width, screen_height};

    exec_kernel(gpu_structures.threads_distribution,
                gpu::test_func,
                render_buffer,
                std::any_cast<cudaTextureObject_t>(texture_resrc.texture_object),
                image_texture.width,
                image_texture.height,
                gpu_structures.random_generator);
    DEVICE_ERROR_CHECK(resrc::copy_buffer(draw_buffer.device_ptr, buffers->partial_buffer, screen_size, resrc::DEVICE_HOST).error_status);
    resrc::synchronize_device();
    DEVICE_ERROR_CHECK(resrc::deallocate_buffer(draw_buffer.device_ptr).error_status);
    resrc::destroy_texture(texture_resrc);
  }
}  // namespace nova

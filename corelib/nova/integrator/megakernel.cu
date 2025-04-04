#include "DrawEngine.h"
#include "GPUIntegrator.cuh"
#include "Integrator.h"
#include "engine/nova_exception.h"
#include "manager/NovaResourceManager.h"
#include <curand_kernel.h>
#include <internal/common/math/math_texturing.h>
#include <internal/debug/Logger.h>
#include <internal/device/gpgpu/cuda/CudaDevice.h>
#include <internal/device/gpgpu/device_transfer_interface.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/device/gpgpu/kernel_launch_interface.h>
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

    struct integrator_args_s {
      utils::gpu_random_generator_t generator;
      nova::shape::MeshCtx geometry_context;
      texturing::TextureCtx texture_context;
      axstd::span<const nova::material::NovaMaterialInterface> materials;
      axstd::span<const primitive::NovaPrimitiveInterface> primitives;
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

    ax_kernel void test_func(render_buffer_t render_buffer, integrator_args_s args) {
      unsigned int x = ax_device_thread_idx_x;
      unsigned int y = ax_device_thread_idx_y;
      if (!AX_GPU_IN_BOUNDS_2D(x, y, render_buffer.width, render_buffer.height))
        return;

      float u = (float)math::texture::pixelToUv(x, render_buffer.width - 1);
      float v = (float)math::texture::pixelToUv(y, render_buffer.height - 1);
      auto mat = args.materials[0];
      Ray in;
      in.origin = {0, 0, 0};
      in.direction = {0, -1, 0};
      Ray out;
      hit_data hit_d;
      auto sampler = sampler::SobolSampler(args.generator.sobol);
      sampler::SamplerInterface sampler_interface = &sampler;
      material::shading_data_s shading_data;
      texturing::texture_data_aggregate_s texture_data_aggregate;
      texture_data_aggregate.texture_ctx = &args.texture_context;
      shading_data.texture_aggregate = &texture_data_aggregate;
      auto prim = args.primitives[0];
      bool b = prim.scatter(in, out, hit_d, sampler_interface, shading_data);
      shade(render_buffer, x, y, glm::vec4(1.f));
    }
  }  // namespace gpu

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
    std::size_t screen_size = screen_width * screen_height * buffers->channels * sizeof(float);

    resrc::GPU_query_result draw_buffer = resrc::allocate_buffer(screen_size);  // TODO: Bottleneck. Generate only 1 buffer for the whole app life
    DEVICE_ERROR_CHECK(draw_buffer.error_status);
    gpu::render_buffer_t render_buffer{static_cast<float *>(draw_buffer.device_ptr), screen_width, screen_height};
    gpu::integrator_args_s arguments;
    arguments.generator = gpu_structures.random_generator;
    arguments.geometry_context = nova::shape::MeshCtx(nova_internals.resource_manager->getShapeData().getMeshSharedViews());
    arguments.materials = resource_manager->getMaterialData().getMaterialView();
    arguments.primitives = resource_manager->getPrimitiveData().getPrimitiveView();
    arguments.texture_context = resource_manager->getTexturesData().getTextureBundleViews();
    exec_kernel(gpu_structures.threads_distribution, gpu::test_func, render_buffer, arguments);
    DEVICE_ERROR_CHECK(resrc::copy_buffer(draw_buffer.device_ptr, buffers->partial_buffer, screen_size, resrc::DEVICE_HOST).error_status);
    resrc::synchronize_device();
    DEVICE_ERROR_CHECK(resrc::deallocate_buffer(draw_buffer.device_ptr).error_status);
  }
}  // namespace nova

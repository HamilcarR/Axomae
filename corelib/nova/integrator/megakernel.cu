#include "DrawEngine.h"
#include "GPUIntegrator.cuh"
#include "Integrator.h"
#include "aggregate/device_acceleration_interface.h"
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

    ax_kernel void test_func(render_buffer_t render_buffer, integrator_args_s args) {
      unsigned int x = ax_device_thread_idx_x;
      unsigned int y = ax_device_thread_idx_y;

      if (!AX_GPU_IN_BOUNDS_2D(x, y, render_buffer.width, render_buffer.height))
        return;
      float u = (float)math::texture::pixelToUv(x, render_buffer.width - 1);
      float v = (float)math::texture::pixelToUv(y, render_buffer.height - 1);

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

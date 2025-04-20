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

namespace resrc = device::gpgpu;
namespace utils = nova::gputils;

namespace nova {

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

    exec_kernel(gpu_structures.threads_distribution, gpu::test_func, render_buffer, arguments);
    DEVICE_ERROR_CHECK(resrc::copy_buffer(draw_buffer.device_ptr, buffers->partial_buffer, screen_size, resrc::DEVICE_HOST).error_status);
    resrc::synchronize_device();
    DEVICE_ERROR_CHECK(resrc::deallocate_buffer(draw_buffer.device_ptr).error_status);
  }
}  // namespace nova

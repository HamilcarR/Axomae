#include "CudaDevice.h"
#include "DrawEngine.h"
#include "Logger.h"
#include "PerformanceLogger.h"
#include "device_utils.h"
#include "engine/nova_exception.h"
#include "gpu.cuh"
#include "integrator/Integrator.h"
#include "kernel_launch_interface.h"
#include "manager/NovaResourceManager.h"
namespace nova {
  namespace gpu {
    AX_KERNEL void test_func(float *ptr, unsigned width, unsigned height, NovaExceptionManager *nova_internals) {
      nova_internals->addError(exception::INVALID_INTEGRATOR);
    }
  }  // namespace gpu
  void launch_gpu_kernel(HdrBufferStruct *buffers,
                         unsigned width_resolution,
                         unsigned height_resolution,
                         NovaRenderEngineInterface *engine_interface,
                         nova::nova_eng_internals &nova_internals) {
    if (ax_cuda::utils::cuda_info_device().empty()) {
      LOGS("No suitable gpu detected.");
      nova_internals.exception_manager->addError(nova::exception::GENERAL_GPU_ERROR);
      return;
    }
    ax_cuda::CudaDevice device;
    ax_cuda::CudaParams params;
    device.GPUFree(0);
    params.setFlags(0x01);
    float *host_buffer = buffers->accumulator_buffer;
    NovaExceptionManager *host_exception = nova_internals.exception_manager;

    AXCUDA_ERROR_CHECK(device.GPUHostRegister(host_buffer, width_resolution * height_resolution * sizeof(float) * 4, cudaHostAllocMapped));
    AXCUDA_ERROR_CHECK(device.GPUHostRegister(host_exception, sizeof(NovaExceptionManager), cudaHostAllocMapped));

    kernel_argpack_t argpack{};
    argpack.num_blocks.x = width_resolution;
    argpack.num_blocks.y = 4;
    argpack.block_size = {32, 1, 1};

    exec_kernel(argpack, gpu::test_func, host_buffer, width_resolution, height_resolution, host_exception);

    AXCUDA_ERROR_CHECK(device.GPUDeviceSynchronize());
    AXCUDA_ERROR_CHECK(device.GPUHostUnregister(host_buffer));
    AXCUDA_ERROR_CHECK(device.GPUHostUnregister(host_exception));
  }
}  // namespace nova
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

    AX_KERNEL void test_func(float *ptr, unsigned width, unsigned height, nova::nova_eng_internals &nova_internals) {
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
      int idx = y * width + x;
      if (idx < width * height)
        ptr[idx] = 1.f;
      nova_internals.exception_manager->addError(exception::INVALID_INTEGRATOR);
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
    params.setFlags(1);
    float *device_buffer = nullptr, *host_buffer = buffers->accumulator_buffer;
    AXCUDA_ERROR_CHECK(device.GPUHostRegister(host_buffer, width_resolution * height_resolution * sizeof(float) * 4, cudaHostAllocMapped));
    AXCUDA_ERROR_CHECK(device.GPUHostGetDevicePointer((void **)&device_buffer, host_buffer, 0));
    AXCUDA_ERROR_CHECK(device.GPUMallocManaged((void **)&nova_internals, sizeof(NovaResourceManager), params));

    kernel_argpack_t argpack{};
    argpack.num_blocks.x = 4;
    argpack.num_blocks.y = 4;
    argpack.block_size = {10, 1, 1};
    PerformanceLogger perf;
    perf.startTimer();
    exec_kernel(argpack, gpu::test_func, host_buffer, width_resolution, height_resolution, nova_internals);
    AXCUDA_ERROR_CHECK(device.GPUDeviceSynchronize());
    perf.endTimer();
    perf.print();

    AXCUDA_ERROR_CHECK(device.GPUHostUnregister(host_buffer));
    AXCUDA_ERROR_CHECK(device.GPUFree((void *)&nova_internals));
  }

}  // namespace nova
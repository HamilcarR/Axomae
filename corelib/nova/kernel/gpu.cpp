#include "CudaDevice.h"
#include "DrawEngine.h"

#include "PerformanceLogger.h"
#include "device_utils.h"
#include "integrator/Integrator.h"
#include "manager/NovaResourceManager.h"

namespace nova {
  AX_KERNEL void test_func(float *ptr, unsigned width, unsigned height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * width + x;

    if (idx < width * height) {
      for (int i = 0; i < 3; i++)
        ptr[idx * 4 + i];
    }
  }

  void gpu_draw(HdrBufferStruct *buffers,
                unsigned width_resolution,
                unsigned height_resolution,
                NovaRenderEngineInterface *engine_interface,
                const NovaResourceManager *nova_resources_manager) {

    if (ax_cuda::utils::cuda_info_device().empty()) {
      LOGS("No suitable gpu detected.");
      nova_resources_manager->addError(nova::exception::GENERAL_GPU_ERROR);
      return;
    }

    ax_cuda::CudaDevice device;
    ax_cuda::CudaParams params;
    device.deallocateMemory(0);
    params.setFlags(1);
    float *device_buffer = nullptr, *host_buffer = buffers->accumulator_buffer;
    AXCUDA_ERROR_CHECK(device.registerMemoryHost(host_buffer, width_resolution * height_resolution * sizeof(float) * 4, cudaHostAllocMapped));
    AXCUDA_ERROR_CHECK(device.getHostDevicePtr((void **)&device_buffer, host_buffer, 0));
    kernel_argpack_t argpack{};
    argpack.num_blocks.x = width_resolution;
    argpack.num_blocks.y = height_resolution;
    argpack.block_size = {1, 1, 1};
    PerformanceLogger perf;
    perf.startTimer();
    exec_kernel(argpack, test_func, host_buffer, width_resolution, height_resolution);
    AXCUDA_ERROR_CHECK(device.synchronize());
    perf.endTimer();
    perf.print();

    AXCUDA_ERROR_CHECK(device.unregisterMemoryHost(host_buffer));
  }
}  // namespace nova

#include "ResourceLoader.h"

#include "CudaDevice.h"
#include "device_utils.h"

namespace nova::gpu::resrc {

  bool validate_gpu_state() { return !ax_cuda::utils::cuda_info_device().empty(); }
  GPUResource ret_error() {
    GPUResource gpu_resource;
    gpu_resource.device_ptr = nullptr;
    gpu_resource.error_status = DeviceError(cudaErrorInvalidDevice);
    return gpu_resource;
  }

  GPUResource allocate(size_t size) {
    if (!validate_gpu_state())
      return ret_error();
    ax_cuda::CudaDevice device;
    GPUResource gpu_resource;
    gpu_resource.error_status = device.GPUMalloc(&gpu_resource.device_ptr, size);
    return gpu_resource;
  }

  GPUResource copy(const void *src, void *dest, std::size_t buffer_size, int copy_type) {
    if (!validate_gpu_state())
      return ret_error();
    ax_cuda::CudaDevice device;
    GPUResource gpu_resource;
    ax_cuda::CudaParams params;
    switch (copy_type) {
      case 0:
        params.setMemcpyKind(cudaMemcpyHostToDevice);
        break;
      case 1:
        params.setMemcpyKind(cudaMemcpyDeviceToHost);
        break;
      case 2:
        params.setMemcpyKind(cudaMemcpyDeviceToDevice);
        break;
      default:
        params.setMemcpyKind(cudaMemcpyHostToDevice);
        break;
    }
    gpu_resource.error_status = device.GPUMemcpy(src, dest, buffer_size, params);
    gpu_resource.device_ptr = dest;
    return gpu_resource;
  }

  GPUResource deallocate(void *device_ptr) {
    if (!validate_gpu_state())
      return ret_error();
    ax_cuda::CudaDevice device;
    GPUResource gpu_resource;
    gpu_resource.error_status = device.GPUFree(device_ptr);
    return gpu_resource;
  }

}  // namespace nova::gpu::resrc
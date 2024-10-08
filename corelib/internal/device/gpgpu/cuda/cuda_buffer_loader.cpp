#include "../device_transfer_interface.h"

#include "../device_utils.h"
#include "CudaDevice.h"

namespace device::gpgpu {

  bool validate_gpu_state() { return !ax_cuda::utils::cuda_info_device().empty(); }
  GPU_query_result ret_error() {
    GPU_query_result gpu_resource;
    gpu_resource.device_ptr = nullptr;
    return gpu_resource;
  }

  GPU_query_result allocate_buffer(size_t size) {
    if (!validate_gpu_state())
      return ret_error();
    ax_cuda::CudaDevice device;
    GPU_query_result gpu_resource;
    device.GPUMalloc(&gpu_resource.device_ptr, size);
    return gpu_resource;
  }

  GPU_query_result copy_buffer(const void *src, void *dest, std::size_t buffer_size, COPY_TYPE copy_type) {
    if (!validate_gpu_state())
      return ret_error();
    ax_cuda::CudaDevice device;
    GPU_query_result gpu_resource;
    ax_cuda::CudaParams params;
    switch (copy_type) {
      case HOST_DEVICE:
        params.setMemcpyKind(cudaMemcpyHostToDevice);
        break;
      case DEVICE_HOST:
        params.setMemcpyKind(cudaMemcpyDeviceToHost);
        break;
      case DEVICE_DEVICE:
        params.setMemcpyKind(cudaMemcpyDeviceToDevice);
        break;
      default:
        params.setMemcpyKind(cudaMemcpyHostToDevice);
        break;
    }
    device.GPUMemcpy(src, dest, buffer_size, params);
    gpu_resource.device_ptr = dest;
    return gpu_resource;
  }

  GPU_query_result deallocate_buffer(void *device_ptr) {
    if (!validate_gpu_state())
      return ret_error();
    ax_cuda::CudaDevice device;
    GPU_query_result gpu_resource;
    device.GPUFree(device_ptr);
    return gpu_resource;
  }

}  // namespace device::gpgpu
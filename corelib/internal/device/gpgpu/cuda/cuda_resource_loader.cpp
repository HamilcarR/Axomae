#include "../device_resource_interface.h"
#include "../device_utils.h"
#include "CudaDevice.h"

namespace device::gpgpu::resource {

  bool validate_gpu_state() { return !ax_cuda::utils::cuda_info_device().empty(); }
  GPU_resource ret_error() {
    GPU_resource gpu_resource;
    gpu_resource.device_ptr = nullptr;
    gpu_resource.error_status = DeviceError(cudaErrorInvalidDevice);
    return gpu_resource;
  }

  GPU_resource allocate_buffer(size_t size) {
    if (!validate_gpu_state())
      return ret_error();
    ax_cuda::CudaDevice device;
    GPU_resource gpu_resource;
    gpu_resource.error_status = device.GPUMalloc(&gpu_resource.device_ptr, size);
    return gpu_resource;
  }

  GPU_resource copy_buffer(const void *src, void *dest, std::size_t buffer_size, int copy_type) {
    if (!validate_gpu_state())
      return ret_error();
    ax_cuda::CudaDevice device;
    GPU_resource gpu_resource;
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

  GPU_resource deallocate_buffer(void *device_ptr) {
    if (!validate_gpu_state())
      return ret_error();
    ax_cuda::CudaDevice device;
    GPU_resource gpu_resource;
    gpu_resource.error_status = device.GPUFree(device_ptr);
    return gpu_resource;
  }

  static int get_channels_num(const texture_channel_descriptor &desc) {
    int i = 0;
    i += desc.bits_size_x ? 1 : 0;
    i += desc.bits_size_y ? 1 : 0;
    i += desc.bits_size_z ? 1 : 0;
    i += desc.bits_size_a ? 1 : 0;
    return i;
  }

  static std::size_t texture_type_size(const texture_channel_descriptor &desc) {
    switch (desc.texture_type) {
      case FLOAT:
        return sizeof(float);
      default:
        return sizeof(uint8_t);
    }
  }

  static void init_descriptors(cudaArray_t cuda_array, ax_cuda::CudaParams &cuda_device_params) {
    cudaResourceDesc resource_descriptor{};
    cudaTextureDesc texture_descriptor{};
    resource_descriptor.resType = cudaResourceTypeArray;
    resource_descriptor.res.array.array = cuda_array;
    // Initialize texture descriptors
    texture_descriptor.addressMode[0] = cudaAddressModeBorder;
    texture_descriptor.addressMode[1] = cudaAddressModeBorder;
    texture_descriptor.filterMode = cudaFilterModeLinear;
    texture_descriptor.readMode = cudaReadModeElementType;
    texture_descriptor.normalizedCoords = 1;
    cuda_device_params.setResourceDesc(resource_descriptor);
    cuda_device_params.setTextureDesc(texture_descriptor);
  }

  static cudaChannelFormatKind get_texture_type(TEXTURE_TYPE type) {
    switch (type) {
      case FLOAT:
        return cudaChannelFormatKindFloat;
      default:
        return cudaChannelFormatKindUnsignedNormalized8X4;
    }
  }

  GPU_texture __attribute((optimize("O0"))) create_texture(const void *src, int width, int height, const texture_channel_descriptor &desc) {
    ax_cuda::CudaDevice device;
    GPU_texture gpu_texture;
    ax_cuda::CudaParams params;
    cudaArray_t cuda_array = nullptr;
    params.setChanDescriptors(desc.bits_size_x, desc.bits_size_y, desc.bits_size_z, desc.bits_size_a, get_texture_type(desc.texture_type));
    AXCUDA_ERROR_CHECK(device.GPUMallocArray(&cuda_array, params, width, height));
    params.setMemcpyKind(cudaMemcpyHostToDevice);
    std::size_t pitch = width * get_channels_num(desc) * texture_type_size(desc);
    AXCUDA_ERROR_CHECK(device.GPUMemcpy2DToArray(cuda_array, 0, 0, src, pitch, pitch, height, params));
    init_descriptors(cuda_array, params);
    cudaTextureObject_t texture_object = 0;
    AXCUDA_ERROR_CHECK(device.GPUCreateTextureObject(&texture_object, params));
    gpu_texture.texture_object = texture_object;
    gpu_texture.array_object = cuda_array;
    return gpu_texture;
  }

  void destroy_texture(GPU_texture &texture) {
    ax_cuda::CudaDevice device;
    AXCUDA_ERROR_CHECK(device.GPUDestroyTextureObject(std::any_cast<cudaTextureObject_t>(texture.texture_object)));
    AXCUDA_ERROR_CHECK(device.GPUFreeArray(std::any_cast<cudaArray_t>(texture.array_object)));
  }

}  // namespace device::gpgpu::resource
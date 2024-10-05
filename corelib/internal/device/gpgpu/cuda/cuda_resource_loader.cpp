#include "../device_transfer_interface.h"

#include "../device_utils.h"
#include "CudaDevice.h"

namespace device::gpgpu {

  bool validate_gpu_state() { return !ax_cuda::utils::cuda_info_device().empty(); }
  GPU_query_result ret_error() {
    GPU_query_result gpu_resource;
    gpu_resource.device_ptr = nullptr;
    gpu_resource.error_status = DeviceError(cudaErrorInvalidDevice);
    return gpu_resource;
  }

  GPU_query_result allocate_buffer(size_t size) {
    if (!validate_gpu_state())
      return ret_error();
    ax_cuda::CudaDevice device;
    GPU_query_result gpu_resource;
    gpu_resource.error_status = device.GPUMalloc(&gpu_resource.device_ptr, size);
    return gpu_resource;
  }

  GPU_query_result copy_buffer(const void *src, void *dest, std::size_t buffer_size, int copy_type) {
    if (!validate_gpu_state())
      return ret_error();
    ax_cuda::CudaDevice device;
    GPU_query_result gpu_resource;
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

  GPU_query_result deallocate_buffer(void *device_ptr) {
    if (!validate_gpu_state())
      return ret_error();
    ax_cuda::CudaDevice device;
    GPU_query_result gpu_resource;
    gpu_resource.error_status = device.GPUFree(device_ptr);
    return gpu_resource;
  }

  static int get_channels_num(const channel_format &desc) {
    int i = 0;
    i += desc.bits_size_x ? 1 : 0;
    i += desc.bits_size_y ? 1 : 0;
    i += desc.bits_size_z ? 1 : 0;
    i += desc.bits_size_a ? 1 : 0;
    return i;
  }

  static std::size_t texture_type_size(const channel_format &desc) {
    switch (desc.format_type) {
      case FLOAT:
        return sizeof(float);
      default:
        return sizeof(uint8_t);
    }
  }

  static cudaResourceType get_resource_type(const resource_descriptor &desc) {
    switch (desc.type) {
      case RESOURCE_ARRAY:
        return cudaResourceTypeArray;
      case RESOURCE_MIPMAP_ARRAY:
        return cudaResourceTypeMipmappedArray;
      case RESOURCE_LINEAR:
        return cudaResourceTypeLinear;
      case RESOURCE_PITCH2D:
        return cudaResourceTypePitch2D;
      default:
        LOG("Invalid descriptor", LogLevel::ERROR);
        return cudaResourceTypeArray;
    }
  }

  static cudaTextureAddressMode get_address_mode(ADDRESS_MODE adrs_mode) {
    switch (adrs_mode) {
      case ADDRESS_WRAP:
        return cudaAddressModeWrap;
      case ADDRESS_MIRROR:
        return cudaAddressModeMirror;
      case ADDRESS_CLAMP:
        return cudaAddressModeClamp;
      case ADDRESS_BORDER:
        return cudaAddressModeBorder;
      default:
        LOG("Invalid address mode", LogLevel::ERROR);
        return cudaAddressModeWrap;
    }
  }

  static void set_address_mode(cudaTextureAddressMode address_mode[3], const texture_descriptor &desc) {
    address_mode[0] = get_address_mode(desc.address_mode[0]);
    address_mode[1] = get_address_mode(desc.address_mode[1]);
    address_mode[2] = get_address_mode(desc.address_mode[2]);
  }

  static cudaTextureFilterMode get_filter_mode(const texture_descriptor &texture_descriptor) {
    switch (texture_descriptor.filter_mode) {
      case FILTER_POINT:
        return cudaFilterModePoint;
      case FILTER_LINEAR:
        return cudaFilterModeLinear;
      default:
        LOG("Invalid filter mode", LogLevel::ERROR);
        return cudaFilterModePoint;
    }
  }

  static cudaTextureReadMode get_read_mode(const texture_descriptor &texture_descriptor) {
    switch (texture_descriptor.read_mode) {
      case READ_ELEMENT_TYPE:
        return cudaReadModeElementType;
      case READ_NORMALIZED_FLOAT:
        return cudaReadModeNormalizedFloat;
      default:
        LOG("Invalid read mode", LogLevel::ERROR);
        return cudaReadModeElementType;
    }
  }

  static void set_resource_struct(const resource_descriptor &desc, cudaResourceDesc &cuda_desc) {
    switch (desc.type) {
      case RESOURCE_ARRAY:
        cuda_desc.res.array.array = static_cast<cudaArray_t>(desc.resource_buffer_descriptors.res.array.array);
        break;
      default:
        LOG("Unsupported resource type", LogLevel::ERROR);
        break;
    }
  }

  static void init_descriptors(ax_cuda::CudaParams &cuda_device_params, const texture_descriptor &tex_desc, const resource_descriptor &resc_desc) {
    cudaResourceDesc cuda_resource_descriptor{};
    cudaTextureDesc cuda_texture_descriptor{};
    cuda_resource_descriptor.resType = get_resource_type(resc_desc);
    set_resource_struct(resc_desc, cuda_resource_descriptor);
    // Initialize texture descriptors
    set_address_mode(cuda_texture_descriptor.addressMode, tex_desc);
    cuda_texture_descriptor.filterMode = get_filter_mode(tex_desc);
    cuda_texture_descriptor.readMode = get_read_mode(tex_desc);
    cuda_texture_descriptor.normalizedCoords = tex_desc.normalized_coords;
    cuda_device_params.setResourceDesc(cuda_resource_descriptor);
    cuda_device_params.setTextureDesc(cuda_texture_descriptor);
  }

  static cudaChannelFormatKind convert_type(FORMAT_TYPE type) {
    switch (type) {
      case FLOAT:
        return cudaChannelFormatKindFloat;
      default:
        return cudaChannelFormatKindUnsignedNormalized8X4;
    }
  }

  GPU_resource create_array(int width, int height, const channel_format &chan_format, int flag) {
    cudaArray_t array;
    ax_cuda::CudaDevice device;
    ax_cuda::CudaParams params;
    params.setChanDescriptors(
        chan_format.bits_size_x, chan_format.bits_size_y, chan_format.bits_size_z, chan_format.bits_size_a, convert_type(chan_format.format_type));
    AXCUDA_ERROR_CHECK(device.GPUMallocArray(&array, params, width, height, flag));
    GPU_resource resource{};
    resource.res.array.array = array;
    return resource;
  }

  void destroy_array(GPU_resource &resource) {
    ax_cuda::CudaDevice device;
    AXCUDA_ERROR_CHECK(device.GPUFreeArray(static_cast<cudaArray_t>(resource.res.array.array)));
  }

  GPU_texture create_texture(const void *src, int width, int height, const texture_descriptor &tex_desc, resource_descriptor &resc_desc) {
    ax_cuda::CudaDevice device;
    GPU_texture gpu_texture;
    ax_cuda::CudaParams params;
    const channel_format &channel_format = tex_desc.channel_descriptor;
    params.setChanDescriptors(channel_format.bits_size_x,
                              channel_format.bits_size_y,
                              channel_format.bits_size_z,
                              channel_format.bits_size_a,
                              convert_type(channel_format.format_type));
    GPU_resource gpu_resource = resc_desc.resource_buffer_descriptors;
    if (!gpu_resource.res.array.array)
      gpu_resource = create_array(width, height, tex_desc.channel_descriptor, 0);
    resc_desc.resource_buffer_descriptors.res.array.array = gpu_resource.res.array.array;
    auto cuda_array = static_cast<cudaArray_t>(resc_desc.resource_buffer_descriptors.res.array.array);

    params.setMemcpyKind(cudaMemcpyHostToDevice);
    std::size_t pitch = width * get_channels_num(channel_format) * texture_type_size(channel_format);
    AXCUDA_ERROR_CHECK(device.GPUMemcpy2DToArray(cuda_array, 0, 0, src, pitch, pitch, height, params));
    init_descriptors(params, tex_desc, resc_desc);
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

}  // namespace device::gpgpu
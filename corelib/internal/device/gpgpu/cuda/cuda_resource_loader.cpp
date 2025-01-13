#include "../device_transfer_interface.h"
#include "../device_utils.h"
#include "CudaDevice.h"
#include <cuda_gl_interop.h>

namespace device::gpgpu {
  bool validate_gpu_state() { return !ax_cuda::utils::cuda_info_device().empty(); }
  GPU_query_result ret_error() {
    GPU_query_result gpu_resource;
    gpu_resource.device_ptr = nullptr;
    gpu_resource.error_status = DeviceError(cudaErrorDeviceUninitialized);
    return gpu_resource;
  }

  GPU_query_result synchronize_device() {
    if (!validate_gpu_state())
      return ret_error();
    ax_cuda::CudaDevice device;
    GPU_query_result gpu_resource;
    gpu_resource.error_status = device.GPUDeviceSynchronize();
    gpu_resource.device_ptr = nullptr;
    return gpu_resource;
  }

  GPU_query_result allocate_symbol(void **sym, std::size_t size_bytes) {
    if (!validate_gpu_state())
      return ret_error();
    ax_cuda::CudaDevice device;
    GPU_query_result gpu_resource;
    gpu_resource.error_status = device.GPUMalloc(sym, size_bytes);
    gpu_resource.device_ptr = sym;
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

  inline cudaMemcpyKind get_kind(COPY_MODE copy_type) {
    switch (copy_type) {
      case HOST_HOST:
        return cudaMemcpyHostToHost;
      case HOST_DEVICE:
        return cudaMemcpyHostToDevice;
        break;
      case DEVICE_HOST:
        return cudaMemcpyDeviceToHost;
        break;
      case DEVICE_DEVICE:
        return cudaMemcpyDeviceToDevice;
        break;
      default:
        return cudaMemcpyDefault;
        break;
    }
  }

  GPU_query_result copy_buffer(const void *src, void *dest, std::size_t buffer_size, COPY_MODE copy_type) {
    if (!validate_gpu_state())
      return ret_error();
    ax_cuda::CudaDevice device;
    GPU_query_result gpu_resource;
    ax_cuda::CudaParams params;
    params.setMemcpyKind(get_kind(copy_type));
    gpu_resource.error_status = device.GPUMemcpy(src, dest, buffer_size, params);
    gpu_resource.device_ptr = dest;
    return gpu_resource;
  }

  GPU_query_result copy_to_symbol(const void *src, void *symbol, std::size_t buffer_size_bytes, COPY_MODE copy_type) {
    if (!validate_gpu_state())
      return ret_error();
    ax_cuda::CudaDevice device;
    GPU_query_result gpu_resource;
    ax_cuda::CudaParams params;
    params.setMemcpyKind(get_kind(copy_type));
    gpu_resource.error_status = device.GPUMemcpyToSym(src, symbol, buffer_size_bytes, 0, params);
    gpu_resource.device_ptr = symbol;
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
      case UINT8X4N:
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
    DEVICE_ERROR_CHECK(device.GPUMallocArray(&array, params, width, height, flag));
    GPU_resource resource{};
    resource.res.array.array = array;
    return resource;
  }

  GPU_query_result destroy_array(GPU_resource &resource) {
    ax_cuda::CudaDevice device;
    GPU_query_result result;
    result.error_status = device.GPUFreeArray(static_cast<cudaArray_t>(resource.res.array.array));
    return result;
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
    DEVICE_ERROR_CHECK(device.GPUMemcpy2DToArray(cuda_array, 0, 0, src, pitch, pitch, height, params));
    init_descriptors(params, tex_desc, resc_desc);
    cudaTextureObject_t texture_object = 0;
    DEVICE_ERROR_CHECK(device.GPUCreateTextureObject(&texture_object, params));
    gpu_texture.texture_object = texture_object;
    gpu_texture.array_object = cuda_array;
    return gpu_texture;
  }

  void destroy_texture(GPU_texture &texture) {
    ax_cuda::CudaDevice device;
    DEVICE_ERROR_CHECK(device.GPUDestroyTextureObject(std::any_cast<cudaTextureObject_t>(texture.texture_object)));
    DEVICE_ERROR_CHECK(device.GPUFreeArray(std::any_cast<cudaArray_t>(texture.array_object)));
  }

  unsigned int get_cuda_host_register_flag(PIN_MODE mode) {
    switch (mode) {
      case PIN_MODE_DEFAULT:
        return cudaHostRegisterDefault;
      case PIN_MODE_PORTABLE:
        return cudaHostRegisterPortable;
      case PIN_MODE_MAPPED:
        return cudaHostRegisterMapped;
      case PIN_MODE_IO:
        return cudaHostRegisterIoMemory;
      case PIN_MODE_RO:
        return cudaHostRegisterReadOnly;
      default:
        return cudaHostRegisterDefault;
    }
  }

  GPU_query_result pin_host_memory(void *buffer, std::size_t size_bytes, PIN_MODE mode) {
    GPU_query_result result;
    ax_cuda::CudaDevice device;
    result.error_status = device.GPUHostRegister(buffer, size_bytes, get_cuda_host_register_flag(mode));
    result.device_ptr = buffer;
    return result;
  }

  GPU_query_result unpin_host_memory(void *buffer) {
    GPU_query_result result;
    ax_cuda::CudaDevice device;
    result.error_status = device.GPUHostUnregister(buffer);
    result.device_ptr = nullptr;
    return result;
  }

  GPU_query_result get_pinned_memory_dptr(void *host, PIN_EXT mode) {
    GPU_query_result result;
    ax_cuda::CudaDevice device;
    result.error_status = device.GPUHostGetDevicePointer(&result.device_ptr, host, 0); /* Keep flag to 0 */
    return result;
  }

  static cudaGraphicsRegisterFlags get_interop_read_flag(ACCESS_TYPE access_type) {
    switch (access_type) {
      case READ_WRITE:
        return cudaGraphicsRegisterFlagsNone;
        break;
      case WRITE_ONLY:
        return cudaGraphicsRegisterFlagsWriteDiscard;
        break;
      case READ_ONLY:
        return cudaGraphicsRegisterFlagsReadOnly;
        break;
      default:
        return cudaGraphicsRegisterFlagsNone;
        break;
    }
  }

  GPU_query_result interop_register_glbuffer(GLuint vbo_id, ACCESS_TYPE access_type) {
    GPU_query_result result;
    ax_cuda::CudaDevice device;
    result.device_ptr = nullptr;
    unsigned flag = get_interop_read_flag(access_type);
    cudaGraphicsResource *dev_ptr;
    result.error_status = DeviceError(cudaGraphicsGLRegisterBuffer(&dev_ptr, vbo_id, flag));
    result.device_ptr = dev_ptr;
    return result;
  }

  GPU_query_result interop_register_glimage(GLuint tex_id, GLenum target, ACCESS_TYPE access_type) {
    GPU_query_result result;
    ax_cuda::CudaDevice device;
    result.device_ptr = nullptr;
    unsigned flag = get_interop_read_flag(access_type);
    cudaGraphicsResource *dev_ptr;
    result.error_status = DeviceError(cudaGraphicsGLRegisterImage(&dev_ptr, tex_id, target, flag));
    result.device_ptr = dev_ptr;
    return result;
  }

  GPU_query_result interop_map_resrc(int count, void **gpu_resources_array, void *stream) {
    GPU_query_result result;
    ax_cuda::CudaDevice device;
    result.error_status = DeviceError(
        cudaGraphicsMapResources(count, reinterpret_cast<cudaGraphicsResource_t *>(gpu_resources_array), static_cast<cudaStream_t>(stream)));
    return result;
  }

  GPU_query_result interop_unmap_resrc(int count, void **gpu_resources_array, void *stream) {
    GPU_query_result result;
    ax_cuda::CudaDevice device;
    result.error_status = DeviceError(
        cudaGraphicsUnmapResources(count, reinterpret_cast<cudaGraphicsResource_t *>(gpu_resources_array), static_cast<cudaStream_t>(stream)));
    return result;
  }

  GPU_query_result interop_get_mapped_ptr(void *gpu_resources) {
    GPU_query_result result;
    ax_cuda::CudaDevice device;
    result.error_status = DeviceError(
        cudaGraphicsResourceGetMappedPointer(&result.device_ptr, &result.size, static_cast<cudaGraphicsResource_t>(gpu_resources)));
    return result;
  }

  GPU_query_result interop_unregister_resrc(void *gpu_resource) {
    GPU_query_result result;
    ax_cuda::CudaDevice device;
    result.error_status = DeviceError(cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource_t>(gpu_resource)));
    return result;
  }

}  // namespace device::gpgpu
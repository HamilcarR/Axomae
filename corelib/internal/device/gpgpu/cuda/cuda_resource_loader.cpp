// clang-format off
#include <internal/device/rendering/opengl/gl_headers.h>
// clang-format on
#include "../device_transfer_interface.h"
#include "../device_utils.h"
#include "CudaDevice.h"
#include "private/utils.h"
#include <cuda_gl_interop.h>
#include <vector>

namespace device::gpgpu {
  bool validate_gpu_state() { return !ax_cuda::utils::cuda_info_device().empty(); }
  GPU_query_result ret_error() {
    GPU_query_result gpu_resource;
    gpu_resource.device_ptr = nullptr;
    gpu_resource.error_status = DeviceError(cudaErrorDeviceUninitialized);
    return gpu_resource;
  }

  /*******************************************************************************************************************************************************************************/
  /** Context Sharing **/
  class GPUContext::Impl {
   public:
    CUcontext context;
    Impl() = default;
    explicit Impl(CUcontext context) : context(context) {}
  };

  GPUContext::GPUContext() : pimpl(std::make_unique<Impl>()) {}
  GPUContext::~GPUContext() = default;
  GPUContext::GPUContext(GPUContext &&) noexcept = default;
  GPUContext &GPUContext::operator=(GPUContext &&) noexcept = default;

  void init_driver_API() { AX_ASSERT_EQ(cuInit(0), CUDA_SUCCESS); }
  void create_context(GPUContext &context) { AX_ASSERT_EQ(cuCtxCreate(&context.pimpl->context, CU_CTX_SCHED_AUTO, 0), CUDA_SUCCESS); }
  void get_current_context(GPUContext &context) { AX_ASSERT_EQ(cuCtxGetCurrent(&context.pimpl->context), CUDA_SUCCESS); }
  void set_current_context(const GPUContext &context) { AX_ASSERT_EQ(cuCtxSetCurrent(context.pimpl->context), CUDA_SUCCESS); }
  void pop_context(GPUContext &context) { AX_ASSERT_EQ(cuCtxPopCurrent(&context.pimpl->context), CUDA_SUCCESS); }
  void push_context(GPUContext &context) { AX_ASSERT_EQ(cuCtxPushCurrent(context.pimpl->context), CUDA_SUCCESS); }

  /*******************************************************************************************************************************************************************************/
  /** Device Synchronizations **/

  GPU_query_result synchronize_device() {
    if (!validate_gpu_state())
      return ret_error();
    ax_cuda::CudaDevice device;
    GPU_query_result gpu_resource;
    gpu_resource.error_status = device.GPUDeviceSynchronize();
    gpu_resource.device_ptr = nullptr;
    return gpu_resource;
  }

  /*******************************************************************************************************************************************************************************/
  /** Streams **/

  class GPUStream::Impl {
   public:
    cudaStream_t stream{};

    Impl() = default;
    explicit Impl(cudaStream_t stream) : stream(stream) {}
  };
  GPUStream::GPUStream() : pimpl(std::make_unique<Impl>()) {}
  GPUStream::~GPUStream() = default;
  GPUStream::GPUStream(GPUStream &&) noexcept = default;
  GPUStream &GPUStream::operator=(GPUStream &&) noexcept = default;

  /*******************************************************************************************************************************************************************************/
  /** Generic buffers allocation and deallocation **/

  GPU_query_result allocate_device_managed(std::size_t buffer_size_byte, bool global_access) {
    if (!validate_gpu_state())
      return ret_error();
    GPU_query_result query_result;
    query_result.error_status = DeviceError(
        cudaMallocManaged(&query_result.device_ptr, buffer_size_byte, global_access ? cudaMemAttachGlobal : cudaMemAttachHost));
    query_result.size = buffer_size_byte;
    return query_result;
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

  cudaMemcpyKind get_kind(COPY_MODE copy_type) {
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

  GPU_query_result async_copy_buffer(const void *src, void *dest, std::size_t buffer_size, COPY_MODE copy_type) {
    cudaMemcpyKind kind = get_kind(copy_type);
    DeviceError error = cudaMemcpyAsync(dest, src, buffer_size, kind);
    DEVICE_ERROR_CHECK(error);
    GPU_query_result result;
    result.error_status = error;
    result.device_ptr = dest;
    result.size = buffer_size;
    return result;
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

  /*******************************************************************************************************************************************************************************/
  /** Textures **/

  int get_channels_num(const channel_format &desc) {
    int i = 0;
    i += desc.bits_size_x ? 1 : 0;
    i += desc.bits_size_y ? 1 : 0;
    i += desc.bits_size_z ? 1 : 0;
    i += desc.bits_size_a ? 1 : 0;
    return i;
  }

  std::size_t texture_type_size(const channel_format &desc) {
    switch (desc.format_type) {
      case FLOAT:
        return sizeof(float);
      default:
        return sizeof(uint8_t);
    }
  }

  cudaResourceType get_resource_type(const resource_descriptor &desc) {
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

  cudaTextureAddressMode get_address_mode(ADDRESS_MODE adrs_mode) {
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

  void set_address_mode(cudaTextureAddressMode address_mode[3], const texture_descriptor &desc) {
    address_mode[0] = get_address_mode(desc.address_mode[0]);
    address_mode[1] = get_address_mode(desc.address_mode[1]);
    address_mode[2] = get_address_mode(desc.address_mode[2]);
  }

  cudaTextureFilterMode get_filter_mode(const texture_descriptor &texture_descriptor) {
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

  cudaTextureReadMode get_read_mode(const texture_descriptor &texture_descriptor) {
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

  void set_resource_struct(const resource_descriptor &desc, cudaResourceDesc &cuda_desc) {
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

  cudaChannelFormatKind convert_type(FORMAT_TYPE type) {
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

  /*******************************************************************************************************************************************************************************/
  /** Unified memory access */

  class DeviceSharedBufferView::Impl {
   public:
    void *buffer{};
    std::size_t size{};  // In bytes
    bool is_pinned{};    // could use

    Impl() = default;
    Impl(void *buffer, std::size_t size) : buffer(buffer), size(size) {}
  };

  DeviceSharedBufferView::DeviceSharedBufferView() : pimpl(std::make_shared<Impl>()) {}
  DeviceSharedBufferView::~DeviceSharedBufferView() = default;
  DeviceSharedBufferView::DeviceSharedBufferView(DeviceSharedBufferView &&) noexcept = default;
  DeviceSharedBufferView &DeviceSharedBufferView::operator=(DeviceSharedBufferView &&) noexcept = default;
  std::size_t DeviceSharedBufferView::bufferSizeBytes() const { return pimpl->size; }
  void *DeviceSharedBufferView::getCastData() const { return pimpl->buffer; }
  void DeviceSharedBufferView::initBuffer(void *buffer, std::size_t size) { pimpl = std::make_shared<Impl>(buffer, size); }
  bool DeviceSharedBufferView::isMapped() const { return pimpl->is_pinned; }

  template<>
  std::size_t DeviceSharedBufferView::size<void>() const {
    return bufferSizeBytes();
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

  GPU_query_result pin_host_memory(DeviceSharedBufferView &buffer, PIN_MODE mode) {
    GPU_query_result result;
    result.error_status = DeviceError(cudaHostRegister(buffer.data<void>(), buffer.size<void>(), get_cuda_host_register_flag(mode)));
    result.device_ptr = buffer.data<void>();
    if (result.error_status.isValid())
      buffer.pimpl->is_pinned = true;
    return result;
  }

  GPU_query_result unpin_host_memory(DeviceSharedBufferView &buffer) {
    GPU_query_result result;
    ax_cuda::CudaDevice device;
    result.error_status = DeviceError(cudaHostUnregister(buffer.data<void>()));
    result.device_ptr = nullptr;
    if (result.error_status.isValid())
      buffer.pimpl->is_pinned = false;
    return result;
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

  /*******************************************************************************************************************************************************************************/
  /** OpenGL<-> CUDA interop **/

  class GPUGraphicsResrcHandle::Impl {
   public:
    cudaGraphicsResource_t handle{};
    bool is_mapped{false};
    bool is_registered{false};
    Impl() = default;
    explicit Impl(cudaGraphicsResource_t resource_) : handle(resource_) {}
  };

  GPUGraphicsResrcHandle::GPUGraphicsResrcHandle() : pimpl(std::make_unique<Impl>()) {}
  GPUGraphicsResrcHandle::~GPUGraphicsResrcHandle() = default;
  GPUGraphicsResrcHandle::GPUGraphicsResrcHandle(GPUGraphicsResrcHandle &&) noexcept = default;
  GPUGraphicsResrcHandle &GPUGraphicsResrcHandle::operator=(GPUGraphicsResrcHandle &&) noexcept = default;

  bool GPUGraphicsResrcHandle::isMapped() const { return pimpl && pimpl->is_mapped; }
  bool GPUGraphicsResrcHandle::isRegistered() const { return pimpl && pimpl->is_registered; }

  class GPUArray::Impl {
   public:
    cudaArray_t array{};

    Impl() = default;
    explicit Impl(cudaArray_t array_) : array(array_) {};
  };

  cudaArray_t &retrieve_internal(const GPUArray &array) { return array.pimpl->array; }

  GPUArray::GPUArray() : pimpl(std::make_unique<Impl>()) {}
  GPUArray::~GPUArray() = default;
  GPUArray::GPUArray(GPUArray &&) noexcept = default;
  GPUArray &GPUArray::operator=(GPUArray &&) noexcept = default;

  cudaGraphicsRegisterFlags get_interop_read_flag(ACCESS_TYPE access_type) {
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

  GPU_query_result interop_register_glbuffer(GLuint vbo_id, GPUGraphicsResrcHandle &graphics_resrc, ACCESS_TYPE access_type) {
    GPU_query_result result{};
    unsigned flag = get_interop_read_flag(access_type);
    result.error_status = DeviceError(cudaGraphicsGLRegisterBuffer(&graphics_resrc.pimpl->handle, vbo_id, flag));
    graphics_resrc.pimpl->is_registered = result.error_status.isValid();
    return result;
  }

  GPU_query_result interop_register_glimage(GLuint tex_id, GLenum target, GPUGraphicsResrcHandle &graphics_resrc, ACCESS_TYPE access_type) {
    GPU_query_result result{};
    unsigned flag = get_interop_read_flag(access_type);
    result.error_status = DeviceError(cudaGraphicsGLRegisterImage(&graphics_resrc.pimpl->handle, tex_id, target, flag));
    graphics_resrc.pimpl->is_registered = result.error_status.isValid();
    return result;
  }

  GPU_query_result interop_map_resrc(int count, GPUGraphicsResrcHandle *gpu_resources_array, GPUStream &stream) {
    GPU_query_result result;
    std::vector<cudaGraphicsResource_t> resources;
    resources.reserve(count);
    for (int i = 0; i < count; i++)
      resources.push_back(gpu_resources_array[i].pimpl->handle);
    result.error_status = DeviceError(cudaGraphicsMapResources(count, resources.data(), stream.pimpl->stream));
    for (int i = 0; i < count; i++)
      gpu_resources_array[i].pimpl->is_mapped = result.error_status.isValid();
    return result;
  }

  GPU_query_result interop_unmap_resrc(int count, GPUGraphicsResrcHandle *gpu_resources_array, GPUStream &stream) {
    GPU_query_result result;
    std::vector<cudaGraphicsResource_t> resources;
    resources.reserve(count);
    for (int i = 0; i < count; i++)
      resources.push_back(gpu_resources_array[i].pimpl->handle);
    result.error_status = DeviceError(cudaGraphicsUnmapResources(count, resources.data(), stream.pimpl->stream));
    for (int i = 0; i < count; i++)
      if (result.error_status.isValid())
        gpu_resources_array[i].pimpl->is_mapped = false;
    ;
    return result;
  }

  GPU_query_result interop_get_mapped_array(GPUArray &texture, GPUGraphicsResrcHandle &gpu_graphics_resource, unsigned idx, unsigned mip) {
    GPU_query_result result{};
    result.error_status = DeviceError(cudaGraphicsSubResourceGetMappedArray(&texture.pimpl->array, gpu_graphics_resource.pimpl->handle, idx, mip));
    return result;
  }

  GPU_query_result interop_get_mapped_ptr(GPUGraphicsResrcHandle &gpu_resources) {
    GPU_query_result result;
    result.error_status = DeviceError(cudaGraphicsResourceGetMappedPointer(&result.device_ptr, &result.size, gpu_resources.pimpl->handle));
    return result;
  }

  GPU_query_result interop_unregister_resrc(GPUGraphicsResrcHandle &gpu_resource) {
    GPU_query_result result;
    result.error_status = DeviceError(cudaGraphicsUnregisterResource(gpu_resource.pimpl->handle));
    gpu_resource.pimpl->is_registered = result.error_status.isValid();
    return result;
  }

  GPU_query_result create_stream(GPUStream &stream) {
    GPU_query_result result;
    result.error_status = DeviceError(cudaStreamCreate(&stream.pimpl->stream));
    return result;
  }

}  // namespace device::gpgpu

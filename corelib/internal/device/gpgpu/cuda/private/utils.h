#ifndef TYPE_DISPATCH_H
#define TYPE_DISPATCH_H
#include <driver_types.h>
#include <internal/device/gpgpu/device_transfer_interface.h>

namespace device::gpgpu {

  cudaMemcpyKind get_kind(COPY_MODE copy_type);

  int get_channels_num(const channel_format &desc);

  std::size_t texture_type_size(const channel_format &desc);

  cudaResourceType get_resource_type(const resource_descriptor &desc);

  cudaTextureAddressMode get_address_mode(ADDRESS_MODE adrs_mode);

  void set_address_mode(cudaTextureAddressMode address_mode[3], const texture_descriptor &desc);

  cudaTextureFilterMode get_filter_mode(const texture_descriptor &texture_descriptor);

  cudaTextureReadMode get_read_mode(const texture_descriptor &texture_descriptor);

  void set_resource_struct(const resource_descriptor &desc, cudaResourceDesc &cuda_desc);

  cudaChannelFormatKind convert_type(FORMAT_TYPE type);

  unsigned int get_cuda_host_register_flag(PIN_MODE mode);

  cudaGraphicsRegisterFlags get_interop_read_flag(ACCESS_TYPE access_type);

  cudaArray_t &retrieve_internal(const GPUArray &array);

}  // namespace device::gpgpu
#endif  // TYPE_DISPATCH_H

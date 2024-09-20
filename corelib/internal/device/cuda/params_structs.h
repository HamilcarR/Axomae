#ifndef PARAMS_STRUCTS_H
#define PARAMS_STRUCTS_H

namespace internals {

  struct device_params_t {
    int device_id{};
    unsigned deviceFlags{};
    unsigned flags{};
  };

  struct memory_params_t {
    cudaMemcpyKind memcpy_kind;
  };

  struct channel_descriptor_params_t {
    cudaChannelFormatDesc format_desc;
  };

  struct resource_descriptor_params_t {
    cudaResourceDesc resource_desc;
    cudaResourceViewDesc resource_view_desc;
  };

  struct texture_descriptor_params_t {
    cudaTextureDesc texture_desc;
  };

}  // namespace internals

#endif  // PARAMS_STRUCTS_H

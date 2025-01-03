#ifndef CUDAPARAMS_H
#define CUDAPARAMS_H
#include "cuda_utils.h"
#include "internal/macro/project_macros.h"
#include "params_structs.h"

namespace ax_cuda {

  class CudaParams {

   private:
    internals::device_params_t device_params{};
    internals::memory_params_t memory_params{};
    internals::channel_descriptor_params_t chan_descriptor_params{};
    internals::resource_descriptor_params_t resource_descriptor_params{};
    internals::texture_descriptor_params_t texture_descriptor_params{};

   public:
    CLASS_CM(CudaParams)

    ax_no_discard int getDeviceID() const { return device_params.device_id; }
    void setDeviceID(int device_id) { this->device_params.device_id = device_id; }
    ax_no_discard unsigned getDeviceFlags() const { return device_params.deviceFlags; }
    void setDeviceFlags(unsigned device_flags) { device_params.deviceFlags = device_flags; }
    ax_no_discard unsigned getFlags() const { return device_params.flags; }
    void setFlags(unsigned flags) { this->device_params.flags = flags; }
    void setMemcpyKind(cudaMemcpyKind copy_kind);
    ax_no_discard const cudaMemcpyKind &getMemcpyKind() const { return memory_params.memcpy_kind; }
    void setChanDescriptors(int x, int y, int z, int a, cudaChannelFormatKind kind);
    ax_no_discard cudaChannelFormatDesc getChanDescriptors() const;
    void setResourceDesc(const cudaResourceDesc &resc) { resource_descriptor_params.resource_desc = resc; }
    ax_no_discard const cudaResourceDesc &getResourceDesc() const { return resource_descriptor_params.resource_desc; }
    void setTextureDesc(const cudaTextureDesc &desc) { texture_descriptor_params.texture_desc = desc; }
    ax_no_discard const cudaTextureDesc &getTextureDesc() const { return texture_descriptor_params.texture_desc; }
    ax_no_discard const cudaResourceViewDesc &getResourceViewDesc() const { return resource_descriptor_params.resource_view_desc; }
    void setResourceViewDesc(const cudaResourceViewDesc &view_desc) { resource_descriptor_params.resource_view_desc = view_desc; }
  };
}  // namespace ax_cuda
#endif  // CUDAPARAMS_H

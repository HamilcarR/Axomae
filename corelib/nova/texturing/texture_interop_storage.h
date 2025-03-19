#ifndef TEXTURE_INTEROP_STORAGE_H
#define TEXTURE_INTEROP_STORAGE_H
#include "interfaces/DeviceReferenceStorageInterface.h"
#include "texture_datastructures.h"
#include <internal/common/axstd/managed_buffer.h>
#include <internal/common/utils.h>
#include <vector>

#ifdef AXOMAE_USE_CUDA
#  include "gpu/DeviceImageTracker.h"
#endif

namespace nova::gpu {
  template<class T>
  class DeviceImageTracker;
}

namespace nova::texturing {

  class DummyBufferTracker {
   public:
    DummyBufferTracker() = default;
    DummyBufferTracker(GLuint, device::gpgpu::ACCESS_TYPE) {}
  };
  template<class T = device::gpgpu::GPUTexture, bool using_gpu = core::build::is_gpu_build>
  using DeviceBuffer = std::conditional_t<using_gpu, gpu::DeviceImageTracker<T>, DummyBufferTracker>;

  struct device_storage {
    // Texture buffers residing on GPU
    std::vector<DeviceBuffer<>> rgba_device_buffers{};
    axstd::managed_vector<device::gpgpu::APITextureHandle> rgba_textures{};  // Points to gpu texture objects from the built backend
  };

  struct host_storage {
    std::vector<U32Texture> rgba_textures{};
  };

  class HostPolicy {
   protected:
    host_storage cpu_storage;

   public:
    std::size_t add(const U32Texture &host_stored_texture);
    CstU32ImgTexView getHost() const;
  };

  /* As with the shape module , we need a reference to host stored resources for many reasons.
   * 1) Mainly, we need to keep in memory the width and height of the texture as this is not retrievable only from interop functions.
   * 2) We still need the ability to render fully on CPU even though the application has been built with gpgpu support.
   */
  class DevicePolicy {
   protected:
    device_storage gpu_storage;
    host_storage cpu_storage;

   public:
    std::size_t add(const U32Texture &host_stored_texture, GLuint gl_stored_texture_id);
    CstU32ImgTexView getHost() const;
    IntopImgTexView getDevice() const;
  };

  template<class StoragePolicy = std::conditional_t<core::build::is_gpu_build, DevicePolicy, HostPolicy>>
  class TextureReferencesStorage : public StoragePolicy, public DeviceReferenceStorageInterface {
    std::size_t container_capacity{};

   public:
    CLASS_CM(TextureReferencesStorage)

    void allocate(std::size_t num_textures) override;
    void clear() override;
    void mapBuffers() override;
    void mapResources() override;
    void release() override;
    std::size_t size() const override;

    u32tex_shared_views_s getU32TexturesViews() const;
  };

  using DefaultTextureReferencesStorage = TextureReferencesStorage<>;

}  // namespace nova::texturing

#endif  // TEXTURE_INTEROP_STORAGE_H

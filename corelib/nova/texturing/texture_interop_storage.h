#ifndef TEXTURE_INTEROP_STORAGE_H
#define TEXTURE_INTEROP_STORAGE_H
#include "gpu/DeviceImageTracker.h"
#include "texture_datastructures.h"
#include <algorithm>
#include <cstddef>
#include <interfaces/DeviceReferenceStorageInterface.h>
#include <internal/common/axstd/managed_buffer.h>
#include <internal/common/utils.h>
#include <internal/device/gpgpu/device_resource_descriptors.h>
#include <vector>

namespace nova::texturing {

  class DummyBufferTracker {
   public:
    DummyBufferTracker() = default;
    DummyBufferTracker(GLuint, GLenum, device::gpgpu::ACCESS_TYPE) {}
  };
  template<class T = device::gpgpu::GPUTexture, bool using_gpu = core::build::is_gpu_build>
  using DeviceBuffer = std::conditional_t<using_gpu, gpu::DeviceImageTracker<T>, DummyBufferTracker>;

  /*
   * Since removing an element from storages can change the indices of the other textures of that collection,
   * We simply register the indices of textures marked as unused so that the next add operation could register in that free index spot.
   */
  template<class T>
  class IdRemovalTracker {
    static constexpr std::size_t INVALID_INDEX = std::size_t(-1);

   protected:
    std::vector<std::size_t> empty_ids{};

   public:
    void disableID(std::size_t element_id) { empty_ids.push_back(element_id); }

    static bool valid(std::size_t index) { return index != INVALID_INDEX; }

    std::size_t getFirstFreeID() const {
      if (empty_ids.empty())
        return INVALID_INDEX;
      else
        return empty_ids[0];
    }

    void useID(std::size_t element_id) {
      auto iter = std::find_if(empty_ids.begin(), empty_ids.end(), [&element_id](const auto &test) { return element_id == test; });
      if (iter != empty_ids.end())
        empty_ids.erase(iter);
    }

    void reset() { empty_ids.clear(); }
  };

  /* Although host_storage is destined for a CPU only build, we use managed vectors in case interops are not available.
   * So instead of accessing to texture through GL handles, we would access them through unified memory.
   * In that case, our buffers would be located host-side, yet accessible still by the GPU.
   */
  struct host_storage {
    axstd::managed_vector<U32Texture> rgba_textures{};
    IdRemovalTracker<U32Texture> u32_id_tracker;
    axstd::managed_vector<F32Texture> frgba_textures{};
    IdRemovalTracker<F32Texture> f32_id_tracker;
  };

  /* Storage description available CPU only build. */
  class HostPolicy {
   protected:
    host_storage cpu_storage;
    CstU32ImgTexView U32Array() const;
    CstF32ImgTexView F32Array() const;

   public:
    std::size_t add(const U32Texture &host_stored_texture);
    std::size_t add(const F32Texture &host_stored_texture);
    void removeF32(std::size_t f32_texture_id);
    void removeU32(std::size_t u32_texture_id);
  };

  class DeviceStorageInterface {  // TODO: plan the same design for host_storage as well for better testability.
   public:
    virtual ~DeviceStorageInterface() = default;
    virtual std::size_t addF32(GLuint texture_id, GLenum format, device::gpgpu::ACCESS_TYPE access_type) = 0;
    virtual std::size_t addU32(GLuint texture_id, GLenum format, device::gpgpu::ACCESS_TYPE access_type) = 0;
    virtual void removeF32(std::size_t index) = 0;
    virtual void removeU32(std::size_t index) = 0;
    virtual void clearF32() = 0;
    virtual void clearU32() = 0;
    virtual IntopImgTexView u32Handles() const = 0;
    virtual IntopImgTexView f32Handles() const = 0;
    virtual void mapBuffers() = 0;
    virtual void mapResources() = 0;
    virtual void release() = 0;
  };

  class DeviceStorage : public DeviceStorageInterface {
    // Texture buffers residing on GPU.
    std::vector<DeviceBuffer<>> f32_device_buffers;
    IdRemovalTracker<DeviceBuffer<>> f32_id_tracker;
    axstd::managed_vector<device::gpgpu::APITextureHandle> f32_api_texture_handles;  // Points to gpu texture objects from the built backend.

    std::vector<DeviceBuffer<>> u32_device_buffers;
    IdRemovalTracker<DeviceBuffer<>> u32_id_tracker;
    axstd::managed_vector<device::gpgpu::APITextureHandle> u32_api_texture_handles;

   public:
    ~DeviceStorage() override = default;
    std::size_t addF32(GLuint texture_id, GLenum format, device::gpgpu::ACCESS_TYPE access_type) override;
    std::size_t addU32(GLuint texture_id, GLenum format, device::gpgpu::ACCESS_TYPE access_type) override;

    void removeF32(std::size_t index) override;
    void removeU32(std::size_t index) override;

    void clearF32() override;
    void clearU32() override;

    IntopImgTexView u32Handles() const override { return u32_api_texture_handles; }
    IntopImgTexView f32Handles() const override { return f32_api_texture_handles; }

    void mapBuffers() override;
    void mapResources() override;
    void release() override;
  };

  /* As with the shape module , we need a reference to host stored resources for many reasons.
   * 1) Mainly, we need to keep in memory the width and height of the texture as this is not retrievable only from interop functions.
   * 2) We still need the ability to render fully on CPU even though the application has been built with gpgpu support.
   * 3) Source of truth for ID retrieval and computations comes from cpu_storage, ie, each time we add a texture, its final ID will be computed
   * from cpu_storage internal structures.
   */
  class DevicePolicy {  // We use a value of 3 arrays, for rgba_textures + frgba_textures + api_texture_handles.

   protected:
    std::unique_ptr<DeviceStorageInterface> gpu_storage;
    host_storage cpu_storage;
    CstU32ImgTexView U32Array() const;
    CstF32ImgTexView F32Array() const;

   public:
    DevicePolicy();
    DevicePolicy(std::unique_ptr<DeviceStorageInterface> dep);
    std::size_t add(const U32Texture &host_stored_texture, GLuint gl_stored_texture_id);
    std::size_t add(const F32Texture &host_stored_texture, GLuint gl_stored_texture_id);
    void removeF32(std::size_t f32_texture_id);
    void removeU32(std::size_t u32_texture_id);
  };

  template<class StoragePolicy = std::conditional_t<core::build::is_gpu_build, DevicePolicy, HostPolicy>>
  class TextureReferencesStorage : public StoragePolicy, public DeviceReferenceStorageInterface {

   public:
    CLASS_CM(TextureReferencesStorage)
    TextureReferencesStorage(std::unique_ptr<DeviceStorageInterface> abst_dev_stor);
    void clear() override;
    void mapBuffers() override;
    void mapResources() override;
    void release() override;

    /* Returns the added size of each type of container of textures. Don't use as a datastructure size for indexing.*/
    std::size_t size() const override;

    u32tex_shared_views_s getU32TexturesViews() const;
    f32tex_shared_views_s getF32TexturesViews() const;
  };

  using DefaultTextureReferencesStorage = TextureReferencesStorage<>;

}  // namespace nova::texturing

#endif  // TEXTURE_INTEROP_STORAGE_H

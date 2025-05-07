#ifndef TEXTURE_INTEROP_STORAGE_H
#define TEXTURE_INTEROP_STORAGE_H
#include "interfaces/DeviceReferenceStorageInterface.h"
#include "texture_datastructures.h"
#include <algorithm>
#include <cstddef>
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

  /*
   * Since removing an element from storages can change the indices of the other textures of that collection,
   * We simply register the indices of textures marked as unused so that the next add operation could register in that free index spot.
   * NUM_COLLECTIONS represents the number of different collections of textures of the structure extending this class.
   */
  template<unsigned NUM_COLLECTIONS>
  class IdRemovalTracker {
   protected:
    static constexpr std::size_t INVALID_INDEX = std::size_t(-1);
    std::vector<std::size_t> empty_ids[NUM_COLLECTIONS]{};

   public:
    template<unsigned COLLECTION_ID>
    void disableID(std::size_t element_id) {
      static_assert(COLLECTION_ID < NUM_COLLECTIONS);
      empty_ids[COLLECTION_ID].push_back(element_id);
    }

    static bool valid(std::size_t index) { return index != INVALID_INDEX; }

   protected:
    template<unsigned COLLECTION_ID>
    std::size_t getFirstFreeID() const {
      if (empty_ids[COLLECTION_ID].empty())
        return INVALID_INDEX;
      else
        return empty_ids[COLLECTION_ID][0];
    }

    template<unsigned COLLECTION_ID>
    void useID(std::size_t element_id) {
      auto iter = std::find_if(
          empty_ids[COLLECTION_ID].begin(), empty_ids[COLLECTION_ID].end(), [&element_id](const auto &test) { return element_id == test; });
      if (iter != empty_ids[COLLECTION_ID].end())
        empty_ids[COLLECTION_ID].erase(iter);
    }
  };

  struct device_storage {
    // Texture buffers residing on GPU.
    std::vector<DeviceBuffer<>> device_buffers{};
    axstd::managed_vector<device::gpgpu::APITextureHandle> api_texture_handles{};  // Points to gpu texture objects from the built backend.
  };

  struct host_storage {
    std::vector<U32Texture> rgba_textures{};
    std::vector<F32Texture> frgba_textures{};
  };

  /* Storage description available CPU only build. */
  class HostPolicy : protected IdRemovalTracker<2> {  // We use a value of 2 arrays, for rgba_textures + frgba_textures.
    enum IDTYPE : unsigned { U32 = 0, F32 = 1 };

   protected:
    host_storage cpu_storage;

   public:
    std::size_t add(const U32Texture &host_stored_texture);
    std::size_t add(const F32Texture &host_stored_texture);
    void removeF32(std::size_t f32_texture_id);
    void removeU32(std::size_t u32_texture_id);
    CstU32ImgTexView U32Array() const;
    CstF32ImgTexView F32Array() const;
  };

  /* As with the shape module , we need a reference to host stored resources for many reasons.
   * 1) Mainly, we need to keep in memory the width and height of the texture as this is not retrievable only from interop functions.
   * 2) We still need the ability to render fully on CPU even though the application has been built with gpgpu support.
   * 3) Source of truth for ID retrieval and computations comes from cpu_storage, ie, each time we add a texture, its final ID will be computed
   * from cpu_storage internal structures.
   */
  class DevicePolicy : protected IdRemovalTracker<3> {  // We use a value of 3 arrays, for rgba_textures + frgba_textures + api_texture_handles.
    enum IDTYPE : unsigned { U32 = 0, F32 = 1, DB = 2 };

   protected:
    device_storage gpu_storage;
    host_storage cpu_storage;

   public:
    std::size_t add(const U32Texture &host_stored_texture, GLuint gl_stored_texture_id);
    std::size_t add(const F32Texture &host_stored_texture, GLuint gl_stored_texture_id);
    void removeF32(std::size_t f32_texture_id);
    void removeU32(std::size_t u32_texture_id);
    CstU32ImgTexView U32Array() const;
    CstF32ImgTexView F32Array() const;
    IntopImgTexView deviceTextures() const;
  };

  template<class StoragePolicy = std::conditional_t<core::build::is_gpu_build, DevicePolicy, HostPolicy>>
  class TextureReferencesStorage : public StoragePolicy, public DeviceReferenceStorageInterface {

   public:
    CLASS_CM(TextureReferencesStorage)

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

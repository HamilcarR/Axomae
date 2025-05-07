#include "texture_interop_storage.h"
#include "texture_datastructures.h"
#include <internal/device/rendering/opengl/gl_headers.h>
namespace nova::texturing {

  struct valid_create_s {
    std::size_t new_size;
    bool valid;
  };
  template<class T>
  valid_create_s construct_device_buffer(T &gpu_storage, GLuint stored_texture_id, GLenum format, device::gpgpu::ACCESS_TYPE access) {
    if constexpr (core::build::is_gpu_build) {
      gpu_storage.device_buffers.emplace_back(stored_texture_id, format, access);
      valid_create_s create_buffer{};
      create_buffer.new_size = gpu_storage.device_buffers.size() - 1;
      create_buffer.valid = true;
      return create_buffer;
    } else {
      valid_create_s create_buffer{};
      create_buffer.valid = false;
      return create_buffer;
    }
  }
  template<class T>
  void map_buffer(T &buffer) {
    if constexpr (core::build::is_gpu_build) {
      buffer.mapBuffer();
    }
  }

  struct valid_handle_s {
    device::gpgpu::APITextureHandle handle;
    bool valid;
  };

  template<class T>
  valid_handle_s image_id(const T &buffer) {
    if constexpr (core::build::is_gpu_build) {
      valid_handle_s handle{};
      handle.handle = buffer.getImageID();
      handle.valid = true;
      return handle;
    } else {
      valid_handle_s handle{};
      handle.valid = false;
      return handle;
    }
  }

  template<class T>
  void map_resource(T &buffer) {
    if constexpr (core::build::is_gpu_build) {
      buffer.mapResource();
    }
  }

  template<class T>
  void unmap_resource(T &buffer) {
    if constexpr (core::build::is_gpu_build) {
      buffer.unmapResource();
    }
  }

  template<>
  std::size_t TextureReferencesStorage<>::size() const {
    return cpu_storage.rgba_textures.size() + cpu_storage.frgba_textures.size();
  }

  /*
   * 1) Check if an ID is free.
   * 2) If returned ID is valid, use it as index of storage internal collections, and assign texture to that index.
   * 3) If not, add the texture at the end of storage.
   */

  std::size_t HostPolicy::add(const U32Texture &host_stored_texture) {

    std::size_t ffid = getFirstFreeID<U32>();
    if (ffid != INVALID_INDEX) {
      useID<U32>(ffid);
      cpu_storage.rgba_textures[ffid] = host_stored_texture;
      return ffid;
    }

    cpu_storage.rgba_textures.push_back(host_stored_texture);
    return cpu_storage.rgba_textures.size() - 1;
  }

  std::size_t HostPolicy::add(const F32Texture &host_stored_texture) {
    std::size_t ffid = getFirstFreeID<F32>();
    if (ffid != INVALID_INDEX) {
      useID<F32>(ffid);
      cpu_storage.frgba_textures[ffid] = host_stored_texture;
      return ffid;
    }
    cpu_storage.frgba_textures.push_back(host_stored_texture);
    return cpu_storage.frgba_textures.size() - 1;
  }

  void HostPolicy::removeF32(std::size_t index) { disableID<F32>(index); }

  void HostPolicy::removeU32(std::size_t index) { disableID<U32>(index); }

  std::size_t DevicePolicy::add(const U32Texture &host_stored_texture, GLuint gl_stored_texture_id) {
    cpu_storage.rgba_textures.push_back(host_stored_texture);
    valid_create_s buffer_create = construct_device_buffer(gpu_storage, gl_stored_texture_id, GL_TEXTURE_2D, device::gpgpu::READ_ONLY);
    AX_ASSERT_TRUE(buffer_create.valid);
    return cpu_storage.rgba_textures.size() - 1;
  }

  std::size_t DevicePolicy::add(const F32Texture &host_stored_texture, GLuint gl_stored_texture_id) {
    cpu_storage.frgba_textures.push_back(host_stored_texture);
    valid_create_s buffer_create = construct_device_buffer(gpu_storage, gl_stored_texture_id, GL_TEXTURE_2D, device::gpgpu::READ_ONLY);
    AX_ASSERT_TRUE(buffer_create.valid);
    return cpu_storage.frgba_textures.size() - 1;
  }

  void DevicePolicy::removeF32(std::size_t index) { disableID<F32>(index); }

  void DevicePolicy::removeU32(std::size_t index) { disableID<U32>(index); }

  inline U32Texture make_rgba_texture(uint32_t *raw_data, int width, int height, int chan) {
    U32Texture texture;
    texture.width = width;
    texture.height = height;
    texture.channels = chan;
    texture.raw_data = axstd::span<const uint32_t>(raw_data, width * height * chan);
    return texture;
  }

  CstU32ImgTexView HostPolicy::U32Array() const { return cpu_storage.rgba_textures; }
  CstF32ImgTexView HostPolicy::F32Array() const { return cpu_storage.frgba_textures; }
  IntopImgTexView DevicePolicy::deviceTextures() const { return gpu_storage.api_texture_handles; }
  CstF32ImgTexView DevicePolicy::F32Array() const { return cpu_storage.frgba_textures; }
  CstU32ImgTexView DevicePolicy::U32Array() const { return cpu_storage.rgba_textures; }

  /***********************************************************************************************************************************/
  /* Device Policy. */

  template<>
  u32tex_shared_views_s TextureReferencesStorage<DevicePolicy>::getU32TexturesViews() const {
    u32tex_shared_views_s views;
    views.interop_handles = gpu_storage.api_texture_handles;
    views.u32_host = cpu_storage.rgba_textures;
    return views;
  }

  template<>
  f32tex_shared_views_s TextureReferencesStorage<DevicePolicy>::getF32TexturesViews() const {
    f32tex_shared_views_s views;
    views.interop_handles = gpu_storage.api_texture_handles;
    views.f32_host = cpu_storage.frgba_textures;
    return views;
  }

  template<>
  void TextureReferencesStorage<DevicePolicy>::clear() {
    gpu_storage.device_buffers.clear();
    cpu_storage.frgba_textures.clear();
    cpu_storage.rgba_textures.clear();
  }

  template<>
  void TextureReferencesStorage<DevicePolicy>::mapBuffers() {
    gpu_storage.api_texture_handles.clear();
    for (auto &dbuffer_iter : gpu_storage.device_buffers) {
      map_buffer(dbuffer_iter);
      valid_handle_s handle = image_id(dbuffer_iter);
      AX_ASSERT_TRUE(handle.valid);
      gpu_storage.api_texture_handles.push_back(handle.handle);
    }
  }

  template<>
  void TextureReferencesStorage<DevicePolicy>::mapResources() {
    for (auto &dbuffer_iter : gpu_storage.device_buffers)
      map_resource(dbuffer_iter);
  }

  template<>
  void TextureReferencesStorage<DevicePolicy>::release() {
    for (auto &dbuffer_iter : gpu_storage.device_buffers)
      unmap_resource(dbuffer_iter);
  }

  /***********************************************************************************************************************************/
  /* Host Policy. */

  template<>
  u32tex_shared_views_s TextureReferencesStorage<HostPolicy>::getU32TexturesViews() const {
    u32tex_shared_views_s views;
    views.u32_host = cpu_storage.rgba_textures;
    return views;
  }

  template<>
  f32tex_shared_views_s TextureReferencesStorage<HostPolicy>::getF32TexturesViews() const {
    f32tex_shared_views_s views;
    views.f32_host = cpu_storage.frgba_textures;
    return views;
  }

  template<>
  void TextureReferencesStorage<HostPolicy>::clear() {
    cpu_storage.rgba_textures.clear();
    cpu_storage.frgba_textures.clear();
  }

  template<>
  void TextureReferencesStorage<HostPolicy>::mapBuffers() {
    EMPTY_FUNCBODY
  }

  template<>
  void TextureReferencesStorage<HostPolicy>::mapResources() {
    EMPTY_FUNCBODY
  }

  template<>
  void TextureReferencesStorage<HostPolicy>::release() {
    EMPTY_FUNCBODY
  }

}  // namespace nova::texturing

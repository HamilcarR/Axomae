#include "texture_interop_storage.h"
#include <internal/device/rendering/opengl/gl_headers.h>
namespace nova::texturing {

  struct valid_create_s {
    std::size_t new_size;
    bool valid;
  };
  template<class T>
  valid_create_s construct_device_buffer(T &gpu_storage, GLuint stored_texture_id, GLenum format, device::gpgpu::ACCESS_TYPE access) {
    if constexpr (core::build::is_gpu_build) {
      gpu_storage.rgba_device_buffers.emplace_back(stored_texture_id, format, access);
      valid_create_s create_buffer{};
      create_buffer.new_size = gpu_storage.rgba_device_buffers.size() - 1;
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
    return container_capacity;
  }

  std::size_t HostPolicy::add(const U32Texture &host_stored_texture) {
    cpu_storage.rgba_textures.push_back(host_stored_texture);
    return cpu_storage.rgba_textures.size() - 1;
  }

  std::size_t DevicePolicy::add(const U32Texture &host_stored_texture, GLuint gl_stored_texture_id) {
    cpu_storage.rgba_textures.push_back(host_stored_texture);
    valid_create_s buffer_create = construct_device_buffer(gpu_storage, gl_stored_texture_id, GL_TEXTURE_2D, device::gpgpu::READ_ONLY);
    AX_ASSERT_TRUE(buffer_create.valid);
    return buffer_create.new_size;
  }

  inline U32Texture make_rgba_texture(uint32_t *raw_data, int width, int height, int chan) {
    U32Texture texture;
    texture.width = width;
    texture.height = height;
    texture.channels = chan;
    texture.raw_data = axstd::span<const uint32_t>(raw_data, width * height * chan);
    return texture;
  }

  CstU32ImgTexView HostPolicy::getHost() const { return cpu_storage.rgba_textures; }
  IntopImgTexView DevicePolicy::getDevice() const { return gpu_storage.rgba_textures; }
  CstU32ImgTexView DevicePolicy::getHost() const { return cpu_storage.rgba_textures; }

  /***********************************************************************************************************************************/
  /* Device Policy. */

  template<>
  u32tex_shared_views_s TextureReferencesStorage<DevicePolicy>::getU32TexturesViews() const {
    u32tex_shared_views_s views;
    views.interop_handles = gpu_storage.rgba_textures;
    views.u32_host = cpu_storage.rgba_textures;
    return views;
  }

  template<>
  void TextureReferencesStorage<DevicePolicy>::allocate(std::size_t num_textures) {
    gpu_storage.rgba_device_buffers.reserve(num_textures);
    cpu_storage.rgba_textures.reserve(num_textures);
    container_capacity = num_textures;
  }

  template<>
  void TextureReferencesStorage<DevicePolicy>::clear() {
    gpu_storage.rgba_device_buffers.clear();
    cpu_storage.rgba_textures.clear();
    container_capacity = 0;
  }

  template<>
  void TextureReferencesStorage<DevicePolicy>::mapBuffers() {
    gpu_storage.rgba_textures.clear();
    for (auto &dbuffer_iter : gpu_storage.rgba_device_buffers) {
      map_buffer(dbuffer_iter);
      valid_handle_s handle = image_id(dbuffer_iter);
      AX_ASSERT_TRUE(handle.valid);
      gpu_storage.rgba_textures.push_back(handle.handle);
    }
  }

  template<>
  void TextureReferencesStorage<DevicePolicy>::mapResources() {
    for (auto &dbuffer_iter : gpu_storage.rgba_device_buffers)
      map_resource(dbuffer_iter);
  }

  template<>
  void TextureReferencesStorage<DevicePolicy>::release() {
    for (auto &dbuffer_iter : gpu_storage.rgba_device_buffers)
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
  void TextureReferencesStorage<HostPolicy>::allocate(std::size_t num_textures) {
    cpu_storage.rgba_textures.reserve(num_textures);
    container_capacity = num_textures;
  }

  template<>
  void TextureReferencesStorage<HostPolicy>::clear() {
    cpu_storage.rgba_textures.clear();
    container_capacity = 0;
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
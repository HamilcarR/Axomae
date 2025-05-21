#include "texture_interop_storage.h"
#include "texture_datastructures.h"
#include <internal/device/gpgpu/device_resource_descriptors.h>
#include <internal/device/rendering/opengl/gl_headers.h>
#include <internal/macro/project_macros.h>
#include <memory>
namespace nova::texturing {

  /*
   * 1) Check if an ID is free.
   * 2) If returned ID is valid, use it as index of storage internal collections, and assign texture to that index.
   * 3) If not, add the texture at the end of storage.
   */
  template<class Container, class E>
  std::size_t indexed_append(Container &collection, IdRemovalTracker<typename Container::value_type> &id_tracker, const E &element) {
    using T = typename Container::value_type;
    using IdRemovalTracker_t = IdRemovalTracker<T>;

    std::size_t index = id_tracker.getFirstFreeID();

    if (!IdRemovalTracker_t::valid(index)) {
      collection.push_back(element);
      return collection.size() - 1;
    }

    id_tracker.useID(index);
    collection[index] = element;

    return index;
  }

  template<class Container, class... Args>
  std::size_t indexed_emplace(Container &collection, IdRemovalTracker<typename Container::value_type> &id_tracker, Args &&...args) {
    using T = typename Container::value_type;
    using IdRemovalTracker_t = IdRemovalTracker<T>;

    std::size_t index = id_tracker.getFirstFreeID();

    if (!IdRemovalTracker_t::valid(index)) {
      collection.emplace_back(std::forward<Args>(args)...);
      return collection.size() - 1;
    }

    id_tracker.useID(index);
    collection.emplace(collection.begin() + index, std::forward<Args>(args)...);
    return index;
  }

  /***********************************************************************************************************************************/
  // TODO : use indexed_append
  std::size_t HostPolicy::add(const U32Texture &host_stored_texture) {
    return indexed_append(cpu_storage.rgba_textures, cpu_storage.u32_id_tracker, host_stored_texture);
  }

  std::size_t HostPolicy::add(const F32Texture &host_stored_texture) {
    return indexed_append(cpu_storage.frgba_textures, cpu_storage.f32_id_tracker, host_stored_texture);
  }

  void HostPolicy::removeF32(std::size_t index) { cpu_storage.f32_id_tracker.disableID(index); }

  void HostPolicy::removeU32(std::size_t index) { cpu_storage.u32_id_tracker.disableID(index); }

  /***********************************************************************************************************************************/

  std::size_t DeviceStorage::addF32(GLuint texture_id, GLenum format, device::gpgpu::ACCESS_TYPE access) {
    return indexed_emplace(f32_device_buffers, f32_id_tracker, texture_id, format, access);
  }

  std::size_t DeviceStorage::addU32(GLuint texture_id, GLenum format, device::gpgpu::ACCESS_TYPE access) {
    return indexed_emplace(u32_device_buffers, u32_id_tracker, texture_id, format, access);
  }

  void DeviceStorage::removeF32(std::size_t index) { f32_id_tracker.disableID(index); }

  void DeviceStorage::removeU32(std::size_t index) { u32_id_tracker.disableID(index); }

  void DeviceStorage::clearF32() {
    f32_device_buffers.clear();
    f32_id_tracker.reset();
    f32_api_texture_handles.clear();
  }

  void DeviceStorage::clearU32() {
    u32_device_buffers.clear();
    u32_id_tracker.reset();
    u32_api_texture_handles.clear();
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
  void map_buffer(T &buffer) {
    if constexpr (core::build::is_gpu_build) {
      buffer.mapBuffer();
    }
  }

  void DeviceStorage::mapBuffers() {
    f32_api_texture_handles.clear();
    for (auto &dbuffer_iter : f32_device_buffers) {
      map_buffer(dbuffer_iter);
      valid_handle_s handle = image_id(dbuffer_iter);
      AX_ASSERT_TRUE(handle.valid);
      f32_api_texture_handles.push_back(handle.handle);
    }
    u32_api_texture_handles.clear();
    for (auto &dbuffer_iter : u32_device_buffers) {
      map_buffer(dbuffer_iter);
      valid_handle_s handle = image_id(dbuffer_iter);
      AX_ASSERT_TRUE(handle.valid);
      u32_api_texture_handles.push_back(handle.handle);
    }
  }

  template<class T>
  void map_resource(T &buffer) {
    if constexpr (core::build::is_gpu_build) {
      buffer.mapResource();
    }
  }

  void DeviceStorage::mapResources() {
    for (auto &dbuffer_iter : f32_device_buffers)
      map_resource(dbuffer_iter);
    for (auto &dbuffer_iter : u32_device_buffers)
      map_resource(dbuffer_iter);
  }

  template<class T>
  void unmap_resource(T &buffer) {
    if constexpr (core::build::is_gpu_build) {
      buffer.unmapResource();
    }
  }

  void DeviceStorage::release() {
    for (auto &dbuffer_iter : u32_device_buffers)
      unmap_resource(dbuffer_iter);
    for (auto &dbuffer_iter : f32_device_buffers)
      unmap_resource(dbuffer_iter);
  }

  /***********************************************************************************************************************************/
  DevicePolicy::DevicePolicy() : gpu_storage(std::make_unique<DeviceStorage>()) {}

  DevicePolicy::DevicePolicy(std::unique_ptr<DeviceStorageInterface> dep) : gpu_storage(std::move(dep)) {}

  std::size_t DevicePolicy::add(const U32Texture &host_stored_texture, GLuint gl_stored_texture_id) {
    std::size_t index = indexed_append(cpu_storage.rgba_textures, cpu_storage.u32_id_tracker, host_stored_texture);
    gpu_storage->addU32(gl_stored_texture_id, GL_TEXTURE_2D, device::gpgpu::READ_ONLY);
    return index;
  }

  std::size_t DevicePolicy::add(const F32Texture &host_stored_texture, GLuint gl_stored_texture_id) {
    std::size_t index = indexed_append(cpu_storage.frgba_textures, cpu_storage.f32_id_tracker, host_stored_texture);
    gpu_storage->addF32(gl_stored_texture_id, GL_TEXTURE_2D, device::gpgpu::READ_ONLY);
    return index;
  }

  void DevicePolicy::removeF32(std::size_t index) {
    cpu_storage.f32_id_tracker.disableID(index);
    gpu_storage->removeF32(index);
  }

  void DevicePolicy::removeU32(std::size_t index) {
    cpu_storage.u32_id_tracker.disableID(index);
    gpu_storage->removeU32(index);
  }

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
  CstF32ImgTexView DevicePolicy::F32Array() const { return cpu_storage.frgba_textures; }
  CstU32ImgTexView DevicePolicy::U32Array() const { return cpu_storage.rgba_textures; }

  /***********************************************************************************************************************************/
  /* Device Policy. */

  template<>
  TextureReferencesStorage<DevicePolicy>::TextureReferencesStorage(std::unique_ptr<DeviceStorageInterface> abst_dev_stor)
      : DevicePolicy(std::move(abst_dev_stor)) {}

  template<>
  u32tex_shared_views_s TextureReferencesStorage<DevicePolicy>::getU32TexturesViews() const {
    u32tex_shared_views_s views;
    views.interop_handles = gpu_storage->u32Handles();
    views.u32_managed = cpu_storage.rgba_textures;
    return views;
  }

  template<>
  f32tex_shared_views_s TextureReferencesStorage<DevicePolicy>::getF32TexturesViews() const {
    f32tex_shared_views_s views;
    views.interop_handles = gpu_storage->f32Handles();
    views.f32_managed = cpu_storage.frgba_textures;
    return views;
  }

  template<>
  void TextureReferencesStorage<DevicePolicy>::clear() {
    gpu_storage->clearU32();
    gpu_storage->clearF32();
    cpu_storage.frgba_textures.clear();
    cpu_storage.rgba_textures.clear();
  }

  template<>
  void TextureReferencesStorage<DevicePolicy>::mapBuffers() {
    gpu_storage->mapBuffers();
  }

  template<>
  void TextureReferencesStorage<DevicePolicy>::mapResources() {
    gpu_storage->mapResources();
  }

  template<>
  void TextureReferencesStorage<DevicePolicy>::release() {
    gpu_storage->release();
  }

  template<>
  std::size_t TextureReferencesStorage<DevicePolicy>::size() const {
    return cpu_storage.rgba_textures.size() + cpu_storage.frgba_textures.size();
  }

  /***********************************************************************************************************************************/
  /* Host Policy. */

  template<>
  TextureReferencesStorage<HostPolicy>::TextureReferencesStorage(std::unique_ptr<DeviceStorageInterface> abst_dev_stor) {}

  template<>
  u32tex_shared_views_s TextureReferencesStorage<HostPolicy>::getU32TexturesViews() const {
    u32tex_shared_views_s views;
    views.u32_managed = cpu_storage.rgba_textures;
    return views;
  }

  template<>
  f32tex_shared_views_s TextureReferencesStorage<HostPolicy>::getF32TexturesViews() const {
    f32tex_shared_views_s views;
    views.f32_managed = cpu_storage.frgba_textures;
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

  template<>
  std::size_t TextureReferencesStorage<HostPolicy>::size() const {
    return cpu_storage.rgba_textures.size() + cpu_storage.frgba_textures.size();
  }
}  // namespace nova::texturing

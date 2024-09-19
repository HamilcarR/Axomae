#include "nova_texturing.h"
#include "NovaTextureInterface.h"
namespace nova::texturing {

  void TextureStorage::allocConstant(std::size_t total_cste_tex) {
    constant_textures.clear();
    constant_textures.reserve(total_cste_tex);
  }

  void TextureStorage::allocImage(std::size_t total_img_tex) {
    image_textures.clear();
    image_textures.reserve(total_img_tex);
  }

  void TextureStorage::allocEnvmap(std::size_t total_env_tex) {
    envmap_textures.clear();
    envmap_textures.reserve(total_env_tex);
  }

  EnvMapCollection TextureStorage::envmaps() const { return envmap_textures; }
  CstTexCollection TextureStorage::constants() const { return constant_textures; }
  ImgTexCollection TextureStorage::images() const { return image_textures; }
  IntfTexCollection TextureStorage::pointers() const { return textures; }

  template<class T>
  static void erase_shared_elements(const axstd::managed_vector<T> &type_collection, axstd::managed_vector<NovaTextureInterface> &interface_vector) {
    for (const auto &elem : type_collection) {
      auto iterator = std::find_if(
          interface_vector.begin(), interface_vector.end(), [&elem](const NovaTextureInterface &interface) { return interface.get() == &elem; });
      if (iterator != interface_vector.end())
        interface_vector.erase(iterator);
    }
  }

  void TextureStorage::clearEnvmap() {
    erase_shared_elements(envmap_textures, textures);
    envmap_textures.clear();
  }

  void TextureStorage::clearConstant() {
    erase_shared_elements(constant_textures, textures);
    constant_textures.clear();
  }

  void TextureStorage::clearImage() {
    erase_shared_elements(image_textures, textures);
    image_textures.clear();
  }

  void TextureStorage::clear() {
    textures.clear();
    constant_textures.clear();
    image_textures.clear();
    envmap_textures.clear();
  }

  void TextureResourcesHolder::allocateMeshTextures(const texture_init_record_s &init_data) {
    texture_storage.allocConstant(init_data.total_constant_textures);
    texture_storage.allocImage(init_data.total_image_textures);
  }

  void TextureResourcesHolder::allocateEnvironmentMaps(std::size_t num_envmaps) {
    EnvMapCollection envmaps = texture_storage.envmaps();
    for (const EnvmapTexture &elem : envmaps) {
      std::size_t texture_index = elem.getTextureIndex();
      texture_raw_data_storage.removeF32(texture_index);
    }
    texture_storage.clearEnvmap();
    texture_storage.allocEnvmap(num_envmaps);
  }

  NovaTextureInterface TextureResourcesHolder::getEnvmap(unsigned index) const {
    EnvMapCollection envmaps = texture_storage.envmaps();
    AX_ASSERT_LT(index, envmaps.size());
    return &envmaps[index];
  }

  NovaTextureInterface TextureResourcesHolder::getCurrentEnvmap() const { return getEnvmap(current_environmentmap_index); }

  void TextureResourcesHolder::lockResources() { texture_raw_data_storage.mapResources(); }

  void TextureResourcesHolder::releaseResources() { texture_raw_data_storage.release(); }

  void TextureResourcesHolder::mapBuffers() { texture_raw_data_storage.mapBuffers(); }

  TextureBundleViews TextureResourcesHolder::getTextureBundleViews() const {
    return TextureBundleViews(texture_raw_data_storage.getU32TexturesViews(), texture_raw_data_storage.getF32TexturesViews());
  }

}  // namespace nova::texturing

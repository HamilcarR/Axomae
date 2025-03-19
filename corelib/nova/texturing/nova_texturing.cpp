#include "nova_texturing.h"
namespace nova::texturing {

  void TextureResourcesHolder::init(const texture_init_record_s &init_data) {
    texture_storage.allocConstant(init_data.constant_texture_size);
    texture_storage.allocImage(init_data.image_texture_size);
    texture_raw_data_storage.allocate(init_data.image_texture_size);
  }
  void TextureResourcesHolder::lockResources() { texture_raw_data_storage.mapResources(); }

  void TextureResourcesHolder::releaseResources() { texture_raw_data_storage.release(); }

  void TextureResourcesHolder::mapBuffers() { texture_raw_data_storage.mapBuffers(); }

  TextureBundleViews TextureResourcesHolder::getTextureBundleViews() const { return {texture_raw_data_storage.getU32TexturesViews()}; }

}  // namespace nova::texturing
#include "Drawable.h"
#include "MaterialInterface.h"
#include "Mesh.h"
#include "bake.h"
#include "extract_scene_internal.h"
#include <internal/common/exception/GenericException.h>
#include <internal/macro/project_macros.h>
#include <nova/api_engine.h>

namespace nova_baker_utils {

  nova::TexturePtr extract_texture(const TextureGroup &tgroup, GenericTexture::TYPE type) {
    const GenericTexture *gltexture = tgroup.getTexturePointer(type);
    if (!gltexture) {
      LOG("Texture lookup in Nova scene initialization has returned null for texture type: " + std::string(type2str(type)), LogLevel::INFO);
      return {};
    }
    nova::TexturePtr image_texture = nova::create_texture();
    image_texture->setTextureBuffer(gltexture->getData());
    image_texture->setWidth(gltexture->getWidth());
    image_texture->setHeight(gltexture->getHeight());
    image_texture->setChannels(4);
    image_texture->setInteropID(gltexture->getSamplerID());
    return image_texture;
  }

  static void set_material_textures(const MaterialInterface *client_material, nova::Material &material) {
    const TextureGroup &texture_group = client_material->getTextureGroup();
    material.registerAlbedo(extract_texture(texture_group, GenericTexture::DIFFUSE));
    material.registerMetallic(extract_texture(texture_group, GenericTexture::METALLIC));
    material.registerNormal(extract_texture(texture_group, GenericTexture::NORMAL));
    material.registerRoughness(extract_texture(texture_group, GenericTexture::ROUGHNESS));
    material.registerAmbientOcclusion(extract_texture(texture_group, GenericTexture::AMBIANTOCCLUSION));
    material.registerEmissive(extract_texture(texture_group, GenericTexture::EMISSIVE));
    material.registerOpacity(extract_texture(texture_group, GenericTexture::OPACITY));
    material.registerSpecular(extract_texture(texture_group, GenericTexture::SPECULAR));
  }

  static void set_material_properties(const MaterialInterface *client_material, nova::Material &material) {
    Vec2f ior = client_material->getRefractiveIndex();
    float eta[3] = {ior.x, ior.x, ior.x};
    material.setRefractiveIndex(eta);
  }

  void setup_material(const drawable_original_transform &drawable, nova::Material &material) {
    const MaterialInterface *mesh_material = drawable.mesh->getMeshPointer()->getMaterial();
    set_material_textures(mesh_material, material);
    set_material_properties(mesh_material, material);
  }
}  // namespace nova_baker_utils

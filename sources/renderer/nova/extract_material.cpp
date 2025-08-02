#include "Drawable.h"
#include "MaterialInterface.h"
#include "Mesh.h"
#include "bake.h"
#include "extract_scene_internal.h"
#include <internal/common/exception/GenericException.h>
#include <internal/macro/project_macros.h>
#include <nova/NovaAPI.h>

namespace exception {
  class InvalidTexTypeConversionException : public CatastrophicFailureException {
   public:
    InvalidTexTypeConversionException() : CatastrophicFailureException() { saveErrorString("Invalid tag pointer conversion."); }
  };
}  // namespace exception

namespace nova_baker_utils {

  nova::texturing::ImageTexture<uint32_t> *extract_texture(const TextureGroup &tgroup,
                                                           nova::NovaResourceManager &manager,
                                                           GenericTexture::TYPE type) {
    const GenericTexture *gltexture = tgroup.getTexturePointer(type);
    if (!gltexture) {
      LOG("Texture lookup in Nova scene initialization has returned null for texture type: " + std::string(type2str(type)), LogLevel::INFO);
      return nullptr;
    }
    const uint32_t *buffer_ptr = gltexture->getData();
    int w = (int)gltexture->getWidth();
    int h = (int)gltexture->getHeight();
    nova::texturing::TextureResourcesHolder &texture_manager = manager.getTexturesData();
    std::size_t texture_index = texture_manager.addTexture(buffer_ptr, w, h, 4, false, false, gltexture->getSamplerID());
    auto ret = texture_manager.addNovaTexture<nova::texturing::ImageTexture<uint32_t>>(texture_index);
    auto *image_tex = ret.get<nova::texturing::ImageTexture<uint32_t>>();
    if (!image_tex)
      throw exception::InvalidTexTypeConversionException();
    return image_tex;
  }

  static nova::material::texture_pack extract_texture_pack(const MaterialInterface *material, nova::NovaResourceManager &manager) {
    const TextureGroup &texture_group = material->getTextureGroup();
    nova::material::texture_pack tpack{};
    tpack.albedo = extract_texture(texture_group, manager, GenericTexture::DIFFUSE);
    tpack.metallic = extract_texture(texture_group, manager, GenericTexture::METALLIC);
    tpack.normalmap = extract_texture(texture_group, manager, GenericTexture::NORMAL);
    tpack.roughness = extract_texture(texture_group, manager, GenericTexture::ROUGHNESS);
    tpack.ao = extract_texture(texture_group, manager, GenericTexture::AMBIANTOCCLUSION);
    tpack.emissive = extract_texture(texture_group, manager, GenericTexture::EMISSIVE);
    tpack.opacity = extract_texture(texture_group, manager, GenericTexture::OPACITY);
    tpack.specular = extract_texture(texture_group, manager, GenericTexture::SPECULAR);
    return tpack;
  }

  // For now it assigns materials randomly, I don't want to waste time with a proper material translation system since I'm gonna scrape it for a more
  // uniform pipeline with PBR
  static nova::material::NovaMaterialInterface assign_random_material(nova::material::texture_pack &tpack, nova::NovaResourceManager &manager) {
    math::random::CPUPseudoRandomGenerator rand_gen;
    nova::material::NovaMaterialInterface mat_ptr{};
    int r = rand_gen.nrandi(0, 2);
    switch (r) {
      case 0:
        mat_ptr = manager.getMaterialData().addMaterial<nova::material::NovaConductorMaterial>(tpack, rand_gen.nrandf(0.001, 0.5));
        break;
      case 1:
        mat_ptr = manager.getMaterialData().addMaterial<nova::material::NovaDielectricMaterial>(tpack, rand_gen.nrandf(1.5, 2.4));
        break;
      case 2:
        mat_ptr = manager.getMaterialData().addMaterial<nova::material::NovaDiffuseMaterial>(tpack);
        break;
      default:
        mat_ptr = manager.getMaterialData().addMaterial<nova::material::NovaConductorMaterial>(tpack, 0.004);
        break;
    }
    return mat_ptr;
  }

  nova::material::NovaMaterialInterface setup_material_data(const Drawable &drawable, nova::NovaResourceManager &manager) {
    const Mesh *mesh = drawable.getMeshPointer();
    const MaterialInterface *material = mesh->getMaterial();
    if (!material) {
      return nullptr;
    }
    nova::material::texture_pack tpack = extract_texture_pack(mesh->getMaterial(), manager);
    nova::material::NovaMaterialInterface mat = assign_random_material(tpack, manager);
    return mat;
  }
}  // namespace nova_baker_utils

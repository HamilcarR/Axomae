#include "manager/NovaResourceManager.h"
#include "private_includes.h"
#include "texturing/NovaTextureInterface.h"
#include "texturing/nova_texturing.h"
namespace nova {
  NvMaterial::NvMaterial(const NvAbstractMaterial &other) { *this = *dynamic_cast<const NvMaterial *>(&other); }

  NvMaterial &NvMaterial::operator=(const NvAbstractMaterial &other) {
    if (this == &other)
      return *this;
    *this = *dynamic_cast<const NvMaterial *>(&other);
    return *this;
  }

  ERROR_STATE NvMaterial::registerAlbedo(const NvAbstractTexture &texture) {
    albedo = texture;
    return SUCCESS;
  }

  ERROR_STATE NvMaterial::registerNormal(const NvAbstractTexture &texture) {
    normal = texture;
    return SUCCESS;
  }

  ERROR_STATE NvMaterial::registerMetallic(const NvAbstractTexture &texture) {
    metallic = texture;
    return SUCCESS;
  }

  ERROR_STATE NvMaterial::registerEmissive(const NvAbstractTexture &texture) {
    emissive = texture;
    return SUCCESS;
  }

  ERROR_STATE NvMaterial::registerRoughness(const NvAbstractTexture &texture) {
    roughness = texture;
    return SUCCESS;
  }

  ERROR_STATE NvMaterial::registerOpacity(const NvAbstractTexture &texture) {
    opacity = texture;
    return SUCCESS;
  }

  ERROR_STATE NvMaterial::registerSpecular(const NvAbstractTexture &texture) {
    specular = texture;
    return SUCCESS;
  }

  ERROR_STATE NvMaterial::registerAmbientOcclusion(const NvAbstractTexture &texture) {
    ao = texture;
    return SUCCESS;
  }

  ERROR_STATE NvMaterial::setRefractCoeff(float eta) {
    refract_coeff = eta;
    return SUCCESS;
  }

  ERROR_STATE NvMaterial::setReflectFuzz(float fuzz) {
    reflect_fuzz = fuzz;
    return SUCCESS;
  }

  std::unique_ptr<NvAbstractMaterial> create_material() { return std::make_unique<NvMaterial>(); }

  static texturing::NovaTextureInterface build_img_texture(const NvTexture &texture, NovaResourceManager &manager) {
    texturing::TextureResourcesHolder &texture_manager = manager.getTexturesData();
    AX_ASSERT_EQ(texture.getDataType(), NvTexture::I_ARRAY);
    std::size_t texture_index = texture_manager.addTexture(
        texture.getData<uint32_t>(), texture.getWidth(), texture.getHeight(), texture.getChannels());
    return texture_manager.addNovaTexture<texturing::ImageTexture<uint32_t>>(texture_index);
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

  material::NovaMaterialInterface setup_material_data(const NvAbstractMesh &mesh, const NvMaterial &material, NovaResourceManager &manager) {
    material::texture_pack tpack;
    tpack.albedo = build_img_texture(material.getAlbedo(), manager);
    tpack.metallic = build_img_texture(material.getMetallic(), manager);
    tpack.roughness = build_img_texture(material.getRoughness(), manager);
    tpack.emissive = build_img_texture(material.getEmissive(), manager);
    tpack.opacity = build_img_texture(material.getOpacity(), manager);
    tpack.specular = build_img_texture(material.getSpecular(), manager);
    tpack.ao = build_img_texture(material.getAmbientOcclusion(), manager);
    return assign_random_material(tpack, manager);
  }

}  // namespace nova

#include "api_common.h"
#include "manager/NovaResourceManager.h"
#include "private_includes.h"
#include "texturing/NovaTextureInterface.h"
#include "texturing/nova_texturing.h"
namespace nova {

  class NvMaterial final : public Material {
    TexturePtr albedo;
    TexturePtr normal;
    TexturePtr metallic;
    TexturePtr emissive;
    TexturePtr roughness;
    TexturePtr opacity;
    TexturePtr specular;
    TexturePtr ao;
    float refract_coeff{1.0f};
    float reflect_fuzz{0.0f};

   public:
    ERROR_STATE registerAlbedo(TexturePtr texture) override {
      albedo = std::move(texture);
      return SUCCESS;
    }

    ERROR_STATE registerNormal(TexturePtr texture) override {
      normal = std::move(texture);
      return SUCCESS;
    }

    ERROR_STATE registerMetallic(TexturePtr texture) override {
      metallic = std::move(texture);
      return SUCCESS;
    }

    ERROR_STATE registerEmissive(TexturePtr texture) override {
      emissive = std::move(texture);
      return SUCCESS;
    }

    ERROR_STATE registerRoughness(TexturePtr texture) override {
      roughness = std::move(texture);
      return SUCCESS;
    }

    ERROR_STATE registerOpacity(TexturePtr texture) override {
      opacity = std::move(texture);
      return SUCCESS;
    }

    ERROR_STATE registerSpecular(TexturePtr texture) override {
      specular = std::move(texture);
      return SUCCESS;
    }

    ERROR_STATE registerAmbientOcclusion(TexturePtr texture) override {
      ao = std::move(texture);
      return SUCCESS;
    }

    ERROR_STATE setRefractCoeff(float eta) override {
      refract_coeff = eta;
      return SUCCESS;
    }

    ERROR_STATE setReflectFuzz(float fuzz) override {
      reflect_fuzz = fuzz;
      return SUCCESS;
    }

    Texture *getAlbedo() override { return albedo.get(); }

    const Texture *getAlbedo() const override { return albedo.get(); }

    Texture *getNormal() override { return normal.get(); }

    const Texture *getNormal() const override { return normal.get(); }

    Texture *getMetallic() override { return metallic.get(); }

    const Texture *getMetallic() const override { return metallic.get(); }

    Texture *getEmissive() override { return emissive.get(); }

    const Texture *getEmissive() const override { return emissive.get(); }

    Texture *getRoughness() override { return roughness.get(); }

    const Texture *getRoughness() const override { return roughness.get(); }

    Texture *getOpacity() override { return opacity.get(); }

    const Texture *getOpacity() const override { return opacity.get(); }

    Texture *getSpecular() override { return specular.get(); }

    const Texture *getSpecular() const override { return specular.get(); }

    Texture *getAmbientOcclusion() override { return ao.get(); }

    const Texture *getAmbientOcclusion() const override { return ao.get(); }

    float getRefractCoeff() const override { return refract_coeff; }

    float getReflectFuzz() const override { return reflect_fuzz; }
  };

  std::unique_ptr<Material> create_material() { return std::make_unique<NvMaterial>(); }

  static texturing::NovaTextureInterface build_img_texture(const Texture *texture, NovaResourceManager &manager) {
    if (!texture)
      return nullptr;
    texturing::TextureResourcesHolder &texture_manager = manager.getTexturesData();
    AX_ASSERT_EQ(texture->getFormat(), texture::UINT8X4);
    std::size_t texture_index = 0;
    switch (texture->getFormat()) {
      case texture::FLOATX4:
        texture_index = texture_manager.addTexture(static_cast<const float *>(texture->getTextureBuffer()),
                                                   texture->getWidth(),
                                                   texture->getHeight(),
                                                   texture->getChannels(),
                                                   texture->getInvertX(),
                                                   texture->getInvertY(),
                                                   texture->getInteropID());

        return texture_manager.addNovaTexture<texturing::ImageTexture<float>>(texture_index);
      case texture::UINT8X4:
      default:
        texture_index = texture_manager.addTexture(static_cast<const uint32_t *>(texture->getTextureBuffer()),
                                                   texture->getWidth(),
                                                   texture->getHeight(),
                                                   texture->getChannels(),
                                                   texture->getInvertX(),
                                                   texture->getInvertY(),
                                                   texture->getInteropID());

        return texture_manager.addNovaTexture<texturing::ImageTexture<uint32_t>>(texture_index);
    }
  }

  // For now it assigns materials randomly, I don't want to waste time with a proper material translation system since I'm gonna scrape it for a more
  // uniform pipeline with PBR
  static nova::material::NovaMaterialInterface assign_random_material(nova::material::texture_pack &tpack, nova::NovaResourceManager &manager) {
    math::random::CPUPseudoRandomGenerator rand_gen;
    nova::material::NovaMaterialInterface mat_ptr{};
    int r = 0;  // rand_gen.nrandi(0, 2);
    switch (r) {
      case 0:
        mat_ptr = manager.getMaterialData().addMaterial<nova::material::NovaConductorMaterial>(tpack, rand_gen.nrandf(0.001, 0.001));
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

  material::NovaMaterialInterface setup_material_data(const AbstractMesh &mesh, const Material &material, NovaResourceManager &manager) {
    material::texture_pack tpack;
    tpack.albedo = build_img_texture(material.getAlbedo(), manager);
    tpack.normalmap = build_img_texture(material.getNormal(), manager);
    tpack.metallic = build_img_texture(material.getMetallic(), manager);
    tpack.roughness = build_img_texture(material.getRoughness(), manager);
    tpack.emissive = build_img_texture(material.getEmissive(), manager);
    tpack.opacity = build_img_texture(material.getOpacity(), manager);
    tpack.specular = build_img_texture(material.getSpecular(), manager);
    tpack.ao = build_img_texture(material.getAmbientOcclusion(), manager);
    return assign_random_material(tpack, manager);
  }

}  // namespace nova

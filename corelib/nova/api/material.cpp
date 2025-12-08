#include "api_common.h"
#include "manager/NovaResourceManager.h"
#include "material/NovaMaterials.h"
#include "private_includes.h"
#include "texturing/NovaTextureInterface.h"
#include "texturing/nova_texturing.h"
namespace nova {
  // Aluminum
  const glm::vec3 eta_Al = glm::vec3(1.657f, 0.880f, 0.521f);
  const glm::vec3 k_Al = glm::vec3(9.223f, 6.269f, 4.837f);

  // Copper
  const glm::vec3 eta_Cu = glm::vec3(0.271f, 0.676f, 1.316f);
  const glm::vec3 k_Cu = glm::vec3(3.609f, 2.624f, 2.292f);

  // Gold
  const glm::vec3 eta_Au = glm::vec3(0.182f, 0.421f, 1.373f);
  const glm::vec3 k_Au = glm::vec3(3.424f, 2.345f, 1.770f);

  // Silver
  const glm::vec3 eta_Ag = glm::vec3(0.155f, 0.143f, 0.135f);
  const glm::vec3 k_Ag = glm::vec3(4.828f, 3.122f, 2.146f);

  // Iron
  const glm::vec3 eta_Fe = glm::vec3(2.911f, 2.949f, 2.584f);
  const glm::vec3 k_Fe = glm::vec3(3.089f, 2.931f, 2.767f);

  // Chromium
  const glm::vec3 eta_Cr = glm::vec3(3.177f, 3.182f, 2.220f);
  const glm::vec3 k_Cr = glm::vec3(3.334f, 3.329f, 3.088f);

  // Nickel
  const glm::vec3 eta_Ni = glm::vec3(1.910f, 1.570f, 1.134f);
  const glm::vec3 k_Ni = glm::vec3(2.452f, 1.935f, 1.654f);

  // Titanium
  const glm::vec3 eta_Ti = glm::vec3(2.745f, 2.389f, 1.532f);
  const glm::vec3 k_Ti = glm::vec3(3.130f, 2.617f, 1.955f);

  // Platinum
  const glm::vec3 eta_Pt = glm::vec3(2.375f, 2.084f, 1.845f);
  const glm::vec3 k_Pt = glm::vec3(4.265f, 3.715f, 3.136f);

  // Zinc
  const glm::vec3 eta_Zn = glm::vec3(1.370f, 0.920f, 0.666f);
  const glm::vec3 k_Zn = glm::vec3(3.100f, 2.570f, 2.330f);

  // Cobalt
  const glm::vec3 eta_Co = glm::vec3(2.260f, 1.590f, 1.310f);
  const glm::vec3 k_Co = glm::vec3(4.980f, 4.430f, 4.090f);

  // Lead
  const glm::vec3 eta_Pb = glm::vec3(1.910f, 1.830f, 1.440f);
  const glm::vec3 k_Pb = glm::vec3(3.510f, 3.400f, 3.180f);

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

    return manager.getMaterialData().addMaterial<nova::material::PrincipledMaterial>(tpack);
  }

}  // namespace nova

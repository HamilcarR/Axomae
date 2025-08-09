#include "private_includes.h"
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

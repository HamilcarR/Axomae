#ifndef API_MATERIAL_H
#define API_MATERIAL_H
#include "api_common.h"

namespace nova {

  class Texture;

  class Material {
   public:
    virtual ~Material() = default;
    virtual ERROR_STATE registerAlbedo(const Texture &texture) = 0;
    virtual ERROR_STATE registerNormal(const Texture &texture) = 0;
    virtual ERROR_STATE registerMetallic(const Texture &texture) = 0;
    virtual ERROR_STATE registerEmissive(const Texture &texture) = 0;
    virtual ERROR_STATE registerRoughness(const Texture &texture) = 0;
    virtual ERROR_STATE registerOpacity(const Texture &texture) = 0;
    virtual ERROR_STATE registerSpecular(const Texture &texture) = 0;
    virtual ERROR_STATE registerAmbientOcclusion(const Texture &texture) = 0;
    virtual ERROR_STATE setRefractCoeff(float eta) = 0;
    virtual ERROR_STATE setReflectFuzz(float fuzz) = 0;
  };

  std::unique_ptr<Material> create_material();

}  // namespace nova

#endif

#ifndef API_MATERIAL_H
#define API_MATERIAL_H
#include "api_common.h"

namespace nova {

  class NvAbstractTexture;

  class NvAbstractMaterial {
   public:
    virtual ~NvAbstractMaterial() = default;
    virtual ERROR_STATE registerAlbedo(const NvAbstractTexture &texture);
    virtual ERROR_STATE registerNormal(const NvAbstractTexture &texture);
    virtual ERROR_STATE registerMetallic(const NvAbstractTexture &texture);
    virtual ERROR_STATE registerEmissive(const NvAbstractTexture &texture);
    virtual ERROR_STATE registerRoughness(const NvAbstractTexture &texture);
    virtual ERROR_STATE registerOpacity(const NvAbstractTexture &texture);
    virtual ERROR_STATE registerSpecular(const NvAbstractTexture &texture);
    virtual ERROR_STATE registerAmbientOcclusion(const NvAbstractTexture &texture);
    virtual ERROR_STATE setRefractCoeff(float eta);
    virtual ERROR_STATE setReflectFuzz(float fuzz);
  };

  inline std::unique_ptr<NvAbstractMaterial> create_material();

}  // namespace nova

#endif

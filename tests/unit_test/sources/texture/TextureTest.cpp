#include "Texture.h"
#include <internal/common/Factory.h>
#include <unit_test/Test.h>

#define TYPE_LIST \
  DiffuseTexture, NormalTexture, MetallicTexture, RoughnessTexture, AmbiantOcclusionTexture, SpecularTexture, EmissiveTexture, OpacityTexture, \
      CubemapTexture, EnvironmentMap2DTexture, IrradianceTexture, BRDFLookupTexture, FrameBufferTexture, RoughnessTexture

template<class HEAD, class... TAIL>
void check_returned_types() {
  if constexpr (sizeof...(TAIL) > 0) {
    U32TexData texdata{};
    std::unique_ptr<HEAD> instance = std::make_unique<PRVINTERFACE<HEAD, U32TexData *>>(&texdata);
    GenericTexture::TYPE type = instance->getTextureType();
    std::string toStr = instance->getTextureTypeCStr();
    EXPECT_EQ(toStr, std::string(type2str(type)));
    EXPECT_EQ(type, str2type(toStr.c_str()));
    check_returned_types<TAIL...>();
  }
}

TEST(TextureTest, getType) { check_returned_types<TYPE_LIST>(); }

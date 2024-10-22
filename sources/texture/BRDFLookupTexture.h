#ifndef BRDFLOOKUPTEXTURE_H
#define BRDFLOOKUPTEXTURE_H
#include "GenericTexture.h"

#include <internal/macro/project_macros.h>

/**
 * @class BRDFLookupTexture
 * @brief PBR BRDF texture .
 * @note Pre baked texture storing the amount of specular reflection for a given
 * triplets of (I , V , R) , where I is the incident light , V is the view
 * direction , and R a roughness value . IE , for (X , Y) being the texture
 * coordinates , Y is a roughness scale , X <==> (I dot V) is the angle between
 * the incident light and view direction.
 */
class BRDFLookupTexture : public GenericTexture {
 protected:
  explicit BRDFLookupTexture(TextureData *data);

 public:
  void bind() override;
  void unbind() override;
  void initialize(Shader *shader) override;
  void initializeTexture2D() override;
  static const char *getTextureTypeCStr();
  ax_no_discard TYPE getTextureType() const override { return BRDFLUT; }
};

#endif  // BRDFLOOKUPTEXTURE_H

#ifndef MaterialInterface_H
#define MaterialInterface_H
#include "internal/common/math/vector/Vector.h"
class TextureGroup;
class MaterialInterface {
 public:
  virtual ~MaterialInterface() = default;
  virtual void setEmissiveFactor(float factor) = 0;
  [[nodiscard]] virtual float getEmissiveFactor() const = 0;
  virtual void clean() = 0;
  virtual void initializeMaterial() = 0;
  [[nodiscard]] virtual bool isTransparent() const = 0;
  [[nodiscard]] virtual float getDielectricFactor() const = 0;
  [[nodiscard]] virtual float getRoughnessFactor() const = 0;
  [[nodiscard]] virtual float getTransmissionFactor() const = 0;
  virtual void setAlphaFactor(float transparency_value) = 0;
  [[nodiscard]] virtual float getAlphaFactor() const = 0;
  [[nodiscard]] virtual Vec2f getRefractiveIndex() const = 0;
  virtual void setRefractiveIndexValue(float n1, float n2) = 0;
  [[nodiscard]] virtual bool hasTextures() const = 0;
  virtual void addTexture(int image_database_index) = 0;
  [[nodiscard]] virtual const TextureGroup &getTextureGroup() const = 0;
};
#endif  // MaterialInterface_H

#ifndef LightInterface_H
#define LightInterface_H
#include "internal/common/math/math_utils.h"
#include "internal/macro/project_macros.h"
class LightInterface {
 public:
  virtual void setPosition(const glm::vec3 &pos) = 0;
  virtual void setSpecularColor(const glm::vec3 &col) = 0;
  virtual void setAmbiantColor(const glm::vec3 &col) = 0;
  virtual void setDiffuseColor(const glm::vec3 &col) = 0;
  ax_no_discard virtual const glm::vec3 &getPosition() const = 0;
  ax_no_discard virtual const glm::vec3 &getDiffuseColor() const = 0;
  ax_no_discard virtual const glm::vec3 &getAmbiantColor() const = 0;
  ax_no_discard virtual const glm::vec3 &getSpecularColor() const = 0;
  virtual void setIntensity(float s) = 0;
  ax_no_discard virtual float getIntensity() const = 0;
  virtual void setID(unsigned light_id) = 0;
  ax_no_discard virtual unsigned int getID() const = 0;
};

#endif  // LightInterface_H

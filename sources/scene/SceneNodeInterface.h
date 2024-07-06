#ifndef SceneNodeInterface_H
#define SceneNodeInterface_H
#include "math_utils.h"
/**
 * @class SceneNodeInterface
 * @brief Provides an interface for a scene node
 */
class SceneNodeInterface {
 public:
  virtual ~SceneNodeInterface() = default;
  virtual glm::mat4 computeFinalTransformation() = 0;
  [[nodiscard]] virtual const glm::mat4 &getLocalModelMatrix() const = 0;
  virtual void setLocalModelMatrix(const glm::mat4 &matrix) = 0;
  virtual void resetLocalModelMatrix() = 0;
  virtual void resetAccumulatedMatrix() = 0;
  virtual void setAccumulatedModelMatrix(const glm::mat4 &matrix) = 0;
  [[nodiscard]] virtual const glm::mat4 &getAccumulatedModelMatrix() const = 0;
  [[nodiscard]] virtual bool isIgnored() const = 0;
  virtual void setIgnore(bool ignore) = 0;
};

#endif  // SceneNodeInterface_H

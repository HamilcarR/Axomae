#ifndef ICAMERA_H
#define ICAMERA_H
#include "math_utils.h"

namespace controller::event {
  class Event;
}
class CameraInterface {
 public:
  virtual void processEvent(const controller::event::Event *event) = 0;
  virtual void computeViewSpace() = 0;
  virtual void computeProjectionSpace() = 0;
  virtual void computeViewProjection() = 0;
  virtual void zoomIn() = 0;
  virtual void zoomOut() = 0;
  virtual void reset() = 0;
  [[nodiscard]] virtual glm::mat4 getSceneRotationMatrix() const = 0;
  [[nodiscard]] virtual glm::mat4 getSceneTranslationMatrix() const = 0;
  [[nodiscard]] virtual const glm::mat4 &getViewProjection() const = 0;
  [[nodiscard]] virtual const glm::mat4 &getProjection() const = 0;
  [[nodiscard]] virtual const glm::mat4 &getView() const = 0;
};

#endif  // ICAMERA_H

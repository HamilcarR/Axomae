#ifndef ICAMERA_H
#define ICAMERA_H
#include "EventInterface.h"
#include "constants.h"
#include "math_utils.h"
namespace controller::event {
  class Event;
}
class CameraInterface : public EventInterface {
 public:
  ~CameraInterface() override = default;
  virtual void computeViewSpace() = 0;
  virtual void computeProjectionSpace() = 0;
  virtual void computeViewProjection() = 0;
  virtual void zoomIn() = 0;
  virtual void zoomOut() = 0;
  virtual void reset() = 0;
  [[nodiscard]] virtual const glm::mat4 &getSceneRotationMatrix() const = 0;
  [[nodiscard]] virtual const glm::mat4 &getSceneTranslationMatrix() const = 0;
  [[nodiscard]] virtual const glm::mat4 &getViewProjection() const = 0;
  [[nodiscard]] virtual const glm::mat4 &getProjection() const = 0;
  [[nodiscard]] virtual const glm::mat4 &getView() const = 0;
  [[nodiscard]] virtual const Dim2 *getScreenDimensions() const = 0;
  [[nodiscard]] virtual float getFar() const = 0;
  [[nodiscard]] virtual float getNear() const = 0;
};

#endif  // ICAMERA_H

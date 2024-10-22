#ifndef FREEPERSPECTIVECAMERA_H
#define FREEPERSPECTIVECAMERA_H
#include "Camera.h"
class FreePerspectiveCamera : public Camera {
 public:
  FreePerspectiveCamera();
  FreePerspectiveCamera(float degrees, Dim2 *screen, float near, float far);
  void processEvent(const controller::event::Event *event) override;
  void zoomIn() override;
  void zoomOut() override;
  void focus(const glm::vec3 &position) override;
  ax_no_discard const glm::mat4 &getSceneTranslationMatrix() const override;
  ax_no_discard const glm::mat4 &getSceneRotationMatrix() const override;
};

#endif  // FreePerspectiveCamera_H

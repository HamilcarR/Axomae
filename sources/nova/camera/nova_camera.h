#ifndef NOVA_CAMERA_H
#define NOVA_CAMERA_H
#include "math_utils.h"
namespace camera {
  struct CameraResourcesHolder {
    glm::mat4 projection;
    glm::mat4 view;
  };
}  // namespace camera
#endif  // NOVA_CAMERA_H

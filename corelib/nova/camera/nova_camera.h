#ifndef NOVA_CAMERA_H
#define NOVA_CAMERA_H
#include <internal/common/math/math_utils.h>
#include <internal/macro/project_macros.h>
namespace nova::camera {
  class CameraResourcesHolder {
   public:
    unsigned int screen_width;
    unsigned int screen_height;
    float far, near;
    float fov;  // In radians
    /*Projection*/
    glm::mat4 P;
    glm::mat4 inv_P;
    /*View*/
    glm::mat4 V;
    glm::mat4 inv_V;
    /* Projection * View*/
    glm::mat4 PV;
    glm::mat4 inv_PV;
    glm::vec3 position;
    glm::vec3 up_vector;
    glm::vec3 direction;
  };
}  // namespace nova::camera
#endif  // NOVA_CAMERA_H

#ifndef NOVA_CAMERA_H
#define NOVA_CAMERA_H
#include "math_utils.h"
namespace camera {
  struct CameraResourcesHolder {
    unsigned int screen_width;
    unsigned int screen_height;

    /*Projection*/
    glm::mat4 P;
    glm::mat4 inv_P;

    /*View*/
    glm::mat4 V;
    glm::mat4 inv_V;

    /* Projection * View*/
    glm::mat4 PV;
    glm::mat4 inv_PV;

    /* Projection * View * Model*/
    glm::mat4 PVM;
    glm::mat4 inv_PVM;

    /* Model */
    glm::mat4 M;
    glm::mat4 inv_M;

    /* Normal matrix */
    glm::mat3 N;
  };
}  // namespace camera
#endif  // NOVA_CAMERA_H

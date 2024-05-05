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

    /* View x Model */
    glm::mat4 VM;
    glm::mat4 inv_VM;

    /* Projection * View * Model*/
    glm::mat4 PVM;
    glm::mat4 inv_PVM;

    /* Primary scene rotation from camera*/
    glm::mat4 R;
    glm::mat4 inv_R;

    /* Primary scene transformation from camera*/
    glm::mat4 T;
    glm::mat4 inv_T;

    /* Primary scene transformation from camera (R x T)*/
    glm::mat4 M;
    glm::mat4 inv_M;

    /* Normal matrix */
    glm::mat3 N;

    glm::vec3 position;
    glm::vec3 up_vector;
    glm::vec3 direction;
  };
}  // namespace camera
#endif  // NOVA_CAMERA_H

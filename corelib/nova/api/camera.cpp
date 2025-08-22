#include "private_includes.h"

namespace nova {

  static void copy_camera(const Camera &cam, camera::CameraResourcesHolder &camera) {
    float vec3[3]{}, mat16[16]{};

    cam.getUpVector(vec3);
    camera.up_vector = f3_to_vec3(vec3);

    cam.getProjectionMatrix(mat16);
    camera.P = f16_to_mat4(mat16);

    cam.getViewMatrix(mat16);
    camera.V = f16_to_mat4(mat16);

    cam.getProjectionViewMatrix(mat16);
    camera.PV = f16_to_mat4(mat16);

    cam.getInverseProjectionViewMatrix(mat16);
    camera.inv_PV = f16_to_mat4(mat16);

    cam.getInverseProjectionMatrix(mat16);
    camera.inv_P = f16_to_mat4(mat16);

    cam.getInverseViewMatrix(mat16);
    camera.inv_V = f16_to_mat4(mat16);

    cam.getUpVector(vec3);
    camera.up_vector = f3_to_vec3(vec3);

    cam.getPosition(vec3);
    camera.position = f3_to_vec3(vec3);

    cam.getDirection(vec3);
    camera.direction = f3_to_vec3(vec3);

    camera.far = cam.getClipPlaneFar();
    camera.near = cam.getClipPlaneNear();
    camera.fov = cam.getFov();
    camera.screen_width = cam.getResolutionWidth();
    camera.screen_height = cam.getResolutionHeight();
  }

  NvCamera &NvCamera::operator=(const Camera &cam) {
    if (this != &cam)
      copy_camera(cam, camera);
    return *this;
  }

  NvCamera::NvCamera(const Camera &cam) { copy_camera(cam, camera); }

  ERROR_STATE NvCamera::getProjectionViewMatrix(float proj[16]) const {
    glm::mat4 pv;
    pv = camera.inv_PV;
    for (int i = 0; i < 16; i++)
      proj[i] = glm::value_ptr(pv)[i];
    return SUCCESS;
  }

  ERROR_STATE NvCamera::getInverseViewMatrix(float proj[16]) const {
    glm::mat4 iv;
    iv = camera.inv_V;
    for (int i = 0; i < 16; i++)
      proj[i] = glm::value_ptr(iv)[i];
    return SUCCESS;
  }

  ERROR_STATE NvCamera::getInverseProjectionViewMatrix(float proj[16]) const {
    glm::mat4 ipv;
    ipv = camera.inv_PV;
    for (int i = 0; i < 16; i++)
      proj[i] = glm::value_ptr(ipv)[i];
    return SUCCESS;
  }

  ERROR_STATE NvCamera::setUpVector(const float up[3]) {
    camera.up_vector = glm::vec3(up[0], up[1], up[2]);
    return SUCCESS;
  }

  ERROR_STATE NvCamera::setMatrices(const float projection_matrix[16], const float view_matrix[16]) {
    for (int i = 0; i < 16; i++)
      glm::value_ptr(camera.P)[i] = projection_matrix[i];
    for (int i = 0; i < 16; i++)
      glm::value_ptr(camera.V)[i] = view_matrix[i];

    camera.inv_P = glm::inverse(camera.P);
    camera.inv_V = glm::inverse(camera.V);
    camera.PV = camera.P * camera.V;
    camera.inv_PV = glm::inverse(camera.PV);
    return SUCCESS;
  }

  ERROR_STATE NvCamera::setPosition(const float pos[3]) {
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    return SUCCESS;
  }

  ERROR_STATE NvCamera::setDirection(const float dir[3]) {
    camera.direction = glm::vec3(dir[0], dir[1], dir[2]);
    return SUCCESS;
  }

  ERROR_STATE NvCamera::setClipPlaneFar(float far_plane) {
    camera.far = far_plane;
    return SUCCESS;
  }

  ERROR_STATE NvCamera::setClipPlaneNear(float near_plane) {
    camera.near = near_plane;
    return SUCCESS;
  }

  ERROR_STATE NvCamera::setResolutionWidth(unsigned width) {
    camera.screen_width = width;
    return SUCCESS;
  }

  ERROR_STATE NvCamera::setResolutionHeight(unsigned height) {
    camera.screen_height = height;
    return SUCCESS;
  }

  ERROR_STATE NvCamera::setFov(float fov) {
    camera.fov = fov;
    return SUCCESS;
  }

  ERROR_STATE NvCamera::getUpVector(float up[3]) const {
    up[0] = camera.up_vector.x;
    up[1] = camera.up_vector.y;
    up[2] = camera.up_vector.z;
    return SUCCESS;
  }

  ERROR_STATE NvCamera::getProjectionMatrix(float proj[16]) const {
    for (int i = 0; i < 16; i++)
      proj[i] = glm::value_ptr(camera.P)[i];
    return SUCCESS;
  }

  ERROR_STATE NvCamera::getViewMatrix(float view[16]) const {
    for (int i = 0; i < 16; i++)
      view[i] = glm::value_ptr(camera.V)[i];
    return SUCCESS;
  }

  ERROR_STATE NvCamera::getPosition(float pos[3]) const {
    pos[0] = camera.position.x;
    pos[1] = camera.position.y;
    pos[2] = camera.position.z;
    return SUCCESS;
  }

  ERROR_STATE NvCamera::getDirection(float dir[3]) const {
    dir[0] = camera.direction.x;
    dir[1] = camera.direction.y;
    dir[2] = camera.direction.z;
    return SUCCESS;
  }

  float NvCamera::getClipPlaneFar() const { return camera.far; }

  float NvCamera::getClipPlaneNear() const { return camera.near; }

  unsigned NvCamera::getResolutionWidth() const { return camera.screen_width; }

  unsigned NvCamera::getResolutionHeight() const { return camera.screen_height; }

  float NvCamera::getFov() const { return camera.fov; }

}  // namespace nova

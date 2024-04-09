#include "Camera.h"
#include "Axomae_macros.h"
#include "EventController.h"
#include "constants.h"
#include "math_utils.h"

/**********************************************************************************************************************************************/
Camera::Camera() : world_up(glm::vec3(0, 1, 0)) {
  position = glm::vec3(0, 0, -1.f);
  local_transformation = glm::mat4(1.f);
  view = glm::mat4(1.f);
  projection = glm::mat4(1.f);
  view_projection = glm::mat4(1.f);
  camera_up = glm::vec3(0.f);
  cursor_position = glm::vec2(0);
  type = EMPTY;
}

Camera::Camera(float deg, float near, float far, const Dim2 *screen) : Camera() {
  fov = deg;
  this->far = far;
  this->near = near;
  screen_dimensions = screen;
}

void Camera::reset() {
  projection = glm::mat4(1.f);
  view = glm::mat4(1.f);
  target = glm::vec3(0.f);
  position = glm::vec3(0, 0, -1.f);
  direction = glm::vec3(0.f);
  camera_up = glm::vec3(0.f);
  view_projection = glm::mat4(1.f);
  cursor_position = glm::vec2(0);
}

void Camera::computeProjectionSpace() {
  projection = glm::perspective(glm::radians(fov), ((float)(screen_dimensions->width)) / ((float)(screen_dimensions->height)), near, far);
}

void Camera::computeViewProjection() {
  computeProjectionSpace();
  computeViewSpace();
  view_projection = projection * view;
}

void Camera::computeViewSpace() {
  direction = glm::normalize(position - target);
  right = glm::normalize(glm::cross(world_up, direction));
  camera_up = glm::cross(direction, right);
  view = glm::lookAt(position, target, camera_up);
}

#include "nova_camera.h"
namespace nova::camera {
  void CameraResourcesHolder::setPosition(const glm::vec3 &position_) { position = position_; }

  void CameraResourcesHolder::setDirection(const glm::vec3 &direction_) { direction = direction_; }

  void CameraResourcesHolder::setScreenWidth(int width) { screen_width = width; }

  void CameraResourcesHolder::setScreenHeight(int height) { screen_height = height; }

  void CameraResourcesHolder::setUpVector(const glm::vec3 &up) { up_vector = up; }

  void CameraResourcesHolder::setProjection(const glm::mat4 &projection) { P = projection; }

  void CameraResourcesHolder::setInvProjection(const glm::mat4 &inv_projection) { inv_P = inv_projection; }

  void CameraResourcesHolder::setView(const glm::mat4 &view) { V = view; }

  void CameraResourcesHolder::setInvView(const glm::mat4 &inv_view) { inv_V = inv_view; }

  const glm::vec3 &CameraResourcesHolder::getUpVector() const { return up_vector; }

  const glm::mat4 &CameraResourcesHolder::getProjection() const { return P; }

  const glm::mat4 &CameraResourcesHolder::getInvProjection() const { return inv_P; }

  const glm::mat4 &CameraResourcesHolder::getView() const { return V; }

  const glm::mat4 &CameraResourcesHolder::getInvView() const { return inv_V; }

  const glm::vec3 &CameraResourcesHolder::getPosition() const { return position; }

  const glm::vec3 &CameraResourcesHolder::getDirection() const { return direction; }

  int CameraResourcesHolder::getScreenWidth() const { return screen_width; }

  int CameraResourcesHolder::getScreenHeight() const { return screen_height; }
}  // namespace nova::camera
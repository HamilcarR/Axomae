#ifndef CAMERA_H
#define CAMERA_H

#include "CameraInterface.h"
#include "Node.h"

/**
 * @file Camera.h
 */

/**
 * @brief Base Camera class
 */
class Camera : public CameraInterface, public SceneTreeNode {
 public:
  enum TYPE : signed { EMPTY = -1, ARCBALL = 0, PERSPECTIVE = 1 };

 protected:
  TYPE type{};
  float near{};
  float far{};
  float fov{};
  glm::mat4 projection{};
  glm::mat4 view{};
  glm::mat4 view_projection{};
  glm::vec3 position{};
  glm::vec3 target{};
  glm::vec3 right{};
  glm::vec3 direction{};
  glm::vec3 camera_up{};
  const glm::vec3 world_up{};
  const Dim2 *screen_dimensions{};
  glm::vec2 cursor_position{}; /**<Screen space coordinates of the cursor*/

 protected:
  Camera();
  Camera(float degrees, float clip_near, float clip_far, const Dim2 *screen);

 public:
  void computeViewSpace() override;
  void computeProjectionSpace() override;
  void computeViewProjection() override;
  void setUpVector(const glm::vec3 &up) { camera_up = up; }
  void setView(const glm::mat4 &_view) { view = _view; }
  void setTarget(const glm::vec3 &_target) { target = _target; }
  void setPosition(const glm::vec3 &new_pos) { position = new_pos; }
  void reset() override;
  [[nodiscard]] virtual const glm::vec3 &getPosition() const { return position; }
  [[nodiscard]] const glm::mat4 &getViewProjection() const override { return view_projection; }
  [[nodiscard]] const glm::mat4 &getProjection() const override { return projection; }
  [[nodiscard]] const glm::mat4 &getView() const override { return view; }
  [[nodiscard]] TYPE getType() const { return type; }
  [[nodiscard]] const Dim2 *getScreenDimensions() const override { return screen_dimensions; }
  [[nodiscard]] float getNear() const override { return near; }
  [[nodiscard]] float getFar() const override { return far; }
};

#endif

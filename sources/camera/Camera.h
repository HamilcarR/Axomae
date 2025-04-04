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
  /* in degrees */
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
  ax_no_discard virtual glm::mat4 getTransformedView() const;
  ax_no_discard const glm::vec3 &getUpVector() const override { return camera_up; }
  ax_no_discard const glm::vec3 &getDirection() const override { return direction; }
  ax_no_discard const glm::vec3 &getPosition() const override { return position; }
  ax_no_discard const glm::mat4 &getViewProjection() const override { return view_projection; }
  ax_no_discard const glm::mat4 &getProjection() const override { return projection; }
  ax_no_discard const glm::mat4 &getView() const override { return view; }
  ax_no_discard TYPE getType() const { return type; }
  ax_no_discard const Dim2 *getScreenDimensions() const override { return screen_dimensions; }
  ax_no_discard float getNear() const override { return near; }
  ax_no_discard float getFar() const override { return far; }
  /* in degrees */
  ax_no_discard float getFov() const override { return fov; }
  ax_no_discard float getRatio() const override { return (float)screen_dimensions->width / (float)screen_dimensions->height; }

  /* static */
 public:
  static glm::mat4 computeProjectionMatrix(float fov_degree, float near, float far, float ratio);
};

#endif

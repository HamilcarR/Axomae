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
  const Dim2 *ratio_dimensions{};
  glm::vec2 cursor_position{}; /**<Screen space coordinates of the cursor*/

 protected:
  Camera();
  Camera(float degrees, float clip_near, float clip_far, const Dim2 *screen);

 public:
  void computeViewSpace() override;
  void computeProjectionSpace() override;
  void computeViewProjection() override;
  void setView(const glm::mat4 &_view) { view = _view; }
  void setTarget(const glm::vec3 &_target) { target = _target; }
  void setPosition(const glm::vec3 &new_pos) { position = new_pos; }
  void reset() override;
  [[nodiscard]] virtual const glm::vec3 &getPosition() const { return position; }
  [[nodiscard]] const glm::mat4 &getViewProjection() const override { return view_projection; }
  [[nodiscard]] const glm::mat4 &getProjection() const override { return projection; }
  [[nodiscard]] const glm::mat4 &getView() const override { return view; }
  [[nodiscard]] TYPE getType() const { return type; }
};

/**
 * @brief Arcball Camera class
 */
class ArcballCamera : public Camera {
 protected:
  float angle{};  /**<Angle of rotation*/
  float radius{}; /**<Camera orbit radius*/
  bool radius_updated{};
  glm::vec3 ndc_mouse_position{};       /**<Current NDC coordinates of the cursor*/
  glm::vec3 ndc_mouse_start_position{}; /**<NDC coordinates of the cursor at the start of a click event*/
  glm::quat rotation{};                 /**<Quaternion representing the scene's rotation*/
  glm::quat last_rotation{};            /**<Last rotation after the release event*/
  glm::mat4 translation{};              /**<Translation of the scene*/
  glm::mat4 last_translation{};         /**<Translation of the scene after release event*/
  glm::vec3 axis{};                     /**<Axis of rotation according to the direction of the mouse sweep*/
  glm::vec3 panning_offset{};           /**<Variable representing the new world position of the scene after translation */
  glm::mat4 scene_rotation_matrix{};    /**<Computed rotation matrix of the scene if the camera is an Arcball*/
  glm::mat4 scene_translation_matrix{}; /**<Computed translation matrix of the scene*/

 private:
  glm::vec3 delta_position{}; /**<NDC coordinates difference of the cursor , between two frames*/
  float default_radius{};

 public:
  ArcballCamera();
  ArcballCamera(float degrees, float near, float far, float radius, const Dim2 *screen);
  void processEvent(const controller::event::Event *event) override;
  void zoomIn() override;
  void zoomOut() override;
  void reset() override;
  [[nodiscard]] glm::mat4 getSceneTranslationMatrix() const override;
  [[nodiscard]] glm::mat4 getSceneRotationMatrix() const override;
  void computeViewSpace() override;

 protected:
  virtual void rotate();
  virtual void translate();
  virtual void updateZoom(float step);
};

// TODO: [AX-10] Implement free perspective camera
class FreePerspectiveCamera : public Camera {
 public:
  FreePerspectiveCamera();
  FreePerspectiveCamera(float degrees, Dim2 *screen, float near, float far);
  void processEvent(const controller::event::Event *event) override;
  void zoomIn() override;
  void zoomOut() override;
  [[nodiscard]] glm::mat4 getSceneTranslationMatrix() const override;
  [[nodiscard]] glm::mat4 getSceneRotationMatrix() const override;
};

#endif

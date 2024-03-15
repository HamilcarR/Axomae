#ifndef CAMERA_H
#define CAMERA_H

#include "ICamera.h"
#include "Node.h"
/**
 * @file Camera.h
 */

/**
 * @brief Base Camera class
 */
class Camera : public ICamera, public SceneTreeNode {
 public:
  enum TYPE : signed { EMPTY = -1, ARCBALL = 0, PERSPECTIVE = 1 };

  void computeViewSpace() override;

  void computeProjectionSpace() override;

  void computeViewProjection() override;

  void setView(const glm::mat4 &_view) { view = _view; }

  void setTarget(const glm::vec3 &_target) { target = _target; }

  void setPosition(const glm::vec3 &new_pos) { position = new_pos; }

  void onLeftClick() override = 0;

  void onRightClick() override = 0;

  void onLeftClickRelease() override = 0;

  void onRightClickRelease() override = 0;

  void movePosition() override = 0;

  void zoomIn() override = 0;

  void zoomOut() override = 0;

  void reset() override;

  [[nodiscard]] virtual const glm::vec3 &getPosition() const { return position; }

  [[nodiscard]] glm::mat4 getSceneRotationMatrix() const override = 0;

  [[nodiscard]] glm::mat4 getSceneTranslationMatrix() const override = 0;

  [[nodiscard]] const glm::mat4 &getViewProjection() const override { return view_projection; }

  [[nodiscard]] const glm::mat4 &getProjection() const override { return projection; }

  [[nodiscard]] const glm::mat4 &getView() const override { return view; }

  [[nodiscard]] TYPE getType() const { return type; }

 protected:
  Camera();

  Camera(float degrees, float clip_near, float clip_far, const Dim2 *screen, const MouseState *pointer = nullptr);

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
  const MouseState *mouse_state_pointer{};
  const Dim2 *ratio_dimensions{};
};

/**
 * @brief Arcball Camera class
 * The Arcball camera is a static camera , that computes a rotation , and a translation that will be applied to the
 * scene , instead of the camera. This gives the illusion that the camera is moving around the scene.
 */
class ArcballCamera : public Camera {
 public:
  ArcballCamera();

  ArcballCamera(float degrees, float near, float far, float radius, const Dim2 *screen, const MouseState *pointer);

  void computeViewSpace() override;
  /**
   * @brief Left click event , calculates the rotation matrix (will rotate the scene around the camera)
   *
   */
  void onLeftClick() override;

  /**
   * @brief Right click event , calculates the translation matrix (will translate the scene relative to the camera)
   *
   */
  void onRightClick() override;

  void onLeftClickRelease() override;

  void onRightClickRelease() override;

  void zoomIn() override;

  void zoomOut() override;

  void reset() override;

  /**
   * @brief Get the Scene Translation Matrix object
   *
   * @return const glm::mat4&
   */
  [[nodiscard]] glm::mat4 getSceneTranslationMatrix() const override;
  /**
   * @brief Get the Scene Rotation Matrix object
   *
   * @return const glm::mat4&
   */
  [[nodiscard]] glm::mat4 getSceneRotationMatrix() const override;

 protected:
  /**
   * @brief Rotates the scene according to the mouse position on left click.
   *
   */
  virtual void rotate();
  /**
   * @brief Translates the scene according to the mouse position on right click .
   *
   */
  virtual void translate();
  /**
   * @brief Combines the translation and rotation matrix.
   *
   */
  void movePosition() override;
  /**
   * @brief Updates the zoom factor.
   *
   * @param step New zoom factor
   */
  virtual void updateZoom(float step);

 protected:
  float angle{};                        /**<Angle of rotation*/
  float radius{};                       /**<Camera orbit radius*/
  glm::vec2 cursor_position{};          /**<Screen space coordinates of the cursor*/
  glm::vec3 ndc_mouse_position{};       /**<Current NDC coordinates of the cursor*/
  glm::vec3 ndc_mouse_start_position{}; /**<NDC coordinates of the cursor at the start of a click event*/
  glm::vec3 ndc_mouse_last_position{};  /**<Last NDC coordinates of the cursor after a release event*/
  glm::quat rotation{};                 /**<Quaternion representing the scene's rotation*/
  glm::quat last_rotation{};            /**<Last rotation after the release event*/
  glm::mat4 translation{};              /**<Translation of the scene*/
  glm::mat4 last_translation{};         /**<Translation of the scene after release event*/
  glm::vec3 axis{};                     /**<Axis of rotation according to the direction of the mouse sweep*/
  glm::vec3 panning_offset{};           /**<Variable representing the new world position of the scene after translation */
  bool radius_updated{};
  glm::mat4 scene_rotation_matrix{};    /**<Computed rotation matrix of the scene if the camera is an Arcball*/
  glm::mat4 scene_translation_matrix{}; /**<Computed translation matrix of the scene*/
 private:
  glm::vec3 delta_position{}; /**<NDC coordinates difference of the cursor , between two frames*/
  float default_radius{};
};

// TODO: [AX-10] Implement free perspective camera
class FreePerspectiveCamera : public Camera {
 public:
  FreePerspectiveCamera();
  FreePerspectiveCamera(float degrees, Dim2 *screen, float near, float far, const MouseState *pointer = nullptr);
  void onLeftClick() override;
  void onRightClick() override;
  void onLeftClickRelease() override;
  void onRightClickRelease() override;
  void movePosition() override;
  void zoomIn() override;
  void zoomOut() override;
  [[nodiscard]] glm::mat4 getSceneTranslationMatrix() const override;
  [[nodiscard]] glm::mat4 getSceneRotationMatrix() const override;
};

#endif

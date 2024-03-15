#ifndef CAMERA_H
#define CAMERA_H

#include "Node.h"

/**
 * @file Camera.h
 *
 * @brief Interface for the Camera classes
 *
 */

/**
 * @brief Base Camera class
 *
 *
 */
class Camera : public SceneTreeNode {
 public:
  enum TYPE : signed { EMPTY = -1, ARCBALL = 0, PERSPECTIVE = 1 };

  Camera();

  /**
   * @brief Constructor for the Camera class , which initializes various properties of the
   * camera.
   *
   * @param degrees The field of view angle in degrees for the camera.
   * @param screen A pointer to an object of type Dim2, which contains information about the size
   * of the screen or window where the camera will be rendering.
   * @param clip_near The distance from the camera to the near clipping plane. Any objects closer to the
   * camera than this distance will not be visible.
   * @param clip_far The far plane of the camera's frustum, which determines the maximum distance at which
   * objects will be visible.
   * @param pointer The "pointer" parameter is a pointer to a MouseState object, which is used to
   * track the state of the mouse (e.g. position, button clicks) for camera movement and control.
   */
  Camera(float degrees, float clip_near, float clip_far, const Dim2 *screen, const MouseState *pointer = nullptr);

  virtual void computeViewSpace();

  virtual void computeProjectionSpace();

  virtual void computeViewProjection();

  virtual void setView(const glm::mat4 &_view) { view = _view; }

  virtual void setTarget(const glm::vec3 &_target) { target = _target; }

  virtual void setPosition(const glm::vec3 &new_pos) { position = new_pos; }

  virtual void onLeftClick() = 0;

  virtual void onRightClick() = 0;

  virtual void onLeftClickRelease() = 0;

  virtual void onRightClickRelease() = 0;

  virtual void movePosition() = 0;

  virtual void zoomIn() = 0;

  virtual void zoomOut() = 0;

  virtual void reset();

  [[nodiscard]] virtual const glm::vec3 &getPosition() const { return position; }

  [[nodiscard]] virtual glm::mat4 getSceneRotationMatrix() const = 0;

  [[nodiscard]] virtual glm::mat4 getSceneTranslationMatrix() const = 0;

  [[nodiscard]] const glm::mat4 &getViewProjection() const { return view_projection; }

  [[nodiscard]] const glm::mat4 &getProjection() const { return projection; }

  [[nodiscard]] const glm::mat4 &getView() const { return view; }

  [[nodiscard]] TYPE getType() const { return type; }

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
   * @brief Left click event , calculates the rotation matrix.
   *
   */
  void onLeftClick() override;
  /**
   * @brief Right click event , calculates the translation matrix.
   *
   */
  void onRightClick() override;
  /**
   * @brief Left click release event.
   *
   */
  void onLeftClickRelease() override;
  /**
   * @brief Right click release event.
   *
   */
  void onRightClickRelease() override;
  /**
   * @brief Mouse wheel up event.
   *
   */
  void zoomIn() override;
  /**
   * @brief Mouse wheel down event.
   *
   */
  void zoomOut() override;
  /**
   * @brief Resets the camera state.
   *
   */
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

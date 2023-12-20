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
  /**
   * @brief Camera types enumeration
   *
   */
  enum TYPE : signed {
    EMPTY = -1,     /**<Empty camera type*/
    ARCBALL = 0,    /**<Arcball camera type*/
    PERSPECTIVE = 1 /**<Free camera type*/
  };

  Camera();

  /**
   * @brief Constructor for the Camera class , which initializes various properties of the
   * camera.
   *
   * @param degrees The field of view angle in degrees for the camera.
   * @param screen A pointer to an object of type ScreenSize, which contains information about the size
   * of the screen or window where the camera will be rendering.
   * @param clip_near The distance from the camera to the near clipping plane. Any objects closer to the
   * camera than this distance will not be visible.
   * @param clip_far The far plane of the camera's frustum, which determines the maximum distance at which
   * objects will be visible.
   * @param pointer The "pointer" parameter is a pointer to a MouseState object, which is used to
   * track the state of the mouse (e.g. position, button clicks) for camera movement and control.
   */
  Camera(float degrees, ScreenSize *screen, float clip_near, float clip_far, const MouseState *pointer = nullptr);

  /**
   * @brief Destroy the Camera object
   */
  virtual ~Camera();

  /**
   * @brief Calculates the view matrix
   */
  virtual void computeViewSpace();

  /**
   * @brief Calculates the projection matrix
   */
  virtual void computeProjectionSpace();

  /**
   * @brief Calculates the product of the two matrices : Projection x View
   */
  virtual void computeViewProjection();

  /**
   * @brief Get the View matrix
   *
   * @return * glm::mat4
   */
  virtual glm::mat4 getView() const { return view; }

  /**
   * @brief Set the new view matrix
   *
   * @param _view View Matrix
   */
  virtual void setView(glm::mat4 _view) { view = _view; }

  /**
   * @brief Set the target position
   *
   * @param _target Point that the camera will view
   */
  virtual void setTarget(glm::vec3 _target) { target = _target; }

  /**
   * @brief Get the product of (Projection x View) matrix
   *
   * @return glm::mat4
   */
  virtual glm::mat4 getViewProjection() const { return view_projection; }

  /**
   * @brief Get the Projection matrix
   *
   * @return glm::mat4
   */
  virtual glm::mat4 getProjection() const { return projection; }

  /**
   * @brief On left click event
   *
   */
  virtual void onLeftClick() = 0;

  /**
   * @brief On right click event
   *
   */
  virtual void onRightClick() = 0;

  /**
   * @brief On left click release event
   *
   */
  virtual void onLeftClickRelease() = 0;

  /**
   * @brief On right click release event
   *
   */
  virtual void onRightClickRelease() = 0;

  /**
   * @brief On move mouse event
   *
   */
  virtual void movePosition() = 0;

  /**
   * @brief On mouse wheel up event
   *
   */
  virtual void zoomIn() = 0;

  /**
   * @brief On mouse wheel down event
   *
   */
  virtual void zoomOut() = 0;

  /**
   * @brief Camera states reset
   *
   */
  virtual void reset();

  /**
   * @brief Get the camera position vector
   *
   * @return const glm::vec3&
   */
  virtual const glm::vec3 &getPosition() const { return position; }

  /**
   * @brief Set the camera's new position
   *
   * @param new_pos New position
   */
  virtual void setPosition(glm::vec3 new_pos) { position = new_pos; }

  /**
   * @brief Get the rotation matrix of the scene
   *
   * @return const glm::mat4&
   */
  virtual const glm::mat4 getSceneRotationMatrix() const = 0;

  /**
   * @brief Get the translation matrix of the scene
   *
   * @return const glm::mat4&
   */
  virtual const glm::mat4 getSceneTranslationMatrix() const = 0;

  /**
   * @brief Returns the type of the camera
   *
   * @return Camera::TYPE
   */
  TYPE getType() const { return type; }

 protected:
  TYPE type;                             /**<Camera type */
  float near;                            /**<Near plane */
  float far;                             /**<Far plane */
  float fov;                             /**<FOV of the camera in degrees*/
  glm::mat4 projection;                  /**<Projection matrix */
  glm::mat4 view;                        /**<View Matrix */
  glm::mat4 view_projection;             /**<Projection x View product matrix */
  glm::vec3 position;                    /**<World space position of the camera */
  glm::vec3 target;                      /**<Target viewed by the camera */
  glm::vec3 right;                       /**<Right vector computed */
  glm::vec3 direction;                   /**<Direction vector of the camera */
  glm::vec3 camera_up;                   /**<Up vector of the camera */
  const glm::vec3 world_up;              /**<World space up vector */
  const MouseState *mouse_state_pointer; /**<Pointer on a MouseState structure keeping track of the mouse data*/
  ScreenSize *ratio_dimensions;          /**<Pointer on a ScreenSize structure with fields giving informations about a width ,
                                            and a height.*/
};

/**
 * @brief Arcball Camera class
 * The Arcball camera is a static camera , that computes a rotation , and a translation that will be applied to the
 * scene , instead of the camera. This gives the illusion that the camera is moving around the scene.
 */
class ArcballCamera : public Camera {
 public:
  /**
   * @brief ArcballCamera default constructor
   *
   */
  ArcballCamera();
  /**
   * @brief ArcballCamera Constructor
   *
   * @param degrees The field of view angle in degrees for the camera .
   * @param screen The `screen` parameter is a pointer to an object of the `ScreenSize` class, which
   * contains information about the size of the screen .
   * @param near The distance to the near clipping plane of the camera's frustum .
   * @param far The distance to the far clipping plane .
   * @param radius The radius of the Camera's orbit.
   * @param pointer Pointer to a MouseState object, contains information about the current state of the mouse (e.g.
   * position, button presses, etc.) .
   */
  ArcballCamera(float degrees, ScreenSize *screen, float near, float far, float radius, const MouseState *pointer);
  /**
   * @brief Destroy the Arcball Camera object
   *
   */
  virtual ~ArcballCamera();
  /**
   * @brief Calculates the view matrix
   *
   */
  virtual void computeViewSpace();
  /**
   * @brief Left click event , calculates the rotation matrix.
   *
   */
  virtual void onLeftClick() override;
  /**
   * @brief Right click event , calculates the translation matrix.
   *
   */
  virtual void onRightClick() override;
  /**
   * @brief Left click release event.
   *
   */
  virtual void onLeftClickRelease() override;
  /**
   * @brief Right click release event.
   *
   */
  virtual void onRightClickRelease() override;
  /**
   * @brief Mouse wheel up event.
   *
   */
  virtual void zoomIn() override;
  /**
   * @brief Mouse wheel down event.
   *
   */
  virtual void zoomOut() override;
  /**
   * @brief Resets the camera state.
   *
   */
  virtual void reset();

  /**
   * @brief Get the Scene Translation Matrix object
   *
   * @return const glm::mat4&
   */
  virtual const glm::mat4 getSceneTranslationMatrix() const;
  /**
   * @brief Get the Scene Rotation Matrix object
   *
   * @return const glm::mat4&
   */
  virtual const glm::mat4 getSceneRotationMatrix() const;

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
  virtual void movePosition() override;
  /**
   * @brief Updates the zoom factor.
   *
   * @param step New zoom factor
   */
  virtual void updateZoom(float step);

 protected:
  float angle;                        /**<Angle of rotation*/
  float radius;                       /**<Camera orbit radius*/
  glm::vec2 cursor_position;          /**<Screen space coordinates of the cursor*/
  glm::vec3 ndc_mouse_position;       /**<Current NDC coordinates of the cursor*/
  glm::vec3 ndc_mouse_start_position; /**<NDC coordinates of the cursor at the start of a click event*/
  glm::vec3 ndc_mouse_last_position;  /**<Last NDC coordinates of the cursor after a release event*/
  glm::quat rotation;                 /**<Quaternion representing the scene's rotation*/
  glm::quat last_rotation;            /**<Last rotation after the release event*/
  glm::mat4 translation;              /**<Translation of the scene*/
  glm::mat4 last_translation;         /**<Translation of the scene after release event*/
  glm::vec3 axis;                     /**<Axis of rotation according to the direction of the mouse sweep*/
  glm::vec3 panning_offset;           /**<Variable representing the new world position of the scene after translation */
  bool radius_updated;
  glm::mat4 scene_rotation_matrix;    /**<Computed rotation matrix of the scene if the camera is an Arcball*/
  glm::mat4 scene_translation_matrix; /**<Computed translation matrix of the scene*/
 private:
  glm::vec3 delta_position; /**<NDC coordinates difference of the cursor , between two frames*/
  float default_radius;
};

// TODO: [AX-10] Implement free perspective camera
class FreePerspectiveCamera : public Camera {
 public:
  FreePerspectiveCamera();
  FreePerspectiveCamera(float degrees, ScreenSize *screen, float near, float far, const MouseState *pointer = nullptr);
  virtual ~FreePerspectiveCamera();
  virtual void onLeftClick() override;
  virtual void onRightClick() override;
  virtual void onLeftClickRelease() override;
  virtual void onRightClickRelease() override;
  virtual void movePosition() override;
  virtual void zoomIn() override;
  virtual void zoomOut() override;
  virtual const glm::mat4 getSceneTranslationMatrix() const override;
  virtual const glm::mat4 getSceneRotationMatrix() const override;
};

#endif

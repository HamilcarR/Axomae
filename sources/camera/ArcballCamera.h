#ifndef ARCBALLCAMERA_H
#define ARCBALLCAMERA_H
#include "Camera.h"

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
  ax_no_discard const glm::mat4 &getSceneTranslationMatrix() const override;
  ax_no_discard const glm::mat4 &getSceneRotationMatrix() const override;
  ax_no_discard glm::mat4 getTransformedView() const override;
  void computeViewSpace() override;
  void focus(const glm::vec3 &position) override;

 protected:
  virtual void rotate();
  virtual void translate();
  virtual void updateZoom(float step);
};
#endif  // ArcballCamera_H

#include "../includes/Camera.h"
#include <cmath>

constexpr float DELTA_ZOOM = 1.f;
constexpr float ANGLE_EPSILON = 0.0001f;   // we use these to avoid nan values when angles or vector lengths become too small
constexpr float VECTOR_EPSILON = 0.0001f;  //
constexpr float PANNING_SENSITIVITY = 10.f;

/**********************************************************************************************************************************************/
Camera::Camera() : world_up(glm::vec3(0, 1, 0)) {
  position = glm::vec3(1, 0, -1.f);
  target = glm::vec3(0, 0, 0);
  direction = glm::vec3(0, 0, 0);
  right = glm::vec3(0, 0, 0);
  local_transformation = glm::mat4(1.f);
  type = EMPTY;
}

Camera::Camera(float deg, ScreenSize *screen, float near, float far, const MouseState *pointer) : Camera() {
  type = EMPTY;
  position = glm::vec3(0, 0, -1.f);
  target = glm::vec3(0, 0, 0);
  direction = glm::vec3(0, 0, 0);
  right = glm::vec3(0, 0, 0);
  view = glm::mat4(1.f);
  projection = glm::mat4(1.f);
  fov = deg;
  view_projection = glm::mat4(1.f);
  this->far = far;
  this->near = near;
  ratio_dimensions = screen;
  mouse_state_pointer = pointer;
  projection = glm::mat4(1.f);
}

Camera::~Camera() {}

void Camera::reset() {
  projection = glm::mat4(1.f);
  view = glm::mat4(1.f);
  target = glm::vec3(0, 0, 0);
  position = glm::vec3(0, 0, -1.f);
  direction = glm::vec3(0, 0, 0);
  camera_up = glm::vec3(0, 0, 0);
  view_projection = glm::mat4(1.f);
}

void Camera::computeProjectionSpace() {
  projection = glm::perspective(glm::radians(fov), ((float)(ratio_dimensions->width)) / ((float)(ratio_dimensions->height)), near, far);
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
/**********************************************************************************************************************************************/
ArcballCamera::ArcballCamera() { reset(); }

/**
 * This is a constructor for an ArcballCamera object that sets its properties and initializes its
 * radius.
 *
 * @param radians The field of view angle in deg for the camera.
 * @param screen The `screen` parameter is a pointer to an object of the `ScreenSize` class, which
 * contains information about the size of the screen or window in which the camera will be used. This
 * information is typically used to calculate the aspect ratio of the screen, which is important for
 * rendering the scene correctly
 * @param near The distance to the near clipping plane of the camera's frustum. Any objects closer than
 * this distance will not be rendered.
 * @param far The "far" parameter in this context refers to the distance from the camera beyond which
 * objects will not be rendered. It is a part of the perspective projection matrix used in 3D graphics
 * rendering.
 * @param radius The distance between the camera and the center of rotation in an arcball camera. It
 * determines how far the camera is from the object being viewed.
 * @param pointer The "pointer" parameter is a pointer to a MouseState object, which  contains
 * information about the current state of the mouse (e.g. position, button presses, etc.). This is
 *  used by the ArcballCamera class to track user input and update the camera's
 * position/orientation accordingly
 */
ArcballCamera::ArcballCamera(float deg, ScreenSize *screen, float near, float far, float radius, const MouseState *pointer)
    : Camera(deg, screen, near, far, pointer) {
  reset();
  default_radius = radius;
  name = "Arcball-Camera";
  this->radius = radius;
}

ArcballCamera::~ArcballCamera() {}

void ArcballCamera::reset() {
  Camera::reset();
  type = ARCBALL;
  angle = 0.f;
  radius = default_radius;
  radius_updated = false;
  ndc_mouse_start_position = ndc_mouse_position = position = glm::vec3(0, 0, radius);
  cursor_position = glm::vec2(0);
  rotation = last_rotation = glm::quat(1.f, 0.f, 0.f, 0.f);
  axis = glm::vec3(0.f);
  target = glm::vec3(0.f);
  panning_offset = glm::vec3(0);
  delta_position = glm::vec3(0.f);
  translation = last_translation = scene_translation_matrix = scene_rotation_matrix = local_transformation = glm::mat4(1.f);
}

/**
 * Calculates a rotation of the scene based on the mouse movement .
 * The rotation isn't applied to the camera here , the camera stays static.
 * The scene is instead rotated , giving the illusion of a camera rotation.
 */
void ArcballCamera::rotate() {
  axis = glm::normalize(glm::cross(ndc_mouse_start_position, ndc_mouse_position + glm::vec3(VECTOR_EPSILON)));
  float temp_angle = glm::acos(glm::dot(ndc_mouse_position, ndc_mouse_start_position));
  angle = temp_angle > ANGLE_EPSILON ? temp_angle : ANGLE_EPSILON;
  rotation = glm::angleAxis(angle, axis);
  rotation = rotation * last_rotation;
}

/**
 * This function translates the scene based on the difference between the current and starting mouse
 * positions in normalized device coordinates.
 *
 */
void ArcballCamera::translate() {
  delta_position = ndc_mouse_position - ndc_mouse_start_position;
  auto new_delta = glm::vec3(glm::inverse(scene_rotation_matrix) * (glm::vec4(delta_position, 1.f)));
  panning_offset += new_delta * PANNING_SENSITIVITY;
  ndc_mouse_start_position = ndc_mouse_position;
}

/**
 * This function computes the view , and model matrices based on the rotation and translation of the scene.
 *
 */
void ArcballCamera::computeViewSpace() {
  cursor_position = glm::vec2(mouse_state_pointer->pos_x, mouse_state_pointer->pos_y);
  if (mouse_state_pointer->left_button_clicked) {
    movePosition();
    rotate();
    glm::mat4 rotation_matrix = glm::mat4_cast(rotation);
    ndc_mouse_position = rotation_matrix * glm::vec4(ndc_mouse_position + glm::vec3(0, 0, radius), 0);
    scene_rotation_matrix = rotation_matrix;
  } else if (mouse_state_pointer->right_button_clicked) {
    movePosition();
    translate();
    translation = glm::translate(glm::mat4(1.f), panning_offset);
    scene_translation_matrix = translation;
    last_translation = translation;
  } else {
    scene_rotation_matrix = glm::mat4_cast(last_rotation);
    scene_translation_matrix = last_translation;
  }
  local_transformation = scene_rotation_matrix * scene_translation_matrix;
  direction = target - glm::vec3(0, 0, radius);
  position = glm::vec3(0, 0, radius);
  view = glm::lookAt(position, target, world_up);
}

const glm::mat4 ArcballCamera::getSceneRotationMatrix() const { return scene_rotation_matrix; }

const glm::mat4 ArcballCamera::getSceneTranslationMatrix() const { return scene_translation_matrix; }

/**
 * The function calculates the z-axis value for a given x and y coordinate within a specified radius.
 *
 * @param x The x-coordinate in NDC.
 * @param y The y-coordinate in NDC.
 * @param radius The radius of the Arcball orbit.
 *
 * @return A float value which represents the projected z coordinate of an NDC (x,y) point on the sphere .
 */
static float get_z_axis(float x, float y, float radius) {
  if (((x * x) + (y * y)) <= (radius * radius / 2))
    return (float)sqrt((radius * radius) - (x * x) - (y * y));
  else
    return (float)((radius * radius) / 2) / sqrt((x * x) + (y * y));
}

/**
 * The function computes the position of the mouse in normalized device coordinates based on the cursor
 * position.
 */
void ArcballCamera::movePosition() {
  if (mouse_state_pointer->left_button_clicked) {
    ndc_mouse_position.x = ((cursor_position.x - (ratio_dimensions->width / 2)) / (ratio_dimensions->width / 2)) * radius;
    ndc_mouse_position.y = (((ratio_dimensions->height / 2) - cursor_position.y) / (ratio_dimensions->height / 2)) * radius;
    ndc_mouse_position.z = get_z_axis(ndc_mouse_position.x, ndc_mouse_position.y, radius);
    ndc_mouse_position = glm::normalize(ndc_mouse_position);
  }
  if (mouse_state_pointer->right_button_clicked) {
    ndc_mouse_position.x = ((cursor_position.x - (ratio_dimensions->width / 2)) / (ratio_dimensions->width / 2));
    ndc_mouse_position.y = (((ratio_dimensions->height / 2) - cursor_position.y) / (ratio_dimensions->height / 2));
    ndc_mouse_position.z = 0.f;
  }
}

/**
 * This function calculates the starting position of the mouse in normalized device coordinates for an
 * Arcball camera when the left mouse button is clicked.
 */
void ArcballCamera::onLeftClick() {
  ndc_mouse_start_position.x = ((cursor_position.x - (ratio_dimensions->width / 2)) / (ratio_dimensions->width / 2)) * radius;
  ndc_mouse_start_position.y = (((ratio_dimensions->height / 2) - cursor_position.y) / (ratio_dimensions->height / 2)) * radius;
  ndc_mouse_start_position.z = get_z_axis(ndc_mouse_start_position.x, ndc_mouse_start_position.y, radius);
  ndc_mouse_start_position = glm::normalize(ndc_mouse_start_position);
}

/**
 * This function updates the camera's rotation and position based on the last mouse click release
 * event.
 */
void ArcballCamera::onLeftClickRelease() {
  last_rotation = rotation;
  rotation = glm::quat(1.f, 0.f, 0.f, 0.f);
  ndc_mouse_last_position = position;
}

/**
 * This function sets the starting position of the mouse in normalized device coordinates when the
 * right mouse button is clicked.
 */
void ArcballCamera::onRightClick() {
  ndc_mouse_start_position.x = ((cursor_position.x - (ratio_dimensions->width / 2)) / (ratio_dimensions->width / 2));
  ndc_mouse_start_position.y = (((ratio_dimensions->height / 2) - cursor_position.y) / (ratio_dimensions->height / 2));
  ndc_mouse_start_position.z = 0.f;
}

void ArcballCamera::onRightClickRelease() {}

void ArcballCamera::updateZoom(float step) { radius += step; }

void ArcballCamera::zoomIn() {
  if ((radius - DELTA_ZOOM) >= DELTA_ZOOM)
    updateZoom(-DELTA_ZOOM);
}

void ArcballCamera::zoomOut() { updateZoom(DELTA_ZOOM); }

/**********************************************************************************************************************************************/
FreePerspectiveCamera::FreePerspectiveCamera() : Camera() { type = PERSPECTIVE; }

FreePerspectiveCamera::FreePerspectiveCamera(float deg, ScreenSize *screen, float near, float far, const MouseState *pointer)
    : Camera(deg, screen, near, far, pointer) {
  type = PERSPECTIVE;
}

FreePerspectiveCamera::~FreePerspectiveCamera() {}
void FreePerspectiveCamera::movePosition() {}

void FreePerspectiveCamera::onLeftClick() {}

void FreePerspectiveCamera::onLeftClickRelease() {}

void FreePerspectiveCamera::onRightClick() {}

void FreePerspectiveCamera::onRightClickRelease() {}

void FreePerspectiveCamera::zoomIn() {}
void FreePerspectiveCamera::zoomOut() {}
const glm::mat4 FreePerspectiveCamera::getSceneTranslationMatrix() const { return glm::mat4(1.f); }

const glm::mat4 FreePerspectiveCamera::getSceneRotationMatrix() const { return glm::mat4(1.f); }

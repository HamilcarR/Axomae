#include "ArcballCamera.h"
#include "EventController.h"
#include "Logger.h"
#include "math_utils.h"
#include "project_macros.h"

constexpr float DELTA_ZOOM = 1.f;
constexpr float ANGLE_EPSILON = 0.0001f;   // we use these to avoid nan values when angles or vector lengths become too small
constexpr float VECTOR_EPSILON = 0.0001f;  //
constexpr float PANNING_SENSITIVITY = 10.f;

ArcballCamera::ArcballCamera() : Camera() { ArcballCamera::reset(); }

ArcballCamera::ArcballCamera(float deg, float near, float far, float radius_, const Dim2 *screen)
    : Camera(deg, near, far, screen), radius(radius_), default_radius(radius_) {
  ArcballCamera::reset();
  name = "Arcball-Camera";
  type = ARCBALL;
}

void ArcballCamera::reset() {
  Camera::reset();
  type = ARCBALL;
  angle = 0.f;
  radius = default_radius;
  radius_updated = false;
  ndc_mouse_start_position = ndc_mouse_position = position = glm::vec3(0, 0, radius);
  rotation = last_rotation = glm::quat(1.f, 0.f, 0.f, 0.f);
  axis = glm::vec3(0.f);
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
 */
void ArcballCamera::translate() {
  delta_position = ndc_mouse_position - ndc_mouse_start_position;
  glm::vec3 new_delta = glm::vec3(glm::inverse(scene_rotation_matrix) * (glm::vec4(delta_position, 1.f)));
  panning_offset += new_delta * PANNING_SENSITIVITY;
  ndc_mouse_start_position = ndc_mouse_position;
}

const glm::mat4 &ArcballCamera::getSceneRotationMatrix() const { return scene_rotation_matrix; }

const glm::mat4 &ArcballCamera::getSceneTranslationMatrix() const { return scene_translation_matrix; }

/**
 * The function calculates the z-axis value for a given x and y coordinate within a specified radius.
 *
 * @param x The x-coordinate in NDC.
 * @param y The y-coordinate in NDC.
 * @param radius The radius of the Arcball orbit.
 *
 * @return A float value which represents the projected z coordinate of an NDC (x,y) point on the sphere .
 */
inline float get_z_axis(float x, float y, float radius) {
  if (((x * x) + (y * y)) <= (radius * radius / 2))
    return (float)sqrtf((radius * radius) - (x * x) - (y * y));
  else
    return (float)((radius * radius) / 2) / sqrtf((x * x) + (y * y));
}

/* register NDC mouse position at the beginning of a click event*/
inline glm::vec3 orbit_start_ndc_position(const glm::vec2 &cursor_position, float radius, float ratio_w, float ratio_h) {
  glm::vec3 ndc_mouse_start_position{};
  ndc_mouse_start_position.x = ((cursor_position.x - ratio_w / 2) / (ratio_w / 2)) * radius;
  ndc_mouse_start_position.y = (((ratio_h / 2) - cursor_position.y) / (ratio_h / 2)) * radius;
  ndc_mouse_start_position.z = get_z_axis(ndc_mouse_start_position.x, ndc_mouse_start_position.y, radius);
  ndc_mouse_start_position = glm::normalize(ndc_mouse_start_position);
  return ndc_mouse_start_position;
}

inline glm::vec3 pan_start_ndc_position(const glm::vec2 &cursor_position, float ratio_w, float ratio_h) {
  glm::vec3 ndc_mouse_start_position{};
  ndc_mouse_start_position.x = ((cursor_position.x - (ratio_w / 2)) / (ratio_w / 2));
  ndc_mouse_start_position.y = (((ratio_h / 2) - cursor_position.y) / (ratio_h / 2));
  ndc_mouse_start_position.z = 0.f;
  return ndc_mouse_start_position;
}

/*NDC mouse positions while in mouse move event */
inline glm::vec3 orbit_on_move_ndc_position(const glm::vec2 &cursor_position, float radius, float ratio_w, float ratio_h) {
  glm::vec3 ndc_mouse_position{};
  ndc_mouse_position.x = ((cursor_position.x - ratio_w / 2) / (ratio_w / 2)) * radius;
  ndc_mouse_position.y = (((ratio_h / 2) - cursor_position.y) / (ratio_h / 2)) * radius;
  ndc_mouse_position.z = get_z_axis(ndc_mouse_position.x, ndc_mouse_position.y, radius);
  ndc_mouse_position = glm::normalize(ndc_mouse_position);
  return ndc_mouse_position;
}
inline glm::vec3 pan_on_move_ndc_position(const glm::vec2 &cursor_position, float ratio_w, float ratio_h) {
  glm::vec3 ndc_mouse_position{};
  ndc_mouse_position.x = ((cursor_position.x - (ratio_w / 2)) / (ratio_w / 2));
  ndc_mouse_position.y = (((ratio_h / 2) - cursor_position.y) / (ratio_h / 2));
  ndc_mouse_position.z = 0.f;
  return ndc_mouse_position;
}

void ArcballCamera::computeViewSpace() {
  position = glm::vec3(0, 0, radius);
  direction = -position;
  right = glm::normalize(glm::cross(world_up, direction));
  camera_up = glm::cross(direction, right);
  local_transformation = scene_rotation_matrix * scene_translation_matrix;
  view = glm::lookAt(position, glm::vec3(0.f), camera_up);
}

/* Move to a controller */
void ArcballCamera::processEvent(const controller::event::Event *event) {
  using ev = controller::event::Event;
  AX_ASSERT(event != nullptr, "Provided event structure null.");
  cursor_position = glm::vec2(event->mouse_state.pos_x, event->mouse_state.pos_y);
  if (!(event->flag & ev::EVENT_MOUSE_MOVE)) {
    /* Left / Right mouse click management */
    if (event->flag & ev::EVENT_MOUSE_L_PRESS) {
      /* computes mouse start position in NDC */
      ndc_mouse_start_position = orbit_start_ndc_position(cursor_position, radius, (float)screen_dimensions->width, (float)screen_dimensions->height);
    } else if (event->flag & ev::EVENT_MOUSE_L_RELEASE) {
      last_rotation = rotation;
    } else if (event->flag & ev::EVENT_MOUSE_R_PRESS) {
      ndc_mouse_start_position = pan_start_ndc_position(cursor_position, (float)screen_dimensions->width, (float)screen_dimensions->height);
    } else if (event->flag & ev::EVENT_MOUSE_R_RELEASE) {
      // EVENT_MOUSE_R_RELEASE
    }
  } else {
    if (event->flag & ev::EVENT_MOUSE_L_PRESS) {
      ndc_mouse_position = orbit_on_move_ndc_position(cursor_position, radius, (float)screen_dimensions->width, (float)screen_dimensions->height);
      rotate();
      glm::mat4 rotation_matrix = glm::mat4_cast(rotation);
      ndc_mouse_position = rotation_matrix * glm::vec4(ndc_mouse_position + glm::vec3(0, 0, radius), 0);
      scene_rotation_matrix = rotation_matrix;
    } else if (event->flag & ev::EVENT_MOUSE_R_PRESS) {
      ndc_mouse_position = pan_on_move_ndc_position(cursor_position, (float)screen_dimensions->width, (float)screen_dimensions->height);
      translate();
      translation = glm::translate(glm::mat4(1.f), panning_offset - target); /*Allows to pan from a different focus position*/
      scene_translation_matrix = translation;
      last_translation = translation;
    } else {
      scene_rotation_matrix = glm::mat4_cast(last_rotation);
      scene_translation_matrix = last_translation;
    }
    computeViewSpace();
  }

  /* Wheel events */
  if (event->flag & ev::EVENT_MOUSE_WHEEL) {
    if (event->mouse_state.wheel_delta > 0) {
      zoomIn();
    } else if (event->mouse_state.wheel_delta < 0) {
      zoomOut();
    }
  }
}

void ArcballCamera::updateZoom(float step) {
  radius += step;
  computeViewSpace();
}

void ArcballCamera::zoomIn() {
  if ((radius - DELTA_ZOOM) >= DELTA_ZOOM)
    updateZoom(-DELTA_ZOOM);
}

void ArcballCamera::zoomOut() { updateZoom(DELTA_ZOOM); }

void ArcballCamera::focus(const glm::vec3 &pos) {
  glm::vec3 local_pos = glm::inverse(local_transformation) * glm::vec4(pos, 1.f);
  setTarget(local_pos);
  last_translation = glm::translate(glm::mat4(1.f), -local_pos);
  scene_translation_matrix = last_translation;
  computeViewSpace();
}

glm::mat4 ArcballCamera::getTransformedView() const { return view * local_transformation; }

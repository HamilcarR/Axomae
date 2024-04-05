#ifndef MATH_CAMERA_H
#define MATH_CAMERA_H
#include "Vector.h"
#include <glm/common.hpp>
#include <glm/glm.hpp>
#include <glm/matrix.hpp>
namespace math::camera {

  enum SPACE : unsigned { WORLD, EYE };

  struct camera_ray {
    Vec3f origin;
    Vec3f direction;
    SPACE compute_space;
  };

  inline camera_ray ray(int screen_x, int screen_y, int width, int height, const glm::mat4 &projection, const glm::mat4 &view, SPACE space) {
    const float ndc_x = (float)(2 * screen_x - width) / (float)width;
    const float ndc_y = (float)(height - 2 * screen_y) / (float)height;
    camera_ray r;
    glm::mat4 inv_P = glm::inverse(projection);
    glm::vec4 o = glm::vec4(0.f);
    glm::vec4 d = glm::normalize(inv_P * glm::vec4(ndc_x, ndc_y, -1.f, 0.f));
    r.compute_space = space;
    if (space == EYE) {
      r.origin = {o.x, o.y, o.z};
      r.direction = {d.x, d.y, d.z};
      return r;
    }
    glm::mat4 inv_V = glm::inverse(view);
    o = inv_P * glm::vec4(ndc_x, ndc_y, -1.f, 1.f);
    o /= o.w;
    o = inv_V * glm::vec4(o.x, o.y, o.z, 1.f);

    d = inv_P * glm::vec4(ndc_x, ndc_y, 0.f, 1.f);
    d /= d.w;
    d = inv_V * glm::vec4(d.x, d.y, d.z, 1.f);

    d = glm::normalize(d - o);
    r.origin = {o.x, o.y, o.z};
    r.direction = {d.x, d.y, d.z};
    return r;
  }

}  // namespace math::camera
#endif  // math_camera_H

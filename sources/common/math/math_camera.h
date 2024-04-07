#ifndef MATH_CAMERA_H
#define MATH_CAMERA_H
#include "Vector.h"
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/matrix.hpp>
namespace math::camera {

  struct camera_ray {
    Vec3f near;
    Vec3f far;
  };

  /* returns world-space ray */
  inline camera_ray ray(int screen_x, int screen_y, int width, int height, const glm::mat4 &projection, const glm::mat4 &view) {
    const float ndc_x = (float)(2 * screen_x - width) / (float)width;
    const float ndc_y = (float)(height - 2 * screen_y) / (float)height;
    const glm::mat4 inv_P = glm::inverse(projection);
    const glm::mat4 inv_V = glm::inverse(view);

    glm::vec4 o = inv_P * glm::vec4(ndc_x, ndc_y, -1.f, 1.f);
    o /= o.w;
    o = inv_V * glm::vec4(o.x, o.y, o.z, 1.f);
    glm::vec4 d = inv_P * glm::vec4(ndc_x, ndc_y, -1.f, 0.f);
    d = glm::normalize(inv_V * glm::vec4(d.x, d.y, -1.f, 0.f));
    camera_ray r;
    r.near = {o.x, o.y, o.z};
    r.far = {d.x, d.y, d.z};
    return r;
  }

}  // namespace math::camera
#endif  // math_camera_H

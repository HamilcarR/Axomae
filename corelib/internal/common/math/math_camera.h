#ifndef MATH_CAMERA_H
#define MATH_CAMERA_H
#include "math_includes.h"
#include "vector/Vector.h"

namespace math::camera {

  struct camera_ray {
    glm::vec3 near;
    glm::vec3 far;
  };

  /* To screen */
  /***********************************************************************************************/

  inline glm::vec2 ndc2screen(float ndc_x, float ndc_y, int width, int height) {
    int x = (int)((float)width * (ndc_x - 1.f) / 2.f);
    int y = (int)((float)height * (1.f - ndc_y) / 2.f);
    return {x, y};
  }

  inline glm::vec2 view2ndc(const glm::vec4 &perspective_vec, const glm::mat4 &P) { return P * perspective_vec; }
  inline glm::vec4 world2view(const glm::vec4 &view_vec, const glm::mat4 &P) { return P * view_vec; }

  /* To world*/
  /***********************************************************************************************/
  inline glm::vec2 screen2ndc(int x, int y, int width, int height) {
    const float ndc_x = (float)(2 * x - width) / (float)width;
    const float ndc_y = (float)(height - 2 * y) / (float)height;
    return {ndc_x, ndc_y};
  }

  inline glm::vec4 ndc2view(float ndc_x, float ndc_y, const glm::mat4 &inv_P, bool is_point) {
    glm::vec4 point = inv_P * glm::vec4(ndc_x, ndc_y, -1.f, is_point ? 1.f : 0.f);
    point /= is_point ? point.w : 1;
    return point;
  }

  inline glm::vec4 view2world(const glm::vec4 &vec, const glm::mat4 &inv_V) { return inv_V * vec; }

  /***********************************************************************************************/

  inline camera_ray ray_inv_mat(float ndc_x, float ndc_y, const glm::mat4 &inv_P, const glm::mat4 &inv_V) {
    glm::vec4 o = inv_P * glm::vec4(ndc_x, ndc_y, -1.f, 1.f);
    o /= o.w;
    o = inv_V * glm::vec4(o.x, o.y, o.z, 1.f);

    glm::vec4 d = inv_P * glm::vec4(ndc_x, ndc_y, -1.f, 0.f);
    d = glm::normalize(inv_V * glm::vec4(d.x, d.y, -1.f, 0.f));

    camera_ray r{};
    r.near = {o.x, o.y, o.z};
    r.far = {d.x, d.y, d.z};
    return r;
  }

  inline camera_ray ray_inv_mat(int x, int y, int width, int height, const glm::mat4 &inv_P, const glm::mat4 &inv_V) {
    const glm::vec2 to_ndc = screen2ndc(x, y, width, height);
    const float ndc_x = to_ndc.x;
    const float ndc_y = to_ndc.y;
    return ray_inv_mat(ndc_x, ndc_y, inv_P, inv_V);
  }

  /* returns world-space ray */
  inline camera_ray ray(int screen_x, int screen_y, int width, int height, const glm::mat4 &projection, const glm::mat4 &view) {
    const glm::mat4 inv_P = glm::inverse(projection);
    const glm::mat4 inv_V = glm::inverse(view);
    return ray_inv_mat(screen_x, screen_y, width, height, inv_P, inv_V);
  }

}  // namespace math::camera
#endif  // math_camera_H

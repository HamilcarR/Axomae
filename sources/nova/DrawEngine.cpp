#include "DrawEngine.h"
#include "math_camera.h"
using namespace nova;
inline bool hit_sphere(const glm::vec3 &center, float radius, const Ray &r, hit_data *r_hit) {
  const glm::vec3 oc = r.origin - center;
  const float b = 2.f * glm::dot(r.direction, oc);
  const float a = glm::dot(r.direction, r.direction);
  const float c = glm::dot(oc, oc) - radius * radius;
  const float determinant = b * b - 4 * a * c;
  if (determinant >= 0) {
    float t1 = (-b - std::sqrt(determinant)) * 0.5f * a;
    if (t1 <= 0)
      return false;
    r_hit->t = t1;
    r_hit->position = r.pointAt(t1);
    r_hit->normal = (r_hit->position - center);
    return true;
  }
  return false;
}

glm::vec4 NovaRenderEngineLR::engine_sample_color(const Ray &ray, const NovaResources *nova_resources) {
  hit_data hit_d;
  float alpha = 1.f;
  glm::vec4 center{0, 0, 0, 1.f};
  center = center;
  if (hit_sphere(center, 1, ray, &hit_d)) {
    glm::vec3 normal = glm::normalize(glm::vec4(hit_d.normal, 0.f));
    glm::vec3 sampled_color = texturing::sample_cubemap(normal, &nova_resources->envmap_data);
    return {sampled_color, alpha};
  }
  glm::vec3 sample_vector = nova_resources->camera_data.inv_T * glm::vec4(ray.direction, 0.f);
  return {texturing::sample_cubemap(sample_vector, &nova_resources->envmap_data), 1.f};
}

void NovaRenderEngineLR::engine_render_tile(
    float *dest_buffer, int width_limit_low, int width_limit_high, int height_limit_low, int height_limit_high, const NovaResources *nova_resources) {
  AX_ASSERT(nova_resources, "Scene description NULL.");
  for (int y = height_limit_high - 1; y > height_limit_low; y--)
    for (int x = width_limit_low; x < width_limit_high; x++) {
      math::camera::camera_ray r = math::camera::ray_inv_mat(x,
                                                             nova_resources->camera_data.screen_height - y,
                                                             nova_resources->camera_data.screen_width,
                                                             nova_resources->camera_data.screen_height,
                                                             nova_resources->camera_data.inv_P,
                                                             glm::inverse(nova_resources->camera_data.V * nova_resources->camera_data.M));
      Ray ray(r.near, r.far);
      glm::vec4 rgb = engine_sample_color(ray, nova_resources);
      unsigned int idx = (y * nova_resources->camera_data.screen_width + x) * 4;
      dest_buffer[idx] = rgb.r;
      dest_buffer[idx + 1] = rgb.g;
      dest_buffer[idx + 2] = rgb.b;
      dest_buffer[idx + 3] = rgb.a;
    }
}

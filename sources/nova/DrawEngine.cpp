#include "DrawEngine.h"
#include "math_camera.h"

#include <unistd.h>

namespace nova {
  glm::vec4 NovaRenderEngineLR::engine_sample_color(const Ray &ray, const NovaResources *nova_resources) {
    hit_data hit_d;
    float alpha = 1.f;
    for (const auto &sphere : nova_resources->scene_data.objects) {
      if (sphere.hit(ray, 0.1f, 1000.f, hit_d, nullptr)) {
        glm::vec3 sample_vector = glm::reflect(ray.direction, glm::normalize(hit_d.normal));
        glm::vec3 sampled_color = texturing::sample_cubemap(sample_vector, &nova_resources->envmap_data);
        return {sampled_color, alpha};
      }
    }
    glm::vec3 sample_vector = ray.direction;  // nova_resources->camera_data.inv_T * glm::vec4(ray.direction, 0.f);
    return {texturing::sample_cubemap(sample_vector, &nova_resources->envmap_data), 1.f};
  }
  void NovaRenderEngineLR::engine_render_tile(float *dest_buffer,
                                              int width_limit_low,
                                              int width_limit_high,
                                              int height_limit_low,
                                              int height_limit_high,
                                              const NovaResources *nova_resources) {
    AX_ASSERT(nova_resources, "Scene description NULL.");
    for (int y = height_limit_high; y > height_limit_low; y--)
      for (int x = width_limit_low; x <= width_limit_high; x++) {
        glm::vec4 rgb{};
        math::camera::camera_ray r = math::camera::ray_inv_mat(x + math::random::fast_randb(),
                                                               nova_resources->camera_data.screen_height - (y + math::random::fast_randb()),
                                                               nova_resources->camera_data.screen_width,
                                                               nova_resources->camera_data.screen_height,
                                                               nova_resources->camera_data.inv_P,
                                                               nova_resources->camera_data.inv_VM);
        Ray ray(r.near, r.far);
        rgb = engine_sample_color(ray, nova_resources);
        unsigned int idx = (y * nova_resources->camera_data.screen_width + x) * 4;
        dest_buffer[idx] = rgb.r;
        dest_buffer[idx + 1] = rgb.g;
        dest_buffer[idx + 2] = rgb.b;
        dest_buffer[idx + 3] = rgb.a;
      }
  }
}  // namespace nova
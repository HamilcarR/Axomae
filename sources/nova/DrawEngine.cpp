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
  void NovaRenderEngineLR::engine_render_tile(HdrBufferStruct *buffers, const Tile &tile, const NovaResources *nova_resources) {
    AX_ASSERT(nova_resources, "Scene description is invalid.");
    for (int y = tile.height_end - 1; y >= tile.height_start; y = y - 1) {
      for (int x = tile.width_start; x < tile.width_end; x = x + 1) {
        const unsigned int idx = (y * tile.image_total_width + x) * 4;
        for (int i = 0; i < tile.sample_per_tile; i++) {
          const int dx = math::random::nrandi(-1, 1);
          const int dy = math::random::nrandi(-1, 1);
          math::camera::camera_ray r = math::camera::ray_inv_mat(x + dx,
                                                                 tile.image_total_height - (y + dy),
                                                                 tile.image_total_width,
                                                                 tile.image_total_height,
                                                                 nova_resources->camera_data.inv_P,
                                                                 nova_resources->camera_data.inv_VM);
          Ray ray(r.near, r.far);
          const glm::vec4 rgb = engine_sample_color(ray, nova_resources);
          for (int k = 0; k < 3; k++)
            buffers->accumulator_buffer[idx + k] += buffers->partial_buffer[idx + k];

          buffers->partial_buffer[idx] = rgb.r;
          buffers->partial_buffer[idx + 1] = rgb.g;
          buffers->partial_buffer[idx + 2] = rgb.b;
          buffers->partial_buffer[idx + 3] = rgb.a;
        }
      }
    }
  }
}  // namespace nova
#include "DrawEngine.h"
#include "math_camera.h"
#include "nova_material.h"

namespace nova {
  glm::vec4 NovaRenderEngineLR::engine_sample_color(const Ray &ray, const NovaResources *nova_resources, int depth) {
    hit_data hit_d;
    float alpha = 1.f;
    bool hit = false;
    float min_t = MAXFLOAT;
    const primitive::NovaPrimitiveInterface *last_primit = nullptr;
    for (const auto &primitive : nova_resources->scene_data.primitives)
      if (primitive->hit(ray, 0.001f, min_t, hit_d, nullptr)) {
        hit = true;
        min_t = hit_d.t;
        last_primit = primitive.get();
      }
    if (hit) {
      Ray out{};
      if (!last_primit || !last_primit->scatter(ray, out, hit_d) || depth < 0)
        return glm::vec4(0.f);
      glm::vec4 color = hit_d.attenuation;
      return color * engine_sample_color(out, nova_resources, depth - 1);
    }
    glm::vec3 sample_vector = ray.direction;
    return {texturing::sample_cubemap(sample_vector, &nova_resources->envmap_data), 1.f};
    // return glm::vec4(0.2f, 0.3f, 0.7f, 1.f) + glm::normalize(ray.direction).y * (glm::vec4(1.f, 1.f, 1.f, 1.f) - glm::vec4(1.f));
  }

  void NovaRenderEngineLR::engine_render_tile(HdrBufferStruct *buffers, Tile &tile, const NovaResources *nova_resources) {
    AX_ASSERT(nova_resources, "Scene description is invalid.");
    for (int y = tile.height_end - 1; y >= tile.height_start; y = y - 1)
      for (int x = tile.width_start; x < tile.width_end; x = x + 1) {
        const unsigned int idx = (y * tile.image_total_width + x) * 4;
        glm::vec4 rgb{};
        const glm::vec2 ndc = math::camera::screen2ndc(x, tile.image_total_height - y, tile.image_total_width, tile.image_total_height);
        for (int i = 0; i < tile.sample_per_tile; i++) {
          const float dx = math::random::nrandf(0, 0.003);
          const float dy = math::random::nrandf(0, 0.003);
          math::camera::camera_ray r = math::camera::ray_inv_mat(
              ndc.x + dx, ndc.y + dy, nova_resources->camera_data.inv_P, nova_resources->camera_data.inv_VM);
          Ray ray(r.near, r.far);
          rgb += engine_sample_color(ray, nova_resources, nova_resources->renderer_data.max_depth);
        }
        rgb /= (float)(tile.sample_per_tile);
        for (int k = 0; k < 3; k++)
          buffers->accumulator_buffer[idx + k] += buffers->partial_buffer[idx + k];
        buffers->partial_buffer[idx] = rgb.r;
        buffers->partial_buffer[idx + 1] = rgb.g;
        buffers->partial_buffer[idx + 2] = rgb.b;
        buffers->partial_buffer[idx + 3] = rgb.a;
      }
    tile.finished_render = true;
  }
}  // namespace nova

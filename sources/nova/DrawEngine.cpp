#include "DrawEngine.h"
#include "math_camera.h"
#include "nova_material.h"

constexpr float RAND_DX = 0.0005;
constexpr float RAND_DY = 0.0005;
namespace nova {
  glm::vec4 NovaRenderEngineLR::engine_sample_color(const Ray &ray, const NovaResources *nova_resources, int depth) {
    hit_data hit_d;
    bool hit = false;
    float min_t = MAXFLOAT;
    const primitive::NovaPrimitiveInterface *last_primit = nullptr;
    const aggregate::Bvhtl &bvh = nova_resources->acceleration_structure.accelerator;
    aggregate::bvh_helper_struct bvh_hit{min_t, nullptr, nova_resources->renderer_data.cancel_render};
    aggregate::base_options_bvh opts;
    opts.data = bvh_hit;

    hit = bvh.hit(ray, 0.001f, min_t, hit_d, &opts);
    last_primit = opts.data.last_prim;
    if (hit) {
      Ray out{};
      if (!last_primit || !last_primit->scatter(ray, out, hit_d) || depth < 0)
        return glm::vec4(0.f);
      glm::vec4 color = hit_d.attenuation;
      return color * engine_sample_color(out, nova_resources, depth - 1);
    }
    glm::vec3 sample_vector = ray.direction;
    return {texturing::sample_cubemap(sample_vector, &nova_resources->envmap_data), 1.f};
  }

  void NovaRenderEngineLR::engine_render_tile(HdrBufferStruct *buffers, Tile &tile, const NovaResources *nova_resources) {
    AX_ASSERT(nova_resources, "Scene description is invalid.");
    for (int y = tile.height_end - 1; y >= tile.height_start; y = y - 1)
      for (int x = tile.width_start; x < tile.width_end; x = x + 1) {
        if (*nova_resources->renderer_data.cancel_render)
          return;

        const unsigned int idx = (y * tile.image_total_width + x) * 4;
        glm::vec4 rgb{};
        /* Converts screen coordinates into NDC.*/
        const glm::vec2 ndc = math::camera::screen2ndc(x, tile.image_total_height - y, tile.image_total_width, tile.image_total_height);

        for (int i = 0; i < tile.sample_per_tile; i++) {

          /* Samples random direction around the pixel for AA. */
          const float dx = math::random::nrandf(-RAND_DX, RAND_DX);
          const float dy = math::random::nrandf(-RAND_DY, RAND_DY);

          math::camera::camera_ray r = math::camera::ray_inv_mat(
              ndc.x + dx, ndc.y + dy, nova_resources->camera_data.inv_P, nova_resources->camera_data.inv_VM);  // TODO : move VM away
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

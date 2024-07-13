#include "DrawEngine.h"
#include "manager/NovaResourceManager.h"
#include "material/nova_material.h"
#include "math_camera.h"
constexpr float RAND_DX = 0.0005;
constexpr float RAND_DY = 0.0005;
namespace nova {
  glm::vec4 NovaRenderEngineLR::engine_sample_color(const Ray &ray, const NovaResourceManager *nova_resources, int depth) {
    hit_data hit_d;
    bool hit = false;
    float min_t = MAXFLOAT;
    const primitive::NovaPrimitiveInterface *last_primit = nullptr;
    const aggregate::Bvhtl &bvh = nova_resources->getAccelerationData().accelerator;
    aggregate::bvh_helper_struct bvh_hit{min_t, nullptr, nova_resources->getEngineData().getCancelPtr()};
    aggregate::base_options_bvh opts;
    opts.data = bvh_hit;

    hit = bvh.hit(ray, 0.001f, min_t, hit_d, &opts);
    last_primit = opts.data.last_prim;
    if (hit) {
      Ray out{};
      if (!last_primit || !last_primit->scatter(ray, out, hit_d) || depth < 0)
        return glm::vec4(0.f);
      glm::vec4 color = hit_d.attenuation;
      glm::vec4 emit = hit_d.emissive;
      return emit + color * engine_sample_color(out, nova_resources, depth - 1);
    }
    glm::vec3 sample_vector = ray.direction;
    return {texturing::sample_cubemap(sample_vector, &nova_resources->getEnvmapData()), 1.f};
  }

  /**
   * TODO : Will use different states :
   * 1) Fast state : On move events , on redraw , on resize etc will trigger fast state .
   * The scheduler needs to be emptied , threads synchronized and stopped , and we redraw with
   * 1 ray/pixel , at 1 sample , at 1 depth in each tile , then copy the sampled value to the other pixels.
   * 2) Intermediary state : increase logarithmically the amount of pixels sampled + number of samples , at half depth .
   * 3) Final state : render at full depth , full sample size , full resolution.
   * Allows proper synchronization.
   *
   */
  void NovaRenderEngineLR::engine_render_tile(HdrBufferStruct *buffers, Tile &tile, const NovaResourceManager *nova_resources) {
    AX_ASSERT(nova_resources, "Scene description is invalid.");

    for (int y = tile.height_end - 1; y >= tile.height_start; y = y - 1)
      for (int x = tile.width_start; x < tile.width_end; x = x + 1) {

        unsigned int idx = 0;
        if (!nova_resources->getEngineData().isAxisVInverted())
          idx = (y * tile.image_total_width + x) * 4;
        else
          idx = ((tile.image_total_height - 1 - y) * tile.image_total_width + x) * 4;
        glm::vec4 rgb{};
        const glm::vec2 ndc = math::camera::screen2ndc(x, tile.image_total_height - y, tile.image_total_width, tile.image_total_height);
        for (int i = 0; i < tile.sample_per_tile; i++) {
          if (*nova_resources->getEngineData().getCancelPtr())
            return;
          /* Samples random direction around the pixel for AA. */
          const float dx = math::random::nrandf(-RAND_DX, RAND_DX);
          const float dy = math::random::nrandf(-RAND_DY, RAND_DY);
          math::camera::camera_ray r = math::camera::ray_inv_mat(
              ndc.x + dx, ndc.y + dy, nova_resources->getCameraData().getInvProjection(), nova_resources->getCameraData().getInvView());
          Ray ray(r.near, r.far);
          rgb += engine_sample_color(ray, nova_resources, nova_resources->getEngineData().getMaxDepth());
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
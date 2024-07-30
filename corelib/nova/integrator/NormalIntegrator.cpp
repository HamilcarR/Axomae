#include "NormalIntegrator.h"

namespace nova::integrator {

  void NormalIntegrator::render(RenderBuffers<float> *buffers, Tile &tile, const NovaResourceManager *nova_resource_manager) const {

    sampler::SobolSampler sobol = sampler::SobolSampler(tile.sample_per_tile, 2);
    sampler::SamplerInterface sampler = &sobol;
    for (int y = tile.height_end - 1; y >= tile.height_start; y = y - 1)
      for (int x = tile.width_start; x < tile.width_end; x = x + 1) {
        unsigned int idx = 0;
        if (!nova_resource_manager->getEngineData().isAxisVInverted())
          idx = (y * tile.image_total_width + x) * 4;
        else
          idx = ((tile.image_total_height - 1 - y) * tile.image_total_width + x) * 4;

        const glm::vec2 ndc = math::camera::screen2ndc(x, tile.image_total_height - y, tile.image_total_width, tile.image_total_height);
        if (*nova_resource_manager->getEngineData().getCancelPtr())
          return;
        math::camera::camera_ray r = math::camera::ray_inv_mat(
            ndc.x, ndc.y, nova_resource_manager->getCameraData().getInvProjection(), nova_resource_manager->getCameraData().getInvView());
        Ray ray(r.near, r.far);

        glm::vec4 rgb = Li(ray, nova_resource_manager, nova_resource_manager->getEngineData().getMaxDepth(), sampler);
        for (int k = 0; k < 3; k++) {
          buffers->accumulator_buffer[idx + k] += buffers->partial_buffer[idx + k];
          buffers->partial_buffer[idx] = rgb.r;
          buffers->partial_buffer[idx + 1] = rgb.g;
          buffers->partial_buffer[idx + 2] = rgb.b;
        }
        buffers->partial_buffer[idx + 3] = 1.f;
      }
    tile.finished_render = true;
  }

  glm::vec4 NormalIntegrator::Li(const Ray &ray, const NovaResourceManager *nova_resources, int depth, sampler::SamplerInterface &sampler) const {

    bvh_hit_data hit = bvh_hit(ray, nova_resources);
    if (hit.is_hit) {
      Ray out{};
      if (!hit.last_primit || !hit.last_primit->scatter(ray, out, hit.hit_d, sampler) || depth < 0)
        return glm::vec4(0.f);
      glm::vec4 normal = {hit.hit_d.normal, 1.f};
      return normal;
    }
    return glm::vec4(0.f);
  }
}  // namespace nova::integrator
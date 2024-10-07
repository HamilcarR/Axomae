#include "NormalIntegrator.h"

namespace nova::integrator {

  void NormalIntegrator::render(RenderBuffers<float> *buffers, Tile &tile, nova_eng_internals &nova_internals) const {
    const NovaResourceManager *nova_resource_manager = nova_internals.resource_manager;
    NovaExceptionManager *nova_exception_manager = nova_internals.exception_manager;
    sampler::RandomSampler random_sampler = sampler::RandomSampler();
    sampler::SamplerInterface sampler = &random_sampler;
    for (int y = tile.height_end - 1; y >= tile.height_start; y = y - 1)
      for (int x = tile.width_start; x < tile.width_end; x = x + 1) {
        validate(sampler, nova_internals);
        if (nova_exception_manager->checkErrorStatus() != exception::NOERR) {
          prepareAbortRender();
          return;
        }

        unsigned int idx = generateImageOffset(tile, nova_resource_manager->getEngineData().vertical_invert, x, y);

        const glm::vec2 ndc = math::camera::screen2ndc(x, tile.image_total_height - y, tile.image_total_width, tile.image_total_height);
        if (!nova_resource_manager->getEngineData().is_rendering)
          return;
        math::camera::camera_ray r = math::camera::ray_inv_mat(
            ndc.x, ndc.y, nova_resource_manager->getCameraData().inv_P, nova_resource_manager->getCameraData().inv_V);
        Ray ray(r.near, r.far);
        glm::vec4 rgb = Li(ray, nova_internals, nova_resource_manager->getEngineData().max_depth, sampler);
        accumulateRgbRenderbuffer(buffers, idx, rgb);
      }
    tile.finished_render = true;
  }

  glm::vec4 NormalIntegrator::Li(const Ray &ray, nova_eng_internals &nova_internals, int depth, sampler::SamplerInterface &sampler) const {

    bvh_hit_data hit = bvh_hit(ray, nova_internals);
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
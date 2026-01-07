#include "NormalIntegrator.h"
#include <internal/common/math/math_random.h>
#include <internal/memory/Allocator.h>

namespace nova::integrator {

  void NormalIntegrator::render(RenderBuffers<float> *buffers, Tile &tile, nova_eng_internals &nova_internals) const {
    const NovaResourceManager *nova_resource_manager = nova_internals.resource_manager;
    NovaExceptionManager *nova_exception_manager = nova_internals.exception_manager;
    math::random::SobolGenerator generator;
    sampler::SobolSampler sobol_sampler(generator);
    sampler::SamplerInterface sampler = &sobol_sampler;
    StackAllocator allocator;
    for (int y = tile.height_end - 1; y >= tile.height_start; y = y - 1)
      for (int x = tile.width_start; x < tile.width_end; x = x + 1) {
        allocator.reset();
        unsigned int idx = generateImageOffset(tile, nova_resource_manager->getEngineData().vertical_invert, x, y);
        sampler.reset(idx);
        validate(sampler, nova_internals);
        if (nova_exception_manager->checkErrorStatus() != exception::NOERR) {
          prepareAbortRender();
          return;
        }

        const glm::vec2 ndc = math::camera::screen2ndc(x, tile.image_total_height - y, tile.image_total_width, tile.image_total_height);
        if (!nova_resource_manager->getEngineData().is_rendering)
          return;
        math::camera::camera_ray r = math::camera::ray_inv_mat(
            ndc.x, ndc.y, nova_resource_manager->getCameraData().inv_P, nova_resource_manager->getCameraData().inv_V);
        Ray ray(r.near, r.far);
        glm::vec4 rgb = Li(ray, nova_internals, nova_resource_manager->getEngineData().max_depth, sampler, allocator);
        accumulateRgbRenderbuffer(buffers, idx, rgb);
      }
    tile.finished_render = true;
  }

  glm::vec4 NormalIntegrator::Li(
      const Ray &ray, nova_eng_internals &nova_internals, int depth, sampler::SamplerInterface &sampler, StackAllocator &allocator) const {
    bvh_hit_data hit = bvh_hit(ray, nova_internals);
    if (hit.is_hit) {
      Ray out{};
      material::shading_data_s shading{};
      texturing::TextureCtx texture_context = nova_internals.resource_manager->getTexturesData().getTextureBundleViews();
      texturing::texture_data_aggregate_s texture_data;
      texture_data.texture_ctx = &texture_context;
      shading.texture_aggregate = &texture_data;
      material_record_s mat_rec{};
      if (depth < 0)
        return glm::vec4{0.f};
      hit.last_primit->scatter(ray, out, hit.hit_d, mat_rec, sampler, allocator, shading);
      glm::vec4 normal = {mat_rec.normal, 1.f};
      return normal;
    }
    return glm::vec4(0.f);
  }
}  // namespace nova::integrator

#ifndef INTEGRATOR_H
#define INTEGRATOR_H
#include "manager/ManagerInternalStructs.h"
#include "math_camera.h"
#include "math_utils.h"
#include "project_macros.h"

namespace nova {
  class Ray;
  struct Tile;
  class NovaResourceManager;
  template<class T>
  struct RenderBuffers;
}  // namespace nova

namespace nova::integrator {

  enum TYPE : int {
    PATH = 1 << 0,
    BIPATH = 1 << 1,
    SPECTRAL = 1 << 2,
    METROPOLIS = 1 << 3,
    PHOTON = 1 << 4,
    MARCHING = 1 << 5,
    HYBRID = 1 << 6,
    VOXEL = 1 << 7,

    /* utility render */
    COMBINED = 1 << 8,
    NORMAL = 1 << 9,
    DEPTH = 1 << 10,
    SPECULAR = 1 << 11,
    DIFFUSE = 1 << 12,
    EMISSIVE = 1 << 13,
  };
  struct bvh_hit_data {
    bool is_hit{false};
    const primitive::NovaPrimitiveInterface *last_primit{nullptr};
    hit_data hit_d;
    float prim_min_t;
    float prim_max_t;
  };

  bvh_hit_data bvh_hit(const Ray &ray, nova_eng_internals &nova_internals);

  void integrator_dispatch(RenderBuffers<float> *buffers, Tile &tile, nova::nova_eng_internals &nova_internals);

  template<class T>
  class AbstractIntegrator {
   public:
    /* leave this here in case there's some memory pools to free */
    void prepareAbortRender() const {}

    void validate(const sampler::SamplerInterface &sampler, nova::nova_eng_internals &nova_internals) const {
      nova::exception::NovaException exception = nova::sampler::retrieve_sampler_error(sampler);
      nova_internals.exception_manager->addError(exception);
    }

    void render(RenderBuffers<float> *buffers, Tile &tile, nova::nova_eng_internals &nova_internals) const {
      constexpr float RAND_DX = 0.0005;
      constexpr float RAND_DY = 0.0005;
      const NovaResourceManager *nova_resource_manager = nova_internals.resource_manager;
      NovaExceptionManager *nova_exception_manager = nova_internals.exception_manager;
      sampler::SobolSampler sobol = sampler::SobolSampler(nova_resource_manager->getEngineData().getMaxSamples(), 20);
      sampler::SamplerInterface sampler = &sobol;
      for (int y = tile.height_end - 1; y >= tile.height_start; y = y - 1)
        for (int x = tile.width_start; x < tile.width_end; x = x + 1) {
          validate(sampler, nova_internals);
          if (nova_exception_manager->checkErrorStatus() != 0) {
            prepareAbortRender();
            return;
          }
          unsigned int idx = 0;
          if (!nova_resource_manager->getEngineData().isAxisVInverted())
            idx = (y * tile.image_total_width + x) * 4;
          else
            idx = ((tile.image_total_height - 1 - y) * tile.image_total_width + x) * 4;
          glm::vec4 rgb{};
          const glm::vec2 ndc = math::camera::screen2ndc(x, tile.image_total_height - y, tile.image_total_width, tile.image_total_height);
          for (int i = 0; i < tile.sample_per_tile; i++) {

            if (!nova_resource_manager->getEngineData().isRendering())
              return;
            /* Samples random direction around the pixel for AA. */
            const float dx = math::random::nrandf(-RAND_DX, RAND_DX);
            const float dy = math::random::nrandf(-RAND_DY, RAND_DY);
            math::camera::camera_ray r = math::camera::ray_inv_mat(ndc.x + dx,
                                                                   ndc.y + dy,
                                                                   nova_resource_manager->getCameraData().getInvProjection(),
                                                                   nova_resource_manager->getCameraData().getInvView());
            Ray ray(r.near, r.far);
            rgb += Li(ray, nova_internals, nova_resource_manager->getEngineData().getMaxDepth(), sampler);
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
    [[nodiscard]] glm::vec4 Li(const Ray &ray, nova_eng_internals &nova_internals, int depth, sampler::SamplerInterface &sampler) const {
      return static_cast<const T *>(this)->Li(ray, nova_internals, depth, sampler);
    }
  };

  class PathIntegrator : public AbstractIntegrator<PathIntegrator> {
   public:
    [[nodiscard]] glm::vec4 Li(const Ray &ray, nova_eng_internals &nova_internals, int depth, sampler::SamplerInterface &sampler) const;
  };

}  // namespace nova::integrator

#endif  // INTEGRATOR_H

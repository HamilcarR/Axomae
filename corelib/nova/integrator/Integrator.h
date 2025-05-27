#ifndef INTEGRATOR_H
#define INTEGRATOR_H
#include "internal/common/math/math_camera.h"
#include "internal/common/math/math_utils.h"
#include "internal/macro/project_macros.h"
#include "manager/ManagerInternalStructs.h"

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

  bvh_hit_data bvh_hit(const Ray &ray, nova_eng_internals &nova_internals);

  void integrator_dispatch(RenderBuffers<float> *buffers, Tile &tile, nova::nova_eng_internals &nova_internals);

  template<class T>
  class AbstractIntegrator {
   protected:
    void accumulateRgbRenderbuffer(RenderBuffers<float> *buffers, unsigned pixel_offset, glm::vec4 color) const {
      for (int k = 0; k < 3; k++)
        buffers->accumulator_buffer[pixel_offset + k] += buffers->partial_buffer[pixel_offset + k];
      buffers->partial_buffer[pixel_offset] = color.r;
      buffers->partial_buffer[pixel_offset + 1] = color.g;
      buffers->partial_buffer[pixel_offset + 2] = color.b;
      buffers->partial_buffer[pixel_offset + 3] = color.a;
    }

    ax_no_discard unsigned generateImageOffset(const Tile &tile, bool axis_inverted, int x, int y, unsigned channels = 4) const {
      if (axis_inverted)
        return ((tile.image_total_height - 1 - y) * tile.image_total_width + x) * channels;
      return (y * tile.image_total_width + x) * channels;
    }

   public:
    /* leave this here in case there's some memory pools to free */
    void prepareAbortRender() const { EMPTY_FUNCBODY }

    void validate(const sampler::SamplerInterface &sampler, nova::nova_eng_internals &nova_internals) const {
      nova::exception::NovaException exception = nova::sampler::retrieve_sampler_error(sampler);
      nova_internals.exception_manager->addError(exception);
    }

    void render(RenderBuffers<float> *buffers, Tile &tile, nova::nova_eng_internals &nova_internals) const {
      constexpr float RAND_DX = 0.0005;
      constexpr float RAND_DY = 0.0005;

      const NovaResourceManager *nova_resource_manager = nova_internals.resource_manager;
      NovaExceptionManager *nova_exception_manager = nova_internals.exception_manager;
      auto quasi_generator = math::random::CPUQuasiRandomGenerator(0xDEADBEEF, 5);
      auto pseudo_generator = math::random::CPUPseudoRandomGenerator(0xDEADBEEF);
      sampler::SobolSampler sobol_s = sampler::SobolSampler(quasi_generator);
      sampler::RandomSampler random_s = sampler::RandomSampler(pseudo_generator);
      sampler::SamplerInterface sampler = &sobol_s;

      for (int y = tile.height_end - 1; y >= tile.height_start; y = y - 1)
        for (int x = tile.width_start; x < tile.width_end; x = x + 1) {

          validate(sampler, nova_internals);
          if (nova_exception_manager->checkErrorStatus() != exception::NOERR) {
            prepareAbortRender();
            return;
          }

          unsigned int idx = generateImageOffset(tile, nova_resource_manager->getEngineData().vertical_invert, x, y);

          glm::vec4 rgb{};

          const glm::vec2 ndc = math::camera::screen2ndc(x, tile.image_total_height - y, tile.image_total_width, tile.image_total_height);
          for (int i = 0; i < tile.sample_per_tile; i++) {
            if (!nova_resource_manager->getEngineData().is_rendering)
              return;
            /* Samples random direction around the pixel for AA. */
            const glm::vec3 sampled_camera_directions = random_s.sample();
            const float dx = sampled_camera_directions.x * RAND_DX;
            const float dy = sampled_camera_directions.y * RAND_DY;
            math::camera::camera_ray r = math::camera::ray_inv_mat(
                ndc.x + dx, ndc.y + dy, nova_resource_manager->getCameraData().inv_P, nova_resource_manager->getCameraData().inv_V);
            Ray ray(r.near, r.far);
            rgb += Li(ray, nova_internals, nova_resource_manager->getEngineData().max_depth, sampler);
          }
          rgb /= (float)(tile.sample_per_tile);
          accumulateRgbRenderbuffer(buffers, idx, rgb);
        }
      tile.finished_render = true;
    }

    ax_no_discard glm::vec4 Li(const Ray &ray, nova_eng_internals &nova_internals, int depth, sampler::SamplerInterface &sampler) const {
      return static_cast<const T *>(this)->Li(ray, nova_internals, depth, sampler);
    }
  };

  class PathIntegrator : public AbstractIntegrator<PathIntegrator> {
   public:
    ax_no_discard glm::vec4 Li(const Ray &ray, nova_eng_internals &nova_internals, int depth, sampler::SamplerInterface &sampler) const;
  };

}  // namespace nova::integrator

#endif  // INTEGRATOR_H

#ifndef NORMALINTEGRATOR_H
#define NORMALINTEGRATOR_H
#include "Integrator.h"
namespace nova::integrator {
  class NormalIntegrator : public AbstractIntegrator<NormalIntegrator> {
   public:
    void render(RenderBuffers<float> *buffers, Tile &tile, nova_eng_internals &nova_internals) const;

    [[nodiscard]] glm::vec4 Li(const Ray &ray, nova_eng_internals &nova_internals, int depth, sampler::SamplerInterface &sampler) const;
  };
}  // namespace nova::integrator
#endif  // NORMALINTEGRATOR_H

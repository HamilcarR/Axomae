#ifndef EMISSIVEINTEGRATOR_H
#define EMISSIVEINTEGRATOR_H
#include "Integrator.h"

namespace nova::integrator {
  class EmissiveIntegrator : public AbstractIntegrator<EmissiveIntegrator> {
   public:
    void render(RenderBuffers<float> *buffers, Tile &tile, nova_eng_internals &nova_internals) const;
    ax_no_discard glm::vec4 Li(const Ray &ray, nova_eng_internals &nova_internals, int depth, sampler::SamplerInterface &sampler) const;
  };
}  // namespace nova::integrator
#endif  // EMISSIVEINTEGRATOR_H

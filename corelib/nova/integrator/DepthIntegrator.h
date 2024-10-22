#ifndef DEPTHINTEGRATOR_H
#define DEPTHINTEGRATOR_H

#include "Integrator.h"

namespace nova::integrator {
  class DepthIntegrator : public AbstractIntegrator<DepthIntegrator> {
   public:
    void render(RenderBuffers<float> *buffers, Tile &tile, nova_eng_internals &nova_internals) const;
    ax_no_discard glm::vec4 Li(const Ray &ray, nova_eng_internals &nova_internals, int depth, sampler::SamplerInterface &sampler) const;
  };
}  // namespace nova::integrator

#endif  // DEPTHINTEGRATOR_H

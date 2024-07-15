#ifndef DEPTHINTEGRATOR_H
#define DEPTHINTEGRATOR_H

#include "Integrator.h"

namespace nova::integrator {
  class DepthIntegrator : public AbstractIntegrator<DepthIntegrator> {
   public:
    void render(RenderBuffers<float> *buffers, Tile &tile, const NovaResourceManager *nova_resource_manager) const;
    [[nodiscard]] glm::vec4 Li(const Ray &ray, const NovaResourceManager *nova_resource_manager, int depth) const;
  };
}  // namespace nova::integrator

#endif  // DEPTHINTEGRATOR_H

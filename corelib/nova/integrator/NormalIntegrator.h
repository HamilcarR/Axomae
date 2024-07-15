#ifndef NORMALINTEGRATOR_H
#define NORMALINTEGRATOR_H
#include "Integrator.h"
namespace nova::integrator {
  class NormalIntegrator : public AbstractIntegrator<NormalIntegrator> {
   public:
    void render(RenderBuffers<float> *buffers, Tile &tile, const NovaResourceManager *nova_resource_manager) const;
    [[nodiscard]] glm::vec4 Li(const Ray &ray, const NovaResourceManager *nova_resources_manager, int depth) const;
  };
}  // namespace nova::integrator
#endif  // NORMALINTEGRATOR_H

#ifndef NOVAGEOPRIMITIVE_H
#define NOVAGEOPRIMITIVE_H
#include "BoundingBox.h"
#include "PrimitiveInterface.h"
#include "project_macros.h"
#include "ray/Hitable.h"
#include "ray/Ray.h"
#include "shape/ShapeInterface.h"

namespace material {
  class NovaMaterialInterface;
}

namespace nova::primitive {
  class NovaGeoPrimitive final : public NovaPrimitiveInterface {
   private:
    const material::NovaMaterialInterface *material{};
    const shape::NovaShapeInterface shape{};

   public:
    CLASS_OCM(NovaGeoPrimitive)

    NovaGeoPrimitive(const shape::NovaShapeInterface &shape, const material::NovaMaterialInterface *material);

    bool hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options *user_options) const override;
    bool scatter(const Ray &in, Ray &out, hit_data &data, sampler::SamplerInterface &sampler) const override;
    [[nodiscard]] glm::vec3 centroid() const override;

    [[nodiscard]] geometry::BoundingBox computeAABB() const override;
  };
}  // namespace nova::primitive
#endif  // NOVAGEOPRIMITIVE_H

#ifndef NOVAGEOPRIMITIVE_H
#define NOVAGEOPRIMITIVE_H
#include "BoundingBox.h"
#include "material/NovaMaterials.h"
#include "project_macros.h"
#include "ray/Hitable.h"
#include "ray/Ray.h"
#include "shape/ShapeInterface.h"

namespace nova::sampler {
  class SamplerInterface;
}

namespace nova::primitive {
  class NovaGeoPrimitive {
   private:
    const material::NovaMaterialInterface material{};
    const shape::NovaShapeInterface shape{};

   public:
    CLASS_CM(NovaGeoPrimitive)

    AX_DEVICE_CALLABLE NovaGeoPrimitive(const shape::NovaShapeInterface &shape, const material::NovaMaterialInterface &material);
    AX_DEVICE_CALLABLE bool hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options *user_options) const;
    AX_DEVICE_CALLABLE bool scatter(const Ray &in, Ray &out, hit_data &data, sampler::SamplerInterface &sampler) const;
    AX_DEVICE_CALLABLE [[nodiscard]] glm::vec3 centroid() const;
    AX_DEVICE_CALLABLE [[nodiscard]] geometry::BoundingBox computeAABB() const;
  };
}  // namespace nova::primitive
#endif  // NOVAGEOPRIMITIVE_H

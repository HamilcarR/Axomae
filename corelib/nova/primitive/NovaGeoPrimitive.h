#ifndef NOVAGEOPRIMITIVE_H
#define NOVAGEOPRIMITIVE_H
#include "material/NovaMaterials.h"
#include "ray/Hitable.h"
#include "ray/Ray.h"
#include "shape/MeshContext.h"
#include "shape/ShapeInterface.h"
#include <internal/geometry/BoundingBox.h>
#include <internal/macro/project_macros.h>

namespace nova::sampler {
  class SamplerInterface;
}

namespace nova::primitive {
  class NovaGeoPrimitive {
   private:
    material::NovaMaterialInterface material{};
    shape::NovaShapeInterface shape{};

   public:
    CLASS_DCM(NovaGeoPrimitive)

    ax_device_callable NovaGeoPrimitive(const shape::NovaShapeInterface &shape, const material::NovaMaterialInterface &material);
    ax_device_callable bool hit(const Ray &r, float tmin, float tmax, hit_data &data, const shape::MeshCtx &geometry) const;
    ax_device_callable bool scatter(
        const Ray &in, Ray &out, hit_data &data, sampler::SamplerInterface &sampler, material::shading_data_s &material_data) const;
    ax_device_callable ax_no_discard glm::vec3 centroid(const shape::MeshCtx &geometry) const;
    ax_device_callable ax_no_discard geometry::BoundingBox computeAABB(const shape::MeshCtx &geometry) const;
  };
}  // namespace nova::primitive
#endif  // NOVAGEOPRIMITIVE_H

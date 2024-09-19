#ifndef NOVAGEOPRIMITIVE_H
#define NOVAGEOPRIMITIVE_H
#include "material/NovaMaterials.h"
#include "ray/Hitable.h"
#include "ray/Ray.h"
#include "sampler/Sampler.h"
#include "shape/MeshContext.h"
#include "shape/ShapeInterface.h"
#include <internal/device/gpgpu/device_macros.h>
#include <internal/geometry/BoundingBox.h>
#include <internal/macro/project_macros.h>

namespace nova::primitive {

  class NovaGeoPrimitive {
   private:
    material::NovaMaterialInterface material{};
    shape::NovaShapeInterface shape{};

   public:
    CLASS_DCM(NovaGeoPrimitive)

    ax_device_callable NovaGeoPrimitive(const shape::NovaShapeInterface &shape_, const material::NovaMaterialInterface &material_)
        : material(material_), shape(shape_) {}

    ax_device_callable bool hit(const Ray &r, float tmin, float tmax, hit_data &data, const shape::MeshCtx &geometry) const {
      return shape.hit(r, tmin, tmax, data, geometry);
    }

    ax_device_callable bool scatter(
        const Ray &in, Ray &out, hit_data &data, sampler::SamplerInterface &sampler, material::shading_data_s &material_ctx) const {
      return material.scatter(in, out, data, sampler, material_ctx);
    }

    ax_device_callable glm::vec3 centroid(const shape::MeshCtx &geometry) const { return shape.centroid(geometry); }

    ax_device_callable geometry::BoundingBox computeAABB(const shape::MeshCtx &geometry) const { return shape.computeAABB(geometry); }

    ax_device_callable_inlined shape::face_data_s getFace(const shape::MeshCtx &geometry) const { return shape.getFace(geometry); }

    ax_device_callable_inlined shape::transform::transform4x4_t getTransform(const shape::MeshCtx &geometry) const {
      return shape.getTransform(geometry);
    }
  };
}  // namespace nova::primitive
#endif  // NOVAGEOPRIMITIVE_H

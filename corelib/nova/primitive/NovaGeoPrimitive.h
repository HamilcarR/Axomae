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
#include <internal/memory/Allocator.h>

namespace nova::primitive {

  class NovaGeoPrimitive {
   private:
    material::NovaMaterialInterface material{};
    shape::NovaShapeInterface shape{};

   public:
    CLASS_DCM(NovaGeoPrimitive)

    ax_device_callable_inlined NovaGeoPrimitive(const shape::NovaShapeInterface &shape_, const material::NovaMaterialInterface &material_)
        : material(material_), shape(shape_) {}

    ax_device_callable_inlined bool hit(const Ray &r, float tmin, float tmax, intersection_record_s &data, const shape::MeshCtx &geometry) const {
      return shape.hit(r, tmin, tmax, data, geometry);
    }

    ax_device_callable_inlined bool scatter(const Ray &in,
                                            Ray &out,
                                            const intersection_record_s &data,
                                            material_record_s &sampled_material,
                                            sampler::SamplerInterface &sampler,
                                            axstd::StaticAllocator64kb &allocator,
                                            material::shading_data_s &material_ctx) const {
      return material.scatter(in, out, data, sampled_material, sampler, allocator, material_ctx);
    }

    ax_device_callable_inlined glm::vec3 centroid(const shape::MeshCtx &geometry) const { return shape.centroid(geometry); }

    ax_device_callable_inlined geometry::BoundingBox computeAABB(const shape::MeshCtx &geometry) const { return shape.computeAABB(geometry); }

    ax_device_callable_inlined shape::face_data_s getFace(const shape::MeshCtx &geometry) const { return shape.getFace(geometry); }

    ax_device_callable_inlined transform4x4_t getTransform(const shape::MeshCtx &geometry) const { return shape.getTransform(geometry); }

    ax_device_callable_inlined const float *getTransformAddr(const shape::MeshCtx &geometry) const { return shape.getTransformAddr(geometry); }
  };
}  // namespace nova::primitive
#endif  // NOVAGEOPRIMITIVE_H

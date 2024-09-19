#ifndef PRIMITIVEINTERFACE_H
#define PRIMITIVEINTERFACE_H
#include "NovaGeoPrimitive.h"
#include "ray/Hitable.h"
#include "sampler/Sampler.h"
#include "shape/shape_datastructures.h"
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/memory/tag_ptr.h>

namespace nova::primitive {
  class NovaPrimitiveInterface : public core::tag_ptr<NovaGeoPrimitive> {
   public:
    using tag_ptr::tag_ptr;
    ax_device_callable_inlined bool hit(const Ray &r, float tmin, float tmax, hit_data &data, const shape::MeshCtx &geometry) const {
      auto disp = [&](auto prim) { return prim->hit(r, tmin, tmax, data, geometry); };
      return dispatch(disp);
    }

    ax_device_callable_inlined bool scatter(
        const Ray &in, Ray &out, hit_data &data, sampler::SamplerInterface &sampler, material::shading_data_s &mat_ctx) const {
      auto disp = [&](auto prim) { return prim->scatter(in, out, data, sampler, mat_ctx); };
      return dispatch(disp);
    }

    ax_device_callable_inlined glm::vec3 centroid(const shape::MeshCtx &geometry) const {
      auto disp = [&](auto prim) { return prim->centroid(geometry); };
      return dispatch(disp);
    }

    ax_device_callable_inlined geometry::BoundingBox computeAABB(const shape::MeshCtx &geometry) const {
      auto disp = [&](auto prim) { return prim->computeAABB(geometry); };
      return dispatch(disp);
    }

    ax_device_callable_inlined shape::face_data_s getFace(const shape::MeshCtx &geometry) const {
      auto disp = [&](auto prim) { return prim->getFace(geometry); };
      return dispatch(disp);
    }

    ax_device_callable_inlined shape::transform::transform4x4_t getTransform(const shape::MeshCtx &geometry) const {
      auto disp = [&](auto prim) { return prim->getTransform(geometry); };
      return dispatch(disp);
    }
  };

  using TYPELIST = NovaPrimitiveInterface::type_pack;

  using primitives_view_tn = axstd::span<NovaPrimitiveInterface>;
  using CstPrimitivesView = axstd::span<const NovaPrimitiveInterface>;

}  // namespace nova::primitive

#endif  // PRIMITIVEINTERFACE_H

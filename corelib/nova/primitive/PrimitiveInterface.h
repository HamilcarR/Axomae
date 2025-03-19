#ifndef PRIMITIVEINTERFACE_H
#define PRIMITIVEINTERFACE_H
#include "NovaGeoPrimitive.h"
#include "ray/Hitable.h"
#include "sampler/Sampler.h"
#include <internal/device/gpgpu/device_utils.h>
#include <internal/memory/tag_ptr.h>

namespace nova::primitive {
  class NovaPrimitiveInterface : public core::tag_ptr<NovaGeoPrimitive> {
   public:
    using tag_ptr::tag_ptr;
    ax_device_callable ax_no_discard bool hit(const Ray &r, float tmin, float tmax, hit_data &data, const shape::MeshCtx &geometry) const;
    ax_device_callable ax_no_discard bool scatter(
        const Ray &in, Ray &out, hit_data &data, sampler::SamplerInterface &sampler, material::shading_data_s &material_opts) const;
    ax_device_callable ax_no_discard glm::vec3 centroid(const shape::MeshCtx &geometry) const;
    ax_device_callable ax_no_discard geometry::BoundingBox computeAABB(const shape::MeshCtx &geometry) const;
  };

  using TYPELIST = NovaPrimitiveInterface::type_pack;
}  // namespace nova::primitive

using primitives_view_tn = axstd::span<nova::primitive::NovaPrimitiveInterface>;

#endif  // PRIMITIVEINTERFACE_H

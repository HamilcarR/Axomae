#ifndef PRIMITIVEINTERFACE_H
#define PRIMITIVEINTERFACE_H
#include "NovaGeoPrimitive.h"
#include "internal/device/gpgpu/device_utils.h"
#include "internal/memory/tag_ptr.h"
#include "ray/Hitable.h"
#include "sampler/Sampler.h"

namespace nova::primitive {
  class NovaPrimitiveInterface : public core::tag_ptr<NovaGeoPrimitive> {
   public:
    using tag_ptr::tag_ptr;
    ax_device_callable ax_no_discard bool hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options *user_options) const;
    ax_device_callable ax_no_discard bool scatter(const Ray &in, Ray &out, hit_data &data, sampler::SamplerInterface &sampler) const;
    ax_device_callable ax_no_discard glm::vec3 centroid() const;
    ax_device_callable ax_no_discard geometry::BoundingBox computeAABB() const;
  };

  using TYPELIST = NovaPrimitiveInterface::type_pack;
}  // namespace nova::primitive

#endif  // PRIMITIVEINTERFACE_H

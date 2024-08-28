
#ifndef PRIMITIVEINTERFACE_H
#define PRIMITIVEINTERFACE_H
#include "NovaGeoPrimitive.h"
#include "device_utils.h"
#include "ray/Hitable.h"
#include "sampler/Sampler.h"
#include "tag_ptr.h"

namespace nova::primitive {
  class NovaPrimitiveInterface : public core::tag_ptr<NovaGeoPrimitive> {
   public:
    using tag_ptr::tag_ptr;
    AX_DEVICE_CALLABLE [[nodiscard]] bool hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options *user_options) const;
    AX_DEVICE_CALLABLE [[nodiscard]] bool scatter(const Ray &in, Ray &out, hit_data &data, sampler::SamplerInterface &sampler) const;
    AX_DEVICE_CALLABLE [[nodiscard]] glm::vec3 centroid() const;
    [[nodiscard]] geometry::BoundingBox computeAABB() const;
  };

  using TYPELIST = NovaPrimitiveInterface::type_pack;
}  // namespace nova::primitive

#endif  // PRIMITIVEINTERFACE_H

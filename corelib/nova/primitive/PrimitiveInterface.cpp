#include "PrimitiveInterface.h"
#include "NovaGeoPrimitive.h"

using namespace nova::primitive;

bool NovaPrimitiveInterface::hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options *user_options) const {
  auto disp = [&](auto prim) { return prim->hit(r, tmin, tmax, data, user_options); };
  return dispatch(disp);
}

bool NovaPrimitiveInterface::scatter(const Ray &in, Ray &out, hit_data &data, sampler::SamplerInterface &sampler) const {
  auto disp = [&](auto prim) { return prim->scatter(in, out, data, sampler); };
  return dispatch(disp);
}

glm::vec3 NovaPrimitiveInterface::centroid() const {
  auto disp = [&](auto prim) { return prim->centroid(); };
  return dispatch(disp);
}

geometry::BoundingBox NovaPrimitiveInterface::computeAABB() const {
  auto disp = [&](auto prim) { return prim->computeAABB(); };
  return dispatch(disp);
}

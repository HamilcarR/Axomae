#include "PrimitiveInterface.h"
#include "NovaGeoPrimitive.h"

using namespace nova::primitive;

bool NovaPrimitiveInterface::hit(const Ray &r, float tmin, float tmax, hit_data &data, const shape::MeshCtx &geometry) const {
  auto disp = [&](auto prim) { return prim->hit(r, tmin, tmax, data, geometry); };
  return dispatch(disp);
}

bool NovaPrimitiveInterface::scatter(
    const Ray &in, Ray &out, hit_data &data, sampler::SamplerInterface &sampler, material::shading_data_s &mat_ctx) const {
  auto disp = [&](auto prim) { return prim->scatter(in, out, data, sampler, mat_ctx); };
  return dispatch(disp);
}

glm::vec3 NovaPrimitiveInterface::centroid(const shape::MeshCtx &geometry) const {
  auto disp = [&](auto prim) { return prim->centroid(geometry); };
  return dispatch(disp);
}

geometry::BoundingBox NovaPrimitiveInterface::computeAABB(const shape::MeshCtx &geometry) const {
  auto disp = [&](auto prim) { return prim->computeAABB(geometry); };
  return dispatch(disp);
}

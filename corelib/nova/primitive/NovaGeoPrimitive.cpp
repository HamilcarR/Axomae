#include "NovaGeoPrimitive.h"
#include "ray/Ray.h"
#include "sampler/Sampler.h"
#include "shape/MeshContext.h"
#include "shape/nova_shape.h"

namespace nova::primitive {
  NovaGeoPrimitive::NovaGeoPrimitive(const shape::NovaShapeInterface &shape_, const material::NovaMaterialInterface &material_)
      : material(material_), shape(shape_) {}

  bool NovaGeoPrimitive::hit(const Ray &r, float tmin, float tmax, hit_data &data, const shape::MeshCtx &geometry) const {
    return shape.hit(r, tmin, tmax, data, geometry);
  }

  bool NovaGeoPrimitive::scatter(const Ray &in, Ray &out, hit_data &data, sampler::SamplerInterface &sampler) const {
    return material.scatter(in, out, data, sampler);
  }

  glm::vec3 NovaGeoPrimitive::centroid(const shape::MeshCtx &geometry) const { return shape.centroid(geometry); }
  geometry::BoundingBox NovaGeoPrimitive::computeAABB(const shape::MeshCtx &geometry) const { return shape.computeAABB(geometry); }
}  // namespace nova::primitive

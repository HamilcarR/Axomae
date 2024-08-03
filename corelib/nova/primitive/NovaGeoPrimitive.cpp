#include "NovaGeoPrimitive.h"
#include "material/nova_material.h"
#include "nova_primitive.h"
#include "ray/Ray.h"
#include "sampler/Sampler.h"
#include "shape/nova_shape.h"

namespace nova::primitive {
  NovaGeoPrimitive::NovaGeoPrimitive(const shape::NovaShapeInterface &shape_, const material::NovaMaterialInterface &material_)
      : material(material_), shape(shape_) {}

  bool NovaGeoPrimitive::hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options *user_options) const {
    if (!shape.hit(r, tmin, tmax, data, user_options))
      return false;
    data.normal = glm::normalize(data.normal);
    data.position = r.pointAt(data.t);
    return true;
  }

  bool NovaGeoPrimitive::scatter(const Ray &in, Ray &out, hit_data &data, sampler::SamplerInterface &sampler) const {
    AX_ASSERT(material, "Material structure is not initialized.");
    return material.scatter(in, out, data, sampler);
  }

  glm::vec3 NovaGeoPrimitive::centroid() const { return shape.centroid(); }
  geometry::BoundingBox NovaGeoPrimitive::computeAABB() const { return shape.computeAABB(); }
}  // namespace nova::primitive
#include "NovaGeoPrimitive.h"
#include "Ray.h"
#include "nova_material.h"
#include "nova_primitive.h"
#include "nova_shape.h"
using namespace nova::primitive;

NovaGeoPrimitive::NovaGeoPrimitive(std::unique_ptr<shape::NovaShapeInterface> &shape_, std::unique_ptr<material::NovaMaterialInterface> &material_) {
  shape = std::move(shape_);
  material = std::move(material_);
}
NovaGeoPrimitive::~NovaGeoPrimitive() = default;

bool NovaGeoPrimitive::hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options *user_options) const {
  if (!shape->hit(r, tmin, tmax, data, user_options))
    return false;
  data.normal = glm::normalize(data.normal);
  data.position = r.pointAt(data.t);
  return true;
}

bool NovaGeoPrimitive::scatter(const Ray &in, Ray &out, hit_data &data) const {
  AX_ASSERT(material, "Material structure is not initialized.");
  return material->scatter(in, out, data);
}

glm::vec3 NovaGeoPrimitive::centroid() const { return shape->centroid(); }
geometry::BoundingBox NovaGeoPrimitive::computeAABB() const { return shape->computeAABB(); }
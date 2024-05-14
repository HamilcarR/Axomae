#include "NovaGeoPrimitive.h"
#include "Ray.h"
#include "nova_material.h"
#include "nova_primitive.h"
#include "nova_shape.h"
using namespace nova::primitive;

NovaGeoPrimitive::NovaGeoPrimitive(shape::NovaShapeInterface *s, material::NovaMaterialInterface *m) : material(m), shape(s) {}

bool NovaGeoPrimitive::hit(const Ray &r, float tmin, float tmax, hit_data &data, const base_options *user_options) const {
  glm::vec3 normal{};
  float t{};
  if (!shape->intersect(r, tmin, tmax, normal, t))
    return false;
  data.normal = glm::normalize(normal);
  data.position = r.pointAt(t);
  data.t = t;
  return true;
}

bool NovaGeoPrimitive::scatter(const Ray &in, Ray &out, hit_data &data) const {
  AX_ASSERT(material, "Material structure is not initialized.");
  return material->scatter(in, out, data);
}

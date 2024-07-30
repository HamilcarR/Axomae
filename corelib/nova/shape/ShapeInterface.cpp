#include "ShapeInterface.h"
#include "Box.h"
#include "Sphere.h"
#include "Square.h"
#include "Triangle.h"

using namespace nova::shape;

AX_DEVICE_CALLABLE glm::vec3 NovaShapeInterface::centroid() const {
  auto d = [&](auto shape) { return shape->centroid(); };
  return dispatch(d);
}
AX_DEVICE_CALLABLE geometry::BoundingBox NovaShapeInterface::computeAABB() const {
  auto d = [&](auto shape) { return shape->computeAABB(); };
  return dispatch(d);
}
AX_DEVICE_CALLABLE bool NovaShapeInterface::hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options *user_options) const {
  auto d = [&](auto shape) { return shape->hit(r, tmin, tmax, data, user_options); };
  return dispatch(d);
}
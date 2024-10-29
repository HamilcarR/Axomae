#include "ShapeInterface.h"
#include "Box.h"
#include "Sphere.h"
#include "Square.h"
#include "Triangle.h"

using namespace nova::shape;

ax_device_callable glm::vec3 NovaShapeInterface::centroid() const {
  auto d = [&](auto shape) { return shape->centroid(); };
  return dispatch(d);
}
ax_device_callable geometry::BoundingBox NovaShapeInterface::computeAABB() const {
  auto d = [&](auto shape) { return shape->computeAABB(); };
  return dispatch(d);
}
ax_device_callable bool NovaShapeInterface::hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options *user_options) const {
  auto d = [&](auto shape) { return shape->hit(r, tmin, tmax, data, user_options); };
  return dispatch(d);
}
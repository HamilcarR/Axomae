#include "ShapeInterface.h"

using namespace nova::shape;

ax_device_callable glm::vec3 NovaShapeInterface::centroid(const MeshCtx &geometry) const {
  auto d = [&](auto shape) { return shape->centroid(geometry); };
  return dispatch(d);
}
ax_device_callable geometry::BoundingBox NovaShapeInterface::computeAABB(const MeshCtx &geometry) const {
  auto d = [&](auto shape) { return shape->computeAABB(geometry); };
  return dispatch(d);
}
ax_device_callable float NovaShapeInterface::area(const MeshCtx &geometry) const {
  auto d = [&](auto shape) { return shape->area(geometry); };
  return dispatch(d);
}

ax_device_callable bool NovaShapeInterface::hit(const Ray &r, float tmin, float tmax, hit_data &data, const MeshCtx &geometry) const {
  auto d = [&](auto shape) { return shape->hit(r, tmin, tmax, data, geometry); };
  return dispatch(d);
}

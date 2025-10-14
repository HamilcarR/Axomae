#include "Sphere.h"
#include "MeshContext.h"
#include "ray/IntersectFrame.h"
#include "ray/Ray.h"

namespace nova::shape {
  ax_device_callable Sphere::Sphere(const glm::vec3 &o, float r) : origin(o), radius(r) {}

  ax_device_callable bool Sphere::hit(const Ray &r, float tmin, float tmax, hit_data &data, const MeshCtx & /*geometry*/) const {
    const glm::vec3 oc = r.origin - origin;
    const float b = 2.f * glm::dot(r.direction, oc);
    const float a = glm::dot(r.direction, r.direction);
    const float c = glm::dot(oc, oc) - radius * radius;
    const float determinant = b * b - 4 * a * c;
    glm::vec3 normal;
    if (determinant > 0) {
      float t1 = (-b - std::sqrt(determinant)) * 0.5f / a;
      if (t1 < tmax && t1 > tmin) {
        data.t = t1;
        normal = r.pointAt(data.t) - origin;
        data.shading_frame = IntersectFrame({}, {}, normal);
        return true;
      }
      t1 = (-b + std::sqrt(determinant)) * 0.5f / a;
      if (t1 < tmax && t1 > tmin) {
        data.t = t1;
        normal = r.pointAt(data.t) - origin;
        data.shading_frame = IntersectFrame({}, {}, normal);
        return true;
      }
    }

    return false;
  }

  ax_device_callable geometry::BoundingBox Sphere::computeAABB(const MeshCtx & /*geometry*/) const {
    return {origin - glm::vec3(radius), origin + glm::vec3(radius)};
  }
}  // namespace nova::shape

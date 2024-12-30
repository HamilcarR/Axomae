#include "Sphere.h"

#include "ray/Ray.h"

namespace nova::shape {
  ax_device_callable Sphere::Sphere(const glm::vec3 &o, float r) : origin(o), radius(r) {}

  ax_device_callable bool Sphere::hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options * /*user_options*/) const {
    const glm::vec3 oc = r.origin - origin;
    const float b = 2.f * glm::dot(r.direction, oc);
    const float a = glm::dot(r.direction, r.direction);
    const float c = glm::dot(oc, oc) - radius * radius;
    const float determinant = b * b - 4 * a * c;
    if (determinant > 0) {
      float t1 = (-b - std::sqrt(determinant)) * 0.5f / a;
      if (t1 < tmax && t1 > tmin) {
        data.t = t1;
        data.normal = r.pointAt(data.t) - origin;
        return true;
      }
      t1 = (-b + std::sqrt(determinant)) * 0.5f / a;
      if (t1 < tmax && t1 > tmin) {
        data.t = t1;
        data.normal = r.pointAt(data.t) - origin;
        return true;
      }
    }
    return false;
  }

  ax_device_callable geometry::BoundingBox Sphere::computeAABB() const { return {origin - glm::vec3(radius), origin + glm::vec3(radius)}; }
}  // namespace nova::shape
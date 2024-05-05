#include "Sphere.h"

#include "ray/Ray.h"

using namespace nova;
Sphere::Sphere(const glm::vec3 &o, float r) : origin(o), radius(r) {}

bool Sphere::hit(const Ray &r, float tmin, float tmax, hit_data &data, const base_options *user_options) const {
  const glm::vec3 oc = r.origin - origin;
  const float b = 2.f * glm::dot(r.direction, oc);
  const float a = glm::dot(r.direction, r.direction);
  const float c = glm::dot(oc, oc) - radius * radius;
  const float determinant = b * b - 4 * a * c;
  if (determinant >= 0) {
    float t1 = (-b - std::sqrt(determinant)) * 0.5f * a;
    if (t1 <= 0)
      return false;
    data.t = t1;
    data.position = r.pointAt(t1);
    data.normal = data.position - origin;
    return true;
  }
  return false;
}

#include "Sphere.h"

#include "ray/Ray.h"

using namespace nova::shape;

Sphere::Sphere(const glm::vec3 &o, float r) : origin(o), radius(r) {}

bool Sphere::intersect(const Ray &r, float tmin, float tmax, glm::vec3 &normal, float &t) const {
  const glm::vec3 oc = r.origin - origin;
  const float b = 2.f * glm::dot(r.direction, oc);
  const float a = glm::dot(r.direction, r.direction);
  const float c = glm::dot(oc, oc) - radius * radius;
  const float determinant = b * b - 4 * a * c;
  if (determinant > 0) {
    float t1 = (-b - std::sqrt(determinant)) * 0.5f / a;
    if (t1 < tmax && t1 > tmin) {
      t = t1;
      normal = r.pointAt(t) - origin;
      return true;
    }
    t1 = (-b + std::sqrt(determinant)) * 0.5f / a;
    if (t1 < tmax && t1 > tmin) {
      t = t1;
      normal = r.pointAt(t) - origin;
      return true;
    }
  }
  return false;
}

geometry::BoundingBox Sphere::computeAABB() const { return {origin - glm::vec3(radius), origin + glm::vec3(radius)}; }

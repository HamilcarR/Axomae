#include "Triangle.h"

#include "Ray.h"

using namespace nova::shape;

Triangle::Triangle(const glm::vec3 &v0_, const glm::vec3 &v1_, const glm::vec3 &v2_) : v0(v0_), v1(v1_), v2(v2_) {
  e1 = v1 - v0;
  e2 = v2 - v0;
  center = (v0 + v1 + v2) * 0.3333f;
}

/*
 * https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
 */
bool Triangle::intersect(const Ray &ray, float tmin, float tmax, glm::vec3 &normal_at_intersection, float &t) const {
  glm::vec3 P = glm::cross(ray.direction, e2);
  const float det = glm::dot(P, e1);
  const float inv_det = 1.f / det;

  /* backface cull */
  if (det < math::epsilon && det > -math::epsilon)
    return false;

  glm::vec3 T = ray.origin - v0;
  const float u = glm::dot(P, T) * inv_det;
  if (u < 0 || u > 1.f)
    return false;

  glm::vec3 Q = glm::cross(T, e1);
  const float v = glm::dot(Q, ray.direction) * inv_det;
  if (v < 0.f || (u + v) > 1.f)
    return false;

  t = glm::dot(Q, e2) * inv_det;
  if (t < tmin || t > tmax)
    return false;
  normal_at_intersection = glm::cross(e1, e2);
  if (glm::dot(normal_at_intersection, -ray.direction) < 0)
    normal_at_intersection = -normal_at_intersection;
  normal_at_intersection = glm::normalize(normal_at_intersection);

  return true;
}
geometry::BoundingBox Triangle::computeAABB() const {
  glm::vec3 min = glm::min(v0, glm::min(v1, v2));
  glm::vec3 max = glm::max(v0, glm::max(v1, v2));
  return {min, max};
}

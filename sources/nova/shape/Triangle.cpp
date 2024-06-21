#include "Triangle.h"
#include "Ray.h"
#include "math_utils.h"

using namespace nova::shape;

Triangle::Triangle(const glm::vec3 &v0_, const glm::vec3 &v1_, const glm::vec3 &v2_) : v0(v0_), v1(v1_), v2(v2_) {
  e1 = v1 - v0;
  e2 = v2 - v0;
  center = (v0 + v1 + v2) * 0.3333f;
}
Triangle::Triangle(const glm::vec3 vertices[3], const glm::vec3 normals[3]) : Triangle(vertices[0], vertices[1], vertices[2]) {
  n0 = normals[0];
  n1 = normals[1];
  n2 = normals[2];
}

/*
 * https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
 */
bool Triangle::hit(const Ray &ray, float tmin, float tmax, hit_data &data, base_options *user_options) const {
  glm::vec3 P = glm::cross(ray.direction, e2);
  const float det = glm::dot(P, e1);
  /* backface cull */
  // if (det < epsilon && det > -epsilon)
  //  return false;

  const float inv_det = 1.f / det;
  glm::vec3 T = ray.origin - v0;
  const float u = glm::dot(P, T) * inv_det;
  if (u < 0 || u > 1.f)
    return false;

  glm::vec3 Q = glm::cross(T, e1);
  const float v = glm::dot(Q, ray.direction) * inv_det;
  if (v < 0.f || (u + v) > 1.f)
    return false;
  data.t = glm::dot(Q, e2) * inv_det;
  /* Early return in case this triangle is farther than the last intersected shape. */
  if (data.t < tmin || data.t > tmax)
    return false;
  if (!hasValidNormals()) {
    data.normal = glm::cross(e1, e2);
    if (glm::dot(data.normal, -ray.direction) < 0)
      data.normal = -data.normal;
    data.normal = glm::normalize(data.normal);
  } else {

    /* Returns barycentric interpolated normal at intersection t.  */
    const float w = 1 - (u + v);
    data.normal = glm::normalize(n0 * u + n1 * v + n2 * w);
  }
  return true;
}

geometry::BoundingBox Triangle::computeAABB() const {
  glm::vec3 min = glm::min(v0, glm::min(v1, v2));
  glm::vec3 max = glm::max(v0, glm::max(v1, v2));
  return {min, max};
}

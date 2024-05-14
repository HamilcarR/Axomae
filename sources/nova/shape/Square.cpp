#include "Square.h"
#include "Ray.h"

using namespace nova::shape;
bool Square::intersect(const Ray &ray, float tmin, float tmax, glm::vec3 &normal_r, float &t) const {
  glm::vec3 n = normal;
  if (glm::dot(ray.origin - origin, n) < 0)
    n = -n;

  float NdotTo_O = glm::dot(n, ray.origin - origin);
  float NdotD = glm::dot(n, -ray.direction);
  normal_r = n;
  if (NdotD <= math::epsilon)
    return false;
  t = NdotTo_O / NdotD;
  if (t < tmin || t > tmax)
    return false;
  glm::vec3 p = ray.pointAt(t);
  glm::vec3 w_abs = glm::abs(side_w);
  glm::vec3 h_abs = glm::abs(side_h);
  float SdotP_w = glm::dot(p - origin, w_abs);
  float SdotP_h = glm::dot(p - origin, h_abs);
  if (SdotP_w <= glm::length2(w_abs) && SdotP_w > 0 && SdotP_h <= glm::length2(h_abs) && SdotP_h > 0)
    return true;
  return false;
}

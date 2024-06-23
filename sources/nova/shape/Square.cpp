#include "Square.h"
#include "ray/Ray.h"

using namespace nova::shape;
bool Square::hit(const Ray &ray, float tmin, float tmax, hit_data &data, base_options *user_options) const {
  glm::vec3 n = normal;
  if (glm::dot(ray.origin - origin, n) < 0)
    n = -n;

  float NdotTo_O = glm::dot(n, ray.origin - origin);
  float NdotD = glm::dot(n, -ray.direction);
  data.normal = n;
  if (NdotD <= math::epsilon)
    return false;
  data.t = NdotTo_O / NdotD;
  if (data.t < tmin || data.t > tmax)
    return false;
  glm::vec3 p = ray.pointAt(data.t);
  glm::vec3 w_abs = glm::abs(side_w);
  glm::vec3 h_abs = glm::abs(side_h);
  float SdotP_w = glm::dot(p - origin, w_abs);
  float SdotP_h = glm::dot(p - origin, h_abs);
  if (SdotP_w <= glm::length2(w_abs) && SdotP_w > 0 && SdotP_h <= glm::length2(h_abs) && SdotP_h > 0)
    return true;
  return false;
}
geometry::BoundingBox Square::computeAABB() const { return {origin, origin + side_w + side_h}; }

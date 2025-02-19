
#include "Box.h"
#include "MeshContext.h"
#include "ray/Ray.h"
namespace nova::shape {
  Box::Box(const glm::vec3 &min_coords, const glm::vec3 &max_coords) : aabb(min_coords, max_coords) {}
  Box::Box(const float *vertices, std::size_t size) : aabb(vertices, size) {}
  Box::Box(const std::vector<float> &vertices) : aabb(vertices) {}
  Box::Box(const geometry::BoundingBox &aabb_) : aabb(aabb_) {}

  static bool test_intersection(const glm::vec3 &x_axis,
                                const glm::vec3 &y_axis,
                                const glm::vec3 &z_axis,
                                const glm::vec3 &ray_origin,
                                const glm::vec3 &ray_direction,
                                const glm::vec3 &tmin_coords,
                                const glm::vec3 &tmax_coords,
                                float /*tmin*/,
                                float tmax,
                                float &t

  ) {
    const glm::vec3 delta = ray_origin;
    /* Intersections on X axis */
    const float delta_x = glm::dot(delta, x_axis);
    const float Dx = glm::dot(ray_direction, x_axis) + math::epsilon;
    float txmin = (tmin_coords.x - delta_x) / Dx;
    float txmax = (tmax_coords.x - delta_x) / Dx;
    if (txmin > txmax) {
      std::swap(txmin, txmax);
    }

    /* Intersections on Y axis */
    const float delta_y = glm::dot(delta, y_axis);
    const float Dy = glm::dot(ray_direction, y_axis) + math::epsilon;
    float tymin = (tmin_coords.y - delta_y) / Dy;
    float tymax = (tmax_coords.y - delta_y) / Dy;
    if (tymin > tymax) {
      std::swap(tymin, tymax);
    }

    /* Intersections on Z axis */
    const float delta_z = glm::dot(delta, z_axis);
    const float Dz = glm::dot(ray_direction, z_axis) + math::epsilon;
    float tzmin = (tmin_coords.z - delta_z) / Dz;
    float tzmax = (tmax_coords.z - delta_z) / Dz;
    if (tzmin > tzmax) {
      std::swap(tzmin, tzmax);
    }

    const float max = std::min(std::min(txmax, tymax), tzmax);
    const float min = std::max(std::max(txmin, tymin), tzmin);
    float ret = min;

    if (max < min)
      return false;
    if (max > 0) {
      if (ret > tmax)
        return false;
      t = ret;
      return true;
    }
    if (min < 0)
      return false;
    t = ret;
    return true;
  }

  bool Box::hit(const Ray &ray, float tmin, float tmax, hit_data &data, const MeshCtx & /*geometry*/) const {
    const glm::vec3 x_axis = glm::vec3(1, 0, 0);
    const glm::vec3 y_axis = glm::vec3(0, 1, 0);
    const glm::vec3 z_axis = glm::vec3(0, 0, 1);
    const glm::vec3 tmin_coords = aabb.getMinCoords();
    const glm::vec3 tmax_coords = aabb.getMaxCoords();
    bool hit_success = test_intersection(x_axis, y_axis, z_axis, ray.origin, ray.direction, tmin_coords, tmax_coords, tmin, tmax, data.t);
    if (hit_success) {
      return true;
    }
    return false;
  }
}  // namespace nova::shape

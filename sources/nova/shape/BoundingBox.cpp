#include "BoundingBox.h"
#include "ray/Ray.h"
#include "scene/Hitable.h"

using namespace nova::shape;
glm::vec3 calculateCenter(glm::vec3 min_coords, glm::vec3 max_coords);

glm::vec3 &update_min(glm::vec3 &min_vec, glm::vec3 &compared) {
  if (compared.x <= min_vec.x)
    min_vec.x = compared.x;
  if (compared.y <= min_vec.y)
    min_vec.y = compared.y;
  if (compared.z <= min_vec.z)
    min_vec.z = compared.z;
  return min_vec;
}

glm::vec3 &update_max(glm::vec3 &max_vec, glm::vec3 &compared) {
  if (compared.x > max_vec.x)
    max_vec.x = compared.x;
  if (compared.y > max_vec.y)
    max_vec.y = compared.y;
  if (compared.z > max_vec.z)
    max_vec.z = compared.z;
  return max_vec;
}

BoundingBox::BoundingBox() {
  center = glm::vec3(0.f);
  max_coords = glm::vec3(0.f);
  min_coords = glm::vec3(0.f);
}

BoundingBox::BoundingBox(const BoundingBox &copy) {
  max_coords = copy.max_coords;
  min_coords = copy.min_coords;
  center = copy.center;
}
BoundingBox::BoundingBox(BoundingBox &&move) noexcept {
  max_coords = move.max_coords;
  min_coords = move.min_coords;
  center = move.center;
}
BoundingBox &BoundingBox::operator=(const BoundingBox &copy) {
  if (&copy != this) {
    max_coords = copy.max_coords;
    min_coords = copy.min_coords;
    center = copy.center;
  }
  return *this;
}
BoundingBox &BoundingBox::operator=(BoundingBox &&move) noexcept {
  if (&move != this) {
    max_coords = move.max_coords;
    min_coords = move.min_coords;
    center = move.center;
  }
  return *this;
}

BoundingBox::BoundingBox(const glm::vec3 &_min_coords, const glm::vec3 &_max_coords) {
  center = calculateCenter(_min_coords, _max_coords);
  min_coords = _min_coords;
  max_coords = _max_coords;
}

BoundingBox::BoundingBox(const std::vector<float> &vertices) : BoundingBox() {
  float x_max, y_max, z_max;
  float x_min, y_min, z_min;
  x_max = y_max = z_max = -INT_MAX;
  x_min = y_min = z_min = INT_MAX;
  for (unsigned i = 0; i < vertices.size(); i += 3) {
    float x_compare = vertices[i], y_compare = vertices[i + 1], z_compare = vertices[i + 2];
    x_max = x_max <= x_compare ? x_compare : x_max;
    y_max = y_max <= y_compare ? y_compare : y_max;
    z_max = z_max <= z_compare ? z_compare : z_max;
    x_min = x_compare <= x_min ? x_compare : x_min;
    y_min = y_compare <= y_min ? y_compare : y_min;
    z_min = z_compare <= z_min ? z_compare : z_min;
  }
  min_coords = {x_min, y_min, z_min};
  max_coords = {x_max, y_max, z_max};
  center = calculateCenter(min_coords, max_coords);
}

std::pair<std::vector<float>, std::vector<unsigned>> BoundingBox::getVertexArray() const {
  std::vector<float> vertices = {
      min_coords.x, min_coords.y, max_coords.z,  // 0
      max_coords.x, min_coords.y, max_coords.z,  // 1
      max_coords.x, max_coords.y, max_coords.z,  // 2
      min_coords.x, max_coords.y, max_coords.z,  // 3
      max_coords.x, min_coords.y, min_coords.z,  // 4
      max_coords.x, max_coords.y, min_coords.z,  // 5
      min_coords.x, max_coords.y, min_coords.z,  // 6
      min_coords.x, min_coords.y, min_coords.z   // 7
  };
  std::vector<unsigned> indices = {0, 1, 2, 0, 2, 3, 1, 4, 5, 1, 5, 2, 7, 6, 5, 7, 5, 4, 3, 6, 7, 3, 7, 0, 2, 5, 6, 2, 6, 3, 7, 4, 1, 7, 1, 0};
  return {vertices, indices};
}

glm::vec3 calculateCenter(glm::vec3 min_coords, glm::vec3 max_coords) {
  glm::vec3 center{0.f};
  center.x = (max_coords.x + min_coords.x) / 2;
  center.y = (max_coords.y + min_coords.y) / 2;
  center.z = (max_coords.z + min_coords.z) / 2;
  return center;
}

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
  t = min;

  if (max < min)
    return false;
  if (max > 0) {
    if (t > tmax)
      return false;
    return true;
  }
  if (min < 0)
    return false;
  return true;
}

bool BoundingBox::intersect(const Ray &ray, float tmin, float tmax, glm::vec3 &normal_at_intersection, float &t) const {
  const glm::vec3 x_axis = glm::vec3(1, 0, 0);
  const glm::vec3 y_axis = glm::vec3(0, 1, 0);
  const glm::vec3 z_axis = glm::vec3(0, 0, 1);
  const glm::vec3 tmin_coords = min_coords;
  const glm::vec3 tmax_coords = max_coords;
  const glm::vec3 delta = ray.origin;
  bool hit_success = test_intersection(x_axis, y_axis, z_axis, ray.origin, ray.direction, tmin_coords, tmax_coords, tmin, tmax, t);
  if (hit_success) {
    return true;
  }
  return false;
}
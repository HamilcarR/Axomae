#include "BoundingBox.h"
#include "internal/macro/project_macros.h"
using namespace geometry;

ax_device_callable glm::vec3 calculateCenter(glm::vec3 min_coords, glm::vec3 max_coords) {
  glm::vec3 center{0.f};
  center.x = (max_coords.x + min_coords.x) * 0.5f;
  center.y = (max_coords.y + min_coords.y) * 0.5f;
  center.z = (max_coords.z + min_coords.z) * 0.5f;
  return center;
}

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

BoundingBox::BoundingBox(const float min_coords_[3], const float max_coords_[3]) {
  AX_ASSERT_NOTNULL(min_coords_);
  AX_ASSERT_NOTNULL(max_coords_);
  min_coords.x = min_coords_[0];
  min_coords.y = min_coords_[1];
  min_coords.z = min_coords_[2];

  max_coords.x = max_coords_[0];
  max_coords.y = max_coords_[1];
  max_coords.z = max_coords_[2];

  center = calculateCenter(min_coords, max_coords);
}

ax_device_callable BoundingBox::BoundingBox(const glm::vec3 &_min_coords, const glm::vec3 &_max_coords) {
  center = calculateCenter(_min_coords, _max_coords);
  min_coords = _min_coords;
  max_coords = _max_coords;
}
BoundingBox::BoundingBox(const std::vector<glm::vec3> &vertices) {
  glm::vec3 min{INT_MAX, INT_MAX, INT_MAX};
  glm::vec3 max{INT_MIN, INT_MIN, INT_MIN};
  for (const auto &elem : vertices) {
    min = glm::min(elem, min);
    max = glm::max(elem, max);
  }
  min_coords = min;
  max_coords = max;
  center = calculateCenter(min_coords, max_coords);
}

BoundingBox::BoundingBox(const std::vector<BoundingBox> &bboxes) {
  glm::vec3 min{INT_MAX, INT_MAX, INT_MAX};
  glm::vec3 max{INT_MIN, INT_MIN, INT_MIN};
  for (const auto &elem : bboxes) {
    min = glm::min(min, elem.min_coords);
    max = glm::max(max, elem.max_coords);
  }
  min_coords = min;
  max_coords = max;
  center = calculateCenter(min_coords, max_coords);
}

ax_device_callable static std::pair<glm::vec3, glm::vec3> compute_min_max(const float *vertices, std::size_t size) {
  float x_max{}, y_max{}, z_max{};
  float x_min{}, y_min{}, z_min{};
  x_max = y_max = z_max = INT_MIN;
  x_min = y_min = z_min = INT_MAX;
  for (unsigned i = 0; i < size; i += 3) {
    float x_compare = vertices[i], y_compare = vertices[i + 1], z_compare = vertices[i + 2];
    x_max = x_max <= x_compare ? x_compare : x_max;
    y_max = y_max <= y_compare ? y_compare : y_max;
    z_max = z_max <= z_compare ? z_compare : z_max;
    x_min = x_compare <= x_min ? x_compare : x_min;
    y_min = y_compare <= y_min ? y_compare : y_min;
    z_min = z_compare <= z_min ? z_compare : z_min;
  }
  glm::vec3 min{x_min, y_min, z_min};
  glm::vec3 max{x_max, y_max, z_max};
  return {min, max};
}

BoundingBox::BoundingBox(const std::vector<float> &vertices) : BoundingBox() {
  std::pair<glm::vec3, glm::vec3> min_max = compute_min_max(vertices.data(), vertices.size());
  min_coords = {min_max.first.x, min_max.first.y, min_max.first.z};
  max_coords = {min_max.second.x, min_max.second.y, min_max.second.z};
  center = calculateCenter(min_coords, max_coords);
}

BoundingBox::BoundingBox(const axstd::span<float> &vertices) : BoundingBox() {
  std::pair<glm::vec3, glm::vec3> min_max = compute_min_max(vertices.data(), vertices.size());
  min_coords = {min_max.first.x, min_max.first.y, min_max.first.z};
  max_coords = {min_max.second.x, max_coords.x, max_coords.y};
  center = calculateCenter(min_coords, max_coords);
}

ax_device_callable BoundingBox::BoundingBox(const float *vertices, size_t size) : BoundingBox() {
  std::pair<glm::vec3, glm::vec3> min_max = compute_min_max(vertices, size);
  min_coords = {min_max.first.x, min_max.first.y, min_max.first.z};
  max_coords = {min_max.second.x, min_max.second.y, min_max.second.z};
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

bool BoundingBox::operator==(const BoundingBox &other) const {
  return min_coords == other.min_coords && max_coords == other.max_coords && center == other.center;
}
BoundingBox BoundingBox::operator+(const BoundingBox &addbox) const {
  glm::vec3 min = glm::min(min_coords, addbox.min_coords);
  glm::vec3 max = glm::max(max_coords, addbox.max_coords);

  return {min, max};
}

bool BoundingBox::intersect(const glm::vec3 &ray_direction, const glm::vec3 &ray_origin) const {
  float tx1 = (min_coords.x - ray_origin.x) / ray_direction.x, tx2 = (max_coords.x - ray_origin.x) / ray_direction.x;
  float tmin = std::min(tx1, tx2), tmax = std::max(tx1, tx2);
  float ty1 = (min_coords.y - ray_origin.y) / ray_direction.y, ty2 = (max_coords.y - ray_origin.y) / ray_direction.y;
  tmin = std::max(tmin, std::min(ty1, ty2)), tmax = std::min(tmax, std::max(ty1, ty2));
  float tz1 = (min_coords.z - ray_origin.z) / ray_direction.z, tz2 = (max_coords.z - ray_origin.z) / ray_direction.z;
  tmin = std::max(tmin, std::min(tz1, tz2)), tmax = std::min(tmax, std::max(tz1, tz2));
  return tmax >= tmin && tmax > 0;
}
float BoundingBox::intersect(const glm::vec3 &ray_direction, const glm::vec3 &ray_origin, float dist_min) const {
  float tx1 = (min_coords.x - ray_origin.x) / ray_direction.x, tx2 = (max_coords.x - ray_origin.x) / ray_direction.x;
  float tmin = std::min(tx1, tx2), tmax = std::max(tx1, tx2);
  float ty1 = (min_coords.y - ray_origin.y) / ray_direction.y, ty2 = (max_coords.y - ray_origin.y) / ray_direction.y;
  tmin = std::max(tmin, std::min(ty1, ty2)), tmax = std::min(tmax, std::max(ty1, ty2));
  float tz1 = (min_coords.z - ray_origin.z) / ray_direction.z, tz2 = (max_coords.z - ray_origin.z) / ray_direction.z;
  tmin = std::max(tmin, std::min(tz1, tz2)), tmax = std::min(tmax, std::max(tz1, tz2));
  if (tmax >= tmin && tmax > 0 && tmin <= dist_min)
    return tmin;
  return MAXFLOAT;
}

float BoundingBox::area() const {
  const glm::vec3 lwh = max_coords - min_coords;
  return lwh.x * lwh.y + lwh.y * lwh.z + lwh.z * lwh.x;
}

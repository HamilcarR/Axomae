#include "BoundingBox.h"
#include "Logger.h"
#include "Ray.h"
#include "utils_3D.h"
#include <glm/ext/quaternion_common.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include <utility>
using namespace axomae;

constexpr unsigned int THREAD_NUMBERS = 8;
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

BoundingBox operator*(const glm::mat4 &matrix, const BoundingBox &bounding_box) {
  glm::vec3 min_c = matrix * glm::vec4(bounding_box.getMinCoords(), 1.f);
  glm::vec3 max_c = matrix * glm::vec4(bounding_box.getMaxCoords(), 1.f);
  return BoundingBox(min_c, max_c);
}

BoundingBox::BoundingBox(const std::vector<float> &vertices) : BoundingBox() {
  center = glm::vec3(0, 0, 0);
  auto lambda_parallel_compute_bbox =
      [](const std::vector<float> &vertices, unsigned min_index, unsigned max_index) -> std::pair<glm::vec3, glm::vec3> {
    float x_max, y_max, z_max;
    float x_min, y_min, z_min;
    x_max = y_max = z_max = -INT_MAX;
    x_min = y_min = z_min = INT_MAX;
    for (unsigned i = min_index; i < max_index; i += 3) {
      float x_compare = vertices[i], y_compare = vertices[i + 1], z_compare = vertices[i + 2];
      x_max = x_max <= x_compare ? x_compare : x_max;
      y_max = y_max <= y_compare ? y_compare : y_max;
      z_max = z_max <= z_compare ? z_compare : z_max;
      x_min = x_compare <= x_min ? x_compare : x_min;
      y_min = y_compare <= y_min ? y_compare : y_min;
      z_min = z_compare <= z_min ? z_compare : z_min;
    }
    return std::pair(glm::vec3(x_min, y_min, z_min), glm::vec3(x_max, y_max, z_max));
  };
  size_t indiv_thread_array_size = (vertices.size() / 3) / THREAD_NUMBERS;
  if (indiv_thread_array_size == 0) {
    max_coords = glm::vec3(-INT_MAX);
    min_coords = glm::vec3(INT_MAX);
    for (unsigned i = 0; i < vertices.size(); i += 3) {
      glm::vec3 compare = glm::vec3(vertices[i], vertices[i + 1], vertices[i + 2]);
      max_coords = update_max(max_coords, compare);
      min_coords = update_min(min_coords, compare);
    }
  } else {
    size_t remainder_vector_left = (vertices.size() / 3) % THREAD_NUMBERS;
    size_t last_thread_job_left = indiv_thread_array_size + remainder_vector_left;
    last_thread_job_left *= 3;
    indiv_thread_array_size *= 3;
    std::vector<std::future<std::pair<glm::vec3, glm::vec3>>> futures;
    for (unsigned i = 0; i < THREAD_NUMBERS; i++) {
      unsigned min_index = i * indiv_thread_array_size;
      unsigned max_index = i * indiv_thread_array_size + indiv_thread_array_size;
      if (i == THREAD_NUMBERS - 1)
        max_index = i * indiv_thread_array_size + last_thread_job_left;
      futures.push_back(std::async(lambda_parallel_compute_bbox, vertices, min_index, max_index));
    }
    max_coords = glm::vec3(-INT_MAX);
    min_coords = glm::vec3(INT_MAX);
    for (auto it = futures.begin(); it != futures.end(); it++) {
      auto result = it->get();
      glm::vec3 min_c = result.first;
      glm::vec3 max_c = result.second;
      max_coords = update_max(max_coords, max_c);
      min_coords = update_min(min_coords, min_c);
    }
  }
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

glm::vec3 BoundingBox::computeModelViewPosition(const glm::mat4 &modelview) const { return {modelview * glm::vec4(center, 1.f)}; }

static bool test_intersection(const glm::vec3 &x_axis,
                              const glm::vec3 &y_axis,
                              const glm::vec3 &z_axis,
                              const glm::vec3 &ray_origin,
                              const glm::vec3 &ray_direction,
                              const glm::vec3 &delta,
                              const glm::vec3 &tmin_coords,
                              const glm::vec3 &tmax_coords,
                              const float tmin,
                              const float tmax,
                              float &ret_tmin

) {
  /* Intersections on X axis */
  const float delta_x = glm::dot(delta, x_axis);
  const float Dx = glm::dot(ray_direction, x_axis);
  float txmin = (tmin_coords.x - delta_x) / Dx;
  float txmax = (tmax_coords.x - delta_x) / Dx;
  if (txmin > txmax) {
    std::swap(txmin, txmax);
  }

  /* Intersections on Y axis */
  const float delta_y = glm::dot(delta, y_axis);
  const float Dy = glm::dot(ray_direction, y_axis);
  float tymin = (tmin_coords.y - delta_y) / Dy;
  float tymax = (tmax_coords.y - delta_y) / Dy;
  if (tymin > tymax) {
    std::swap(tymin, tymax);
  }

  /* Intersections on Z axis */
  const float delta_z = glm::dot(delta, z_axis);
  const float Dz = glm::dot(ray_direction, z_axis);
  float tzmin = (tmin_coords.z - delta_z) / Dz;
  float tzmax = (tmax_coords.z - delta_z) / Dz;
  if (tzmin > tzmax) {
    std::swap(tzmin, tzmax);
  }

  const float max = std::min(std::min(txmax, tymax), tzmax);
  const float min = std::max(std::max(txmin, tymin), tzmin);
  if (max < min)
    return false;
  if (min < 0)
    return false;
  ret_tmin = min;
  return true;
}

bool BoundingBox::hit(const nova::Ray &ray, float tmin, float tmax, nova::hit_data &hit_data, const nova::base_optionals *user_opts) const {
  auto opt_struct = dynamic_cast<const nova::hit_optionals<glm::mat4> *>(user_opts);
  const glm::mat4 &world_matrix = opt_struct->data;
  const glm::mat4 inv_W = glm::inverse(world_matrix);
  const glm::vec3 x_axis = glm::vec3(1, 0, 0);
  const glm::vec3 y_axis = glm::vec3(0, 1, 0);
  const glm::vec3 z_axis = glm::vec3(0, 0, 1);
  const glm::vec3 ray_origin = inv_W * glm::vec4(ray.origin.x, ray.origin.y, ray.origin.z, 1.f);
  const glm::vec3 ray_direction = inv_W * glm::vec4(ray.direction.x, ray.direction.y, ray.direction.z, 0.f);
  const glm::vec3 tmin_coords = min_coords;
  const glm::vec3 tmax_coords = max_coords;
  const glm::vec3 delta = ray_origin;

  return test_intersection(x_axis, y_axis, z_axis, ray_origin, ray_direction, delta, tmin_coords, tmax_coords, tmin, tmax, hit_data.t);
}

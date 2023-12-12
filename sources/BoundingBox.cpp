#include "../includes/BoundingBox.h"

using namespace axomae;

constexpr unsigned int THREAD_NUMBERS = 12;  // TODO: [AX-27] Create a class reading/writing a config file
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

#include "../includes/PerformanceLogger.h"
// TODO: [AX-12] Parallelize bounding box computation
BoundingBox::BoundingBox(const std::vector<float> &vertices) : BoundingBox() {
  PerformanceLogger log;
  center = glm::vec3(0, 0, 0);
  /*
   * This lambda calculates asynchroneously the minimum/maximum coordinates of every meshes
   */
  auto lambda_parallel_compute_bbox = [](const std::vector<float> &vertices, unsigned min_index, unsigned max_index) -> std::pair<glm::vec3, glm::vec3> {
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

BoundingBox::BoundingBox(glm::vec3 _min_coords, glm::vec3 _max_coords) {
  center = calculateCenter(_min_coords, _max_coords);
  min_coords = _min_coords;
  max_coords = _max_coords;
}

BoundingBox::~BoundingBox() {}

BoundingBox operator*(const glm::mat4 &matrix, const BoundingBox &bounding_box) {
  glm::vec3 min_c = matrix * glm::vec4(bounding_box.getMinCoords(), 1.f);
  glm::vec3 max_c = matrix * glm::vec4(bounding_box.getMaxCoords(), 1.f);
  return BoundingBox(min_c, max_c);
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
  return std::pair<std::vector<float>, std::vector<unsigned>>(vertices, indices);
}

glm::vec3 calculateCenter(glm::vec3 min_coords, glm::vec3 max_coords) {
  glm::vec3 center;
  center.x = (max_coords.x + min_coords.x) / 2;
  center.y = (max_coords.y + min_coords.y) / 2;
  center.z = (max_coords.z + min_coords.z) / 2;
  return center;
}
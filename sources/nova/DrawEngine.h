#ifndef NOVA_DRAW_ENGINE_H
#define NOVA_DRAW_ENGINE_H
#include "Hitable.h"
#include "PerformanceLogger.h"
#include "Ray.h"
#include "ThreadPool.h"
#include "Vector.h"
#include "image_utils.h"
#include "math_camera.h"
#include "math_texturing.h"
#include "nova_camera.h"
#include "nova_texturing.h"
#include "thread_utils.h"
#include <vector>

namespace nova {
  struct NovaResourceHolder {
    texturing::EnvmapResourcesHolder envmap_data;
    camera::CameraResourcesHolder camera_data{};
    threading::ThreadPool *thread_pool{};
    int render_samples{};
  };

  inline bool hit_sphere(const glm::vec3 &center, float radius, const Ray &r, hit_data *r_hit) {
    const glm::vec3 oc = r.origin - center;
    const float b = 2.f * glm::dot(r.direction, oc);
    const float a = glm::dot(r.direction, r.direction);
    const float c = glm::dot(oc, oc) - radius * radius;
    const float determinant = b * b - 4 * a * c;
    if (determinant >= 0) {
      float t1 = (-b - std::sqrt(determinant)) * 0.5f * a;
      if (t1 <= 0)
        return false;
      r_hit->t = t1;
      r_hit->position = r.pointAt(t1);
      r_hit->normal = (r_hit->position - center);
      return true;
    }
    return false;
  }

  inline glm::vec4 color(const Ray &r, const NovaResourceHolder *scene_data) {
    hit_data hit_d;
    float alpha = 1.f;
    glm::vec4 center{0, 0, 0, 1.f};
    center = center;
    if (hit_sphere(center, 1, r, &hit_d)) {
      glm::vec3 normal = glm::normalize(glm::vec4(hit_d.normal, 0.f));
      glm::vec3 sampled_color = texturing::sample_cubemap(normal, &scene_data->envmap_data);
      return {sampled_color, alpha};
    }
    glm::vec3 sample_vector = scene_data->camera_data.inv_T * glm::vec4(r.direction, 0.f);
    return {texturing::sample_cubemap(sample_vector, &scene_data->envmap_data), 1.f};
  }

  inline glm::vec4 color_fragment(const glm::vec2 &coord, const NovaResourceHolder *scene_data) {
    math::camera::camera_ray r = math::camera::ray_inv_mat(coord.x,
                                                           coord.y,
                                                           scene_data->camera_data.screen_width,
                                                           scene_data->camera_data.screen_height,
                                                           scene_data->camera_data.inv_P,
                                                           glm::inverse(scene_data->camera_data.V * scene_data->camera_data.M));
    Ray ray(r.near, r.far);
    return color(ray, scene_data);
  }

  inline void fill_buffer(float *display_buffer,
                          int width_l,
                          int width_h,
                          int height_l,
                          int height_h,
                          const unsigned width,
                          const unsigned height,
                          const NovaResourceHolder *scene_data) {
    for (int y = height_h - 1; y > height_l; y--)
      for (int x = width_l; x < width_h; x++) {
        glm::vec4 rgb = color_fragment({x, height - y}, scene_data);
        unsigned int idx = (y * width + x) * 4;
        display_buffer[idx] = rgb.r;
        display_buffer[idx + 1] = rgb.g;
        display_buffer[idx + 2] = rgb.b;
        display_buffer[idx + 3] = rgb.a;
      }
  }

  inline std::vector<std::future<void>> draw(float *display_buffer,
                                             const unsigned width_resolution,
                                             const unsigned height_resolution,
                                             const NovaResourceHolder *scene_data) {
    AX_ASSERT(scene_data != nullptr, "");
    std::vector<std::future<void>> futs;
    int THREAD_NUM = scene_data->thread_pool->threadNumber();
    std::vector<threading::Tile> tiles = threading::divideByTiles(width_resolution, height_resolution, THREAD_NUM);
    for (const auto &elem : tiles) {
      futs.push_back(scene_data->thread_pool->addTask(true,
                                                      fill_buffer,
                                                      display_buffer,
                                                      elem.width_start,
                                                      elem.width_end,
                                                      elem.height_start,
                                                      elem.height_end,
                                                      width_resolution,
                                                      height_resolution,
                                                      scene_data));
    }
    return futs;
  }

}  // namespace nova
#endif

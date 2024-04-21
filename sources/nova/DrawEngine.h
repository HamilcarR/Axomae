#ifndef NOVA_DRAW_ENGINE_H
#define NOVA_DRAW_ENGINE_H
#include "Hitable.h"
#include "PerformanceLogger.h"
#include "Ray.h"
#include "ThreadPool.h"
#include "Vector.h"
#include "image_utils.h"
#include "math_texturing.h"
#include "nova_camera.h"
#include "nova_texturing.h"
#include "thread_utils.h"
#include <vector>

namespace nova {
  static std::mutex mutex;
  struct NovaResourceHolder {
    texturing::EnvmapResourcesHolder envmap_data;
    camera::CameraResourcesHolder camera_data{};
    threading::ThreadPool *thread_pool{};
    int render_samples;
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
      r_hit->normal = (r_hit->position - center) + glm::vec3(1.f) * 0.5f;
      return true;
    }
    return false;
  }

  inline glm::vec3 color(Ray &r, const NovaResourceHolder *scene_data) {
    hit_data hit_d;
    glm::vec3 center(0, 0, -2);
    if (hit_sphere(center, 0.5, r, &hit_d)) {
      glm::vec3 normal = hit_d.normal;
      glm::vec3 sampled_color = texturing::sample_cubemap(normal, &scene_data->envmap_data);
      return sampled_color;
    }
    return texturing::compute_envmap_background(r, &scene_data->envmap_data);
  }

  inline void fill_buffer(float *display_buffer,
                          int width_l,
                          int width_h,
                          int height_l,
                          int height_h,
                          const int width,
                          const int height,
                          const NovaResourceHolder *scene_data) {
    glm::vec3 llc{-2.f, -1.f, -1.f};
    glm::vec3 hor{4.f, 0.f, 0.f};
    glm::vec3 ver{0.f, 2.f, 0.f};
    glm::vec3 ori(0.f);
    for (int x = width_l; x < width_h; x++)
      for (int y = height_h - 1; y > height_l; y--) {
        double u = math::texture::pixelToUv(x, width);
        double v = math::texture::pixelToUv(y, height);
        unsigned int idx = (y * width + x) * 3;

        Ray r(ori, llc + glm::vec3(u) * hor + ver * glm::vec3(v));
        glm::vec3 rgb = color(r, scene_data);
        display_buffer[idx] += rgb.x * 0.1f;
        display_buffer[idx + 1] += rgb.y * 0.1f;
        display_buffer[idx + 2] += rgb.z * 0.1f;
      }
  }

  inline std::vector<std::future<void>> draw(float *display_buffer,
                                             const int width_resolution,
                                             const int height_resolution,
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

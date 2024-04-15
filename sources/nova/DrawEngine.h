#ifndef NOVA_DRAW_ENGINE_H
#define NOVA_DRAW_ENGINE_H
#include "Hitable.h"
#include "PerformanceLogger.h"
#include "Ray.h"
#include "Vector.h"
#include "image_utils.h"
#include "math_texturing.h"
#include "nova_texturing.h"

#include <vector>
namespace nova {
  constexpr uint8_t THREAD_NUM = 8;

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

  inline glm::vec3 color(Ray &r, const texturing::SceneResourcesHolder *scene_data) {
    hit_data hit_d;
    glm::vec3 center(0, 0, -2);
    if (hit_sphere(center, 0.5, r, &hit_d)) {
      glm::vec3 normal = hit_d.normal;
      glm::vec3 sampled_color = texturing::sample_cubemap(normal, scene_data);
      return sampled_color;
    }
    return texturing::compute_envmap_background(r, scene_data);
  }

  inline void fill_buffer(float *display_buffer,
                          float width_l,
                          float width_h,
                          float height_l,
                          float height_h,
                          int width,
                          int height,
                          const texturing::SceneResourcesHolder *scene_data) {
    glm::vec3 llc{-2.f, -1.f, -1.f};
    glm::vec3 hor{4.f, 0.f, 0.f};
    glm::vec3 ver{0.f, 2.f, 0.f};
    glm::vec3 ori(0.f);
    for (int x = (int)width_l; x <= (int)width_h; x++)
      for (int y = (int)height_h - 1; y >= (int)height_l; y--) {
        double u = math::texture::pixelToUv(x, width);
        double v = math::texture::pixelToUv(y, height);
        Ray r(ori, llc + glm::vec3(u) * hor + ver * glm::vec3(v));
        unsigned int idx = (y * width + x) * 3;
        AX_ASSERT(idx < width * height * 3, "");
        glm::vec3 rgb = color(r, scene_data);
        display_buffer[idx] = rgb.x;
        display_buffer[idx + 1] = rgb.y;
        display_buffer[idx + 2] = rgb.z;
      }
  }

  inline void draw(float *display_buffer, int width, int height, const texturing::SceneResourcesHolder *scene_data) {
    std::vector<std::future<void>> futures;
    futures.reserve(THREAD_NUM);
    for (int i = 0; i < THREAD_NUM; i++) {
      float lb = i * width / THREAD_NUM;
      float hb = i * width / THREAD_NUM + (float)width / THREAD_NUM;
      if (i == THREAD_NUM - 1)
        hb = hb - 1;
      futures.push_back(std::async(std::launch::async, fill_buffer, display_buffer, lb, hb, 0, height, width, height, scene_data));
    }

    for (auto &futs : futures) {
      futs.get();
    }
  }
}  // namespace nova
#endif

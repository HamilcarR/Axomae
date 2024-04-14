#ifndef NOVA_DRAW_ENGINE_H
#define NOVA_DRAW_ENGINE_H
#include "Hitable.h"
#include "Ray.h"
#include "Vector.h"
#include "math_texturing.h"
#include <vector>
namespace nova {
  constexpr uint8_t THREAD_NUM = 8;

  struct SceneResourcesHolder {
    const std::vector<float> *ptr_on_envmap;
    int width;
    int height;
  };

  inline bool hit_sphere(const Vec3f &center, float radius, const Ray &r, hit_data *r_hit) {
    const Vec3f oc = r.origin - center;
    const float b = 2.f * r.direction.dot(oc);
    const float a = r.direction.dot(r.direction);
    const float c = oc.dot(oc) - radius * radius;
    const float determinant = b * b - 4 * a * c;
    if (determinant >= 0) {
      float t1 = (-b - std::sqrt(determinant)) * 0.5f * a;
      if (t1 <= 0)
        return false;
      r_hit->t = t1;
      r_hit->position = r.pointAt(t1);
      r_hit->normal = (r_hit->position - center) + Vec3f(1.f) * 0.5f;
      return true;
    }
    return false;
  }

  inline Vec3f color(Ray &r, const SceneResourcesHolder *scene_data) {
    hit_data hit_d;
    Vec3f center(0, 0, -1);
    if (hit_sphere(center, 0.5, r, &hit_d)) {
      return hit_d.normal;
    }
    Vec3f unit = r.direction;
    unit.normalize();
    float t = 0.5 * (unit.y + 1.f);
    return Vec3f(1.f) * (1 - t) + Vec3f(0.2, 0.7, 1.f) * t;
  }

  inline void fill_buffer(float *display_buffer,
                          float width_l,
                          float width_h,
                          float height_l,
                          float height_h,
                          int width,
                          int height,
                          const SceneResourcesHolder *scene_data) {
    Vec3f llc{-2.f, -1.f, -1.f};
    Vec3f hor{4.f, 0.f, 0.f};
    Vec3f ver{0.f, 2.f, 0.f};
    Vec3f ori(0.f);
    for (int x = (int)width_l; x <= (int)width_h; x++)
      for (int y = (int)height_h - 1; y >= (int)height_l; y--) {
        double u = math::texture::pixelToUv(x, width);
        double v = math::texture::pixelToUv(y, height);
        Ray r(ori, llc + hor * u + ver * v);
        unsigned int idx = (y * width + x) * 3;
        AX_ASSERT(idx < width * height * 3, "");
        Vec3f rgb = color(r, scene_data);
        display_buffer[idx] = rgb.x;
        display_buffer[idx + 1] = rgb.y;
        display_buffer[idx + 2] = rgb.z;
      }
  }

  inline void draw(float *display_buffer, int width, int height, const SceneResourcesHolder *scene_data) {
    std::vector<std::future<void>> futures;
    futures.reserve(THREAD_NUM);
    for (int i = 0; i < THREAD_NUM; i++) {
      float lb = i * width / THREAD_NUM;
      float hb = i * width / THREAD_NUM + width / THREAD_NUM;
      if (i == THREAD_NUM - 1)
        hb = hb - 1;
      futures.push_back(std::async(std::launch::async, fill_buffer, display_buffer, lb, hb, 0, height, width, height, scene_data));
    }

    for (auto &futs : futures)
      futs.get();
  }
}  // namespace nova
#endif

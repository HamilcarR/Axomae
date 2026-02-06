#ifndef HITABLE_H
#define HITABLE_H
#include "IntersectFrame.h"
#include <internal/common/math/utils_3D.h>
namespace nova {

  /**
   * @brief Provides additional data that can be passed to the hit method.
   */

  class base_options {
   public:
    virtual ~base_options() = default;
  };
  template<class T>
  class hit_options : public base_options {
   public:
    T data;
  };

  struct hit_geometry_s {
    glm::vec3 dpdu;
    glm::vec3 dpdv;
    glm::vec3 dndu;
    glm::vec3 dndv;
    glm::vec3 dndx;
    glm::vec3 dndy;
    glm::vec3 e1;
    glm::vec3 e2;
    glm::vec3 ng;
    glm::vec3 position;

    float u, v;
    float t{1e30f};
    float wo_dot_n;

    bool degenerate;
  };

  struct hit_shading_s {
    IntersectFrame frame;
    bool fallback{false};  // In case the mesh doesn't provide attribute buffers;
    float handedness{};    // 1.f for right handed , -1.f if not
  };
  struct intersection_record_s {
    hit_geometry_s geometry{};
    hit_shading_s shading{};
  };

  class Ray;
  class Hitable {
   public:
    virtual ~Hitable() = default;
    virtual bool hit(const Ray &r, float tmin, float tmax, intersection_record_s &data, base_options *user_options) const = 0;
  };
}  // namespace nova
#endif  // Hitable_H

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

  struct intersection_record_s {
    glm::vec3 geometric_normal{}, binormal{}, position{};
    float u{}, v{}, t{1e30f}, wo_dot_n;
  };

  class Ray;
  class Hitable {
   public:
    virtual ~Hitable() = default;
    virtual bool hit(const Ray &r, float tmin, float tmax, intersection_record_s &data, base_options *user_options) const = 0;
  };
}  // namespace nova
#endif  // Hitable_H

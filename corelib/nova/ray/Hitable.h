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

  struct derivatives_s {
    glm::vec3 dpdu;
    glm::vec3 dpdv;
    glm::vec3 dndu;
    glm::vec3 dndv;
    glm::vec3 dndx;
    glm::vec3 dndy;

    glm::vec3 e1;
    glm::vec3 e2;
  };

  struct intersection_record_s {
    derivatives_s deriv;
    glm::vec3 geometric_normal{}, binormal{}, tangent{}, position{};
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

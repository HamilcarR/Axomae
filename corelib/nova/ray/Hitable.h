#ifndef HITABLE_H
#define HITABLE_H
#include "internal/common/math/math_utils.h"
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

  /* Each hit with a shape , primitive and their materials, fills this structure with the hit computed data. */
  struct hit_data {
    glm::vec4 attenuation{};
    glm::vec4 emissive{};
    glm::vec3 normal{};
    glm::vec3 tangent{};
    glm::vec3 bitangent{};
    glm::vec3 position{};
    float u{}, v{}, t{1e30f};
  };

  class Ray;
  class Hitable {
   public:
    virtual ~Hitable() = default;
    virtual bool hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options *user_options) const = 0;
  };
}  // namespace nova
#endif  // Hitable_H

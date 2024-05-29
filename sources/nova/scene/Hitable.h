#ifndef HITABLE_H
#define HITABLE_H
#include "math_utils.h"
namespace nova {

  /**
   * @brief Provides additional data that can be passed to the hit method
   */

  struct base_options {
    virtual ~base_options() = default;
  };
  template<class T>
  struct hit_options : public base_options {
    T data;
  };

  struct hit_data {
    float t{};
    glm::vec3 normal{};
    glm::vec3 position{};
    glm::vec4 attenuation{};
  };

  class Ray;
  class Hitable {
   public:
    virtual ~Hitable() = default;
    virtual bool hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options *user_options) const = 0;
  };
}  // namespace nova
#endif  // Hitable_H

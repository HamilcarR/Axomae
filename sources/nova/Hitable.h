#ifndef HITABLE_H
#define HITABLE_H
#include "Vector.h"
namespace nova {

  struct base_optionals {
    virtual ~base_optionals() = default;
  };

  /**
   * @brief Provides additional data that can be passed to the hit method
   * @tparam T
   */
  template<class T>
  struct hit_optionals : public base_optionals {
    T data;
  };

  struct hit_data {
    float t{};
    Vec3f normal{};
    Vec3f position{};
  };
  class Ray;
  class Hitable {
   public:
    virtual ~Hitable() = default;
    virtual bool hit(const Ray &r, float tmin, float tmax, hit_data &data, const base_optionals *user_options) const = 0;
  };
}  // namespace nova
#endif  // Hitable_H

#ifndef HITABLE_H
#define HITABLE_H
#include "Vector.h"
namespace nova {

  struct base_options {
    virtual ~base_options() = default;
  };

  /**
   * @brief Provides additional data that can be passed to the hit method
   */
  template<class T>
  struct hit_options : public base_options {
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
    virtual bool hit(const Ray &r, float tmin, float tmax, hit_data &data, const base_options *user_options) const = 0;
  };
}  // namespace nova
#endif  // Hitable_H

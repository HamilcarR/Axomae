#ifndef VEC3_H
#define VEC3_H
#include "internal/macro/project_macros.h"
#include <cmath>
#include <cstdlib>
#include <ostream>
namespace math::geometry {
  template<class T>
  class Vec3 {
   public:
    T x{};
    T y{};
    T z{};

   public:
    Vec3() = default;
    explicit Vec3(T a);
    Vec3(T a, T b, T c);

    inline void normalize();
    [[nodiscard]] inline float magnitude() const;
    [[nodiscard]] inline float angle(const Vec3 &arg) const;
    [[nodiscard]] inline float dot(const Vec3 &arg) const;
    [[nodiscard]] inline Vec3 cross(const Vec3 &arg) const;
    [[nodiscard]] inline Vec3 operator-(const Vec3 &arg) const;
    [[nodiscard]] inline Vec3 operator+(const Vec3 &arg) const;
    template<class Y>
    [[nodiscard]] inline Vec3 operator*(const Y &arg) const;
    [[nodiscard]] inline Vec3 operator/(const Vec3 &arg) const;
    [[nodiscard]] inline bool operator==(const Vec3 &arg) const;

    friend std::ostream &operator<<(std::ostream &os, const Vec3 &v) {
      os << "(" << v.x << "," << v.y << "," << v.z << ")";
      return os;
    }
  };

  template<class T>
  Vec3<T>::Vec3(T a) : x(a), y(a), z(a) {}

  template<class T>
  Vec3<T>::Vec3(T a, T b, T c) : x(a), y(b), z(c) {}

  template<class T>
  float Vec3<T>::magnitude() const {
    return std::sqrt(x * x + y * y + z * z);
  }

  template<class T>
  void Vec3<T>::normalize() {
    float mag = this->magnitude();
    x = std::abs(x / mag);
    y = std::abs(y / mag);
    z = std::abs(z / mag);
  }

  template<class T>
  float Vec3<T>::dot(const Vec3 &arg) const {
    return x * arg.x + y * arg.y + z * arg.z;
  }

  template<class T>
  bool Vec3<T>::operator==(const Vec3 &arg) const {
    return x == arg.x && y == arg.y && z == arg.z;
  }
  template<class T>
  Vec3<T> Vec3<T>::operator/(const Vec3 &arg) const {
#ifndef NDEBUG
    AX_ASSERT(arg.x != 0 && arg.y != 0 && arg.z != 0, "");
#endif
    return {x / arg.x, y / arg.y, z / arg.z};
  }

  template<class T>
  Vec3<T> Vec3<T>::operator+(const Vec3 &arg) const {
    return {x + arg.x, y + arg.y, z + arg.z};
  }
  template<class T>
  Vec3<T> Vec3<T>::operator-(const Vec3 &arg) const {
    return {x - arg.x, y - arg.y, z - arg.z};
  }

  template<class T>
  Vec3<T> Vec3<T>::cross(const Vec3 &arg) const {
    return {y * arg.z - z * arg.y, z * arg.x - x * arg.z, x * arg.y - y * arg.x};
  }

  template<class T>
  template<class Y>
  Vec3<T> Vec3<T>::operator*(const Y &arg) const {
    if constexpr (IS_ARITHMETHIC(Y))
      return *this * Vec3(arg);
    else
      return {x * arg.x, y * arg.y, z * arg.z};
  }

  template<class T>
  float Vec3<T>::angle(const Vec3 &arg) const {
    Vec3 n1 = normalize();
    Vec3 n2 = arg.normalize();
    float cosine = std::abs(n1.dot(n2));
    return std::acos(cosine);
  }
}  // namespace math::geometry
#endif  // Vec3_H

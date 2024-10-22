#ifndef VEC2_H
#define VEC2_H
#include "internal/macro/project_macros.h"
#include <cmath>
#include <cstdlib>
#include <ostream>
namespace math::geometry {
  template<class T>
  class Vec2 {
   public:
    T x{};
    T y{};

   public:
    Vec2() = default;
    explicit Vec2(T a);
    Vec2(T a, T b);

    inline void normalize();
    ax_no_discard inline float angle(const Vec2 &arg) const;
    ax_no_discard inline float dot(const Vec2 &arg) const;
    ax_no_discard inline float magnitude() const;
    ax_no_discard inline Vec2 operator-(const Vec2 &arg) const;
    ax_no_discard inline Vec2 operator+(const Vec2 &arg) const;
    template<class Y>
    ax_no_discard inline Vec2 operator*(const Y &arg) const;
    ax_no_discard inline Vec2 operator/(const Vec2 &arg) const;
    ax_no_discard inline bool operator==(const Vec2 &arg) const;

    friend std::ostream &operator<<(std::ostream &os, const Vec2 &p) {
      os << "(" << p.x << "," << p.y << ")";
      return os;
    }
  };

  template<class T>
  Vec2<T>::Vec2(T a) : x(a), y(a) {}

  template<class T>
  Vec2<T>::Vec2(T a, T b) : x(a), y(b) {}

  template<class T>
  Vec2<T> Vec2<T>::operator-(const Vec2 &arg) const {
    return {x - arg.x, y - arg.y};
  }

  template<class T>
  Vec2<T> Vec2<T>::operator+(const Vec2 &arg) const {
    return {x + arg.x, y + arg.y};
  }

  template<class T>
  template<class Y>
  Vec2<T> Vec2<T>::operator*(const Y &arg) const {
    if constexpr (IS_ARITHMETHIC(Y))
      return *this * Vec2(arg);
    else
      return {x * arg.x, y * arg.y};
  }

  template<class T>
  Vec2<T> Vec2<T>::operator/(const Vec2 &arg) const {
#ifndef NDEBUG
    AX_ASSERT(arg.x != 0 && arg.y != 0, "");
#endif
    return {x / arg.x, y / arg.y};
  }

  template<class T>
  bool Vec2<T>::operator==(const Vec2 &arg) const {
    return x == arg.x && y == arg.y;
  }

  template<class T>
  float Vec2<T>::magnitude() const {
    return std::sqrt(x * x + y * y);
  }

  template<class T>
  void Vec2<T>::normalize() {
    float mag = this->magnitude();
    x = std::abs(x / mag);
    y = std::abs(y / mag);
  }

  template<class T>
  float Vec2<T>::angle(const Vec2 &arg) const {
    Vec2 n1 = normalize();
    Vec2 n2 = arg.normalize();
    float cosine = std::abs(n1.dot(n2));
    return std::acos(cosine);
  }

  template<class T>
  float Vec2<T>::dot(const Vec2 &arg) const {
    return arg.x * x + arg.y * y;
  }
}  // namespace math::geometry
#endif  // Vec2_H

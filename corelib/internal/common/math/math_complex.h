#ifndef MATH_COMPLEX_H
#define MATH_COMPLEX_H
#include "math_utils.h"
#include <cmath>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_utils.h>

namespace math {

  template<class T>
  class complex {
   public:
    T real{}, imaginary{};

    ax_device_callable_inlined complex() : real(T{}), imaginary(T{}) {}
    ax_device_callable_inlined complex(T real, T imaginary) : real(real), imaginary(imaginary) {}

    ax_device_callable_inlined complex operator+(const complex &c) const { return complex(real + c.real, imaginary + c.imaginary); }

    ax_device_callable_inlined friend complex operator+(T val, const complex &c) { return {c.real + val, c.imaginary}; }

    ax_device_callable_inlined complex operator-(const complex &c) const { return complex(real - c.real, imaginary - c.imaginary); }

    ax_device_callable_inlined friend complex operator-(T val, const complex &c) { return {val - c.real, c.imaginary}; }

    ax_device_callable_inlined complex operator*(const complex &c) const {
      return complex(real * c.real - imaginary * c.imaginary, real * c.imaginary + imaginary * c.real);
    }

    ax_device_callable_inlined friend complex operator*(T value, const complex &c) { return {c.real * value, c.imaginary * value}; }

    ax_device_callable_inlined complex operator/(const complex &c) const {
      T denom = c.real * c.real + c.imaginary * c.imaginary;
      return complex((real * c.real + imaginary * c.imaginary) / denom, (imaginary * c.real - real * c.imaginary) / denom);
    }

    ax_device_callable_inlined friend complex operator/(T value, complex c) {
      complex nominator = value * c.conjugate();
      complex denominator = c * c.conjugate();
      return nominator / denominator;
    }

    ax_device_callable_inlined complex &operator+=(const complex &c) {
      real += c.real;
      imaginary += c.imaginary;
      return *this;
    }

    ax_device_callable_inlined complex &operator-=(const complex &c) {
      real -= c.real;
      imaginary -= c.imaginary;
      return *this;
    }

    ax_device_callable_inlined complex &operator*=(const complex &c) {
      T new_real = real * c.real - imaginary * c.imaginary;
      T new_imag = real * c.imaginary + imaginary * c.real;
      real = new_real;
      imaginary = new_imag;
      return *this;
    }

    ax_device_callable_inlined complex &operator/=(const complex &c) {
      T denom = c.real * c.real + c.imaginary * c.imaginary;
      T new_real = (real * c.real + imaginary * c.imaginary) / denom;
      T new_imag = (imaginary * c.real - real * c.imaginary) / denom;
      real = new_real;
      imaginary = new_imag;
      return *this;
    }

    ax_device_callable_inlined bool operator==(const complex &c) const { return real == c.real && imaginary == c.imaginary; }
    ax_device_callable_inlined bool operator!=(const complex &c) const { return !(*this == c); }

    ax_device_callable_inlined complex operator-() const { return complex(-real, -imaginary); }

    ax_device_callable_inlined complex conjugate() const { return complex(real, -imaginary); }

    ax_device_callable_inlined T norm() const { return real * real + imaginary * imaginary; }

    ax_device_callable_inlined T abs() const { return math::sqrt(norm()); }
  };

  using fcomplex = complex<float>;

  template<class T>
  ax_device_callable_inlined constexpr complex<T> sqrt(const complex<T> &c) {
    T a = c.real;
    T b = c.imaginary;
    T mag = math::sqrt(math::sqr(a) + math::sqr(b));
    T x = math::sqrt((mag + a) / 2.f);
    T y = math::sqrt((mag - a) / 2.f);

    complex<T> res(x, y);
    return res;  // Returns only one solution as the second is trivial (-res).
  }
}  // namespace math

#endif
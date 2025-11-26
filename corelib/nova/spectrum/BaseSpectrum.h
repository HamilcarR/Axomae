#ifndef BASESPECTRUM_H
#define BASESPECTRUM_H

#include <internal/common/math/utils_3D.h>
#include <internal/common/utils.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/macro/project_macros.h>

namespace nova {

  template<class S>
  class BaseSpectrum {

    ax_device_callable_inlined float *samples() { return static_cast<S *>(this)->samples; }
    ax_device_callable_inlined const float *samples() const { return static_cast<const S *>(this)->samples; }
    ax_device_callable_inlined unsigned size() const { return static_cast<const S *>(this)->SPECTRUM_SAMPLES; }

   public:
    ax_device_callable_inlined friend S operator/(float r, const S &s) {
      AX_ASSERT_TRUE(s.isValid());
      S result;
      for (unsigned i = 0; i < s.size(); i++) {
        AX_ASSERT_TRUE(s[i] != 0.f);
        result[i] = r / s[i];
      }
      return result;
    }

    ax_device_callable_inlined bool operator<=(const S &other) const {
      S self = *static_cast<const S *>(this);
      AX_ASSERT_EQ(size(), other.size());
      for (unsigned i = 0; i < size(); i++)
        if (self[i] > other[i])
          return false;
      return true;
    }

    ax_device_callable_inlined bool operator>=(const S &other) const {
      S self = *static_cast<const S *>(this);
      AX_ASSERT_EQ(size(), other.size());
      for (unsigned i = 0; i < size(); i++)
        if (self[i] < other[i])
          return false;
      return true;
    }

    ax_device_callable_inlined bool operator!=(const S &other) const {
      S self = *static_cast<const S *>(this);
      AX_ASSERT_EQ(size(), other.size());
      for (unsigned i = 0; i < size(); i++)
        if (self[i] == other[i])
          return false;
      return true;
    }

    ax_device_callable_inlined bool operator<(const S &other) const {
      S self = *static_cast<const S *>(this);
      AX_ASSERT_EQ(size(), other.size());
      for (unsigned i = 0; i < size(); i++)
        if (self[i] >= other[i])
          return false;
      return true;
    }

    ax_device_callable_inlined bool operator>(const S &other) const {
      S self = *static_cast<const S *>(this);
      AX_ASSERT_EQ(size(), other.size());
      for (unsigned i = 0; i < size(); i++)
        if (self[i] <= other[i])
          return false;
      return true;
    }

    ax_device_callable_inlined bool operator==(const S &other) const {
      S self = *static_cast<const S *>(this);
      AX_ASSERT_EQ(size(), other.size());
      for (unsigned i = 0; i < size(); i++) {
        if (self[i] != other[i])
          return false;
      }
      return true;
    }

    ax_device_callable_inlined S operator+(float v) const {
      S result = *static_cast<const S *>(this);
      for (unsigned i = 0; i < size(); i++)
        result[i] += v;
      return result;
    }

    ax_device_callable_inlined S operator-(float v) const {
      S result = *static_cast<const S *>(this);
      for (unsigned i = 0; i < size(); i++)
        result[i] -= v;
      return result;
    }

    ax_device_callable_inlined S operator*(float v) const {
      S result = *static_cast<const S *>(this);
      for (unsigned i = 0; i < size(); i++)
        result[i] *= v;
      return result;
    }

    ax_device_callable_inlined S operator/(float v) const {
      AX_ASSERT_NEQ(v, 0);
      S result = *static_cast<const S *>(this);
      for (unsigned i = 0; i < size(); i++)
        result[i] /= v;
      return result;
    }

    ax_device_callable_inlined S &operator+=(float v) {
      for (unsigned i = 0; i < size(); i++)
        (*this)[i] += v;
      return *static_cast<S *>(this);
    }

    ax_device_callable_inlined S &operator-=(float v) {
      for (unsigned i = 0; i < size(); i++)
        (*this)[i] -= v;
      return *static_cast<S *>(this);
    }

    ax_device_callable_inlined S &operator*=(float v) {
      for (unsigned i = 0; i < size(); i++)
        (*this)[i] *= v;
      return *static_cast<S *>(this);
    }

    ax_device_callable_inlined S &operator/=(float v) {
      AX_ASSERT_NEQ(v, 0);
      for (unsigned i = 0; i < size(); i++)
        (*this)[i] /= v;
      return *static_cast<S *>(this);
    }

    ax_device_callable_inlined S operator+(const S &other) const {
      S result = *static_cast<const S *>(this);
      for (unsigned i = 0; i < size(); i++)
        result[i] += other[i];
      return result;
    }

    ax_device_callable_inlined friend S operator+(float v, const S &other) {
      S result;
      for (unsigned i = 0; i < other.size(); i++)
        result[i] = v + other[i];
      return result;
    }

    ax_device_callable_inlined friend S operator-(float v, const S &other) {
      S result;
      for (unsigned i = 0; i < other.size(); i++)
        result[i] = v - other[i];
      return result;
    }

    ax_device_callable_inlined friend S operator*(float v, const S &other) {
      S result;
      for (unsigned i = 0; i < other.size(); i++)
        result[i] = v * other[i];
      return result;
    }

    ax_device_callable_inlined S operator-(const S &other) const {
      S result = *static_cast<const S *>(this);
      for (unsigned i = 0; i < size(); i++)
        result[i] -= other[i];
      return result;
    }

    ax_device_callable_inlined S operator*(const S &other) const {
      S result = *static_cast<const S *>(this);
      for (unsigned i = 0; i < size(); i++)
        result[i] *= other[i];
      return result;
    }

    ax_device_callable_inlined S operator/(const S &other) const {
      S result = *static_cast<const S *>(this);
      for (unsigned i = 0; i < size(); i++) {
        AX_ASSERT_NEQ(other[i], 0);
        result[i] /= other[i];
      }
      return result;
    }

    ax_device_callable_inlined S &operator+=(const S &other) {
      for (unsigned i = 0; i < size(); i++)
        (*this)[i] += other[i];
      return *static_cast<S *>(this);
    }

    ax_device_callable_inlined S &operator-=(const S &other) {
      for (unsigned i = 0; i < size(); i++)
        (*this)[i] -= other[i];
      return *static_cast<S *>(this);
    }

    ax_device_callable_inlined S &operator*=(const S &other) {
      for (unsigned i = 0; i < size(); i++)
        (*this)[i] *= other[i];
      return *static_cast<S *>(this);
    }

    ax_device_callable_inlined S &operator/=(const S &other) {
      for (unsigned i = 0; i < size(); i++) {
        AX_ASSERT_NEQ(other[i], 0);
        (*this)[i] /= other[i];
      }
      return *static_cast<S *>(this);
    }

    ax_device_callable_inlined explicit operator bool() const {
      const S &self = *static_cast<const S *>(this);
      for (unsigned i = 0; i < size(); i++)
        if (self[i] != 0.f)
          return true;
      return false;
    }

    ax_device_callable_inlined bool isValid() const {
      const S &self = *static_cast<const S *>(this);
      for (unsigned i = 0; i < size(); i++)
        if (ISNAN(self[i]) && ISINF(self[i]))
          return false;
      return true;
    }

    ax_device_callable_inlined float max() const {
      const S *self = static_cast<const S *>(this);
      float max = -1e30f;
      for (unsigned i = 0; i < size(); i++)
        if (max <= samples()[i])
          max = samples()[i];
      return max;
    }
    ax_device_callable_inlined float min() const {
      const S &self = *static_cast<const S *>(this);
      float min = 1e30f;
      for (unsigned i = 0; i < size(); i++)
        if (min >= self[i])
          min = self[i];
      return min;
    }

    ax_device_callable_inlined float operator[](unsigned i) const {
      AX_ASSERT_LT(i, size());
      return samples()[i];
    }

    ax_device_callable_inlined float &operator[](unsigned i) {
      AX_ASSERT_LT(i, size());
      return samples()[i];
    }

    ax_device_callable_inlined float average() const {
      float v = 0;
      for (unsigned i = 0; i < size(); i++)
        v += (*this)[i];
      return v / size();
    }
  };

}  // namespace nova
#endif

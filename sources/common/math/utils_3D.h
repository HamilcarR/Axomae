#ifndef UTILS_3D_H
#define UTILS_3D_H
#include "constants.h"
#include "math_utils.h"
#include <map>

namespace math {
  namespace geometry {

    struct Point2D {
      float x;
      float y;
      void print() { std::cout << x << "     " << y << "\n"; }
      friend std::ostream &operator<<(std::ostream &os, const Point2D &p) {
        os << "(" << p.x << "," << p.y << ")";
        return os;
      }
    };

    class Vect3D {
     public:
      friend std::ostream &operator<<(std::ostream &os, const Vect3D &v) {
        os << "(" << v.x << "," << v.y << "," << v.z << ")";
        return os;
      }

      void print() { std::cout << x << "     " << y << "      " << z << "\n"; }
      inline float magnitude() { return std::sqrt(x * x + y * y + z * z); }
      inline void normalize() {
        auto mag = this->magnitude();
        x = std::abs(x / mag);
        y = std::abs(y / mag);
        z = std::abs(z / mag);
      }
      inline float dot(Vect3D V) { return x * V.x + y * V.y + z * V.z; }

      float x;
      float y;
      float z;
    };

  }  // namespace geometry
}  // namespace math
#endif

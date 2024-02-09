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

    /***************************************************************************************************************/
    /* Get barycentric coordinates of I in triangle P1P2P3 */
    inline Vect3D barycentric_lerp(Point2D P1, Point2D P2, Point2D P3, Point2D I) {
      float W1 = ((P2.y - P3.y) * (I.x - P3.x) + (P3.x - P2.x) * (I.y - P3.y)) / ((P2.y - P3.y) * (P1.x - P3.x) + (P3.x - P2.x) * (P1.y - P3.y));
      float W2 = ((P3.y - P1.y) * (I.x - P3.x) + (P1.x - P3.x) * (I.y - P3.y)) / ((P2.y - P3.y) * (P1.x - P3.x) + (P3.x - P2.x) * (P1.y - P3.y));
      float W3 = 1 - W1 - W2;
      Vect3D v = {W1, W2, W3};
      return v;
    }

  }  // namespace geometry
}  // namespace math
#endif

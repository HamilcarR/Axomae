#ifndef UTILS_3D_H
#define UTILS_3D_H
#include "Math.h"
#include "constants.h"
#include <map>

/*TODO : Create classes , implement clean up */
namespace axomae {

  struct Point2D {
    float x;
    float y;
    void print() { std::cout << x << "     " << y << "\n"; }
    friend std::ostream &operator<<(std::ostream &os, const Point2D &p) {
      os << "(" << p.x << "," << p.y << ")";
      return os;
    }
  };

  struct Vect3D {
    float x;
    float y;
    float z;

    friend std::ostream &operator<<(std::ostream &os, const Vect3D &v) {
      os << "(" << v.x << "," << v.y << "," << v.z << ")";
      return os;
    }

    void print() { std::cout << x << "     " << y << "      " << z << "\n"; }
    auto magnitude() { return sqrt(x * x + y * y + z * z); }
    void normalize() {
      auto mag = this->magnitude();
      x = abs(x / mag);
      y = abs(y / mag);
      z = abs(z / mag);
    }
    auto dot(Vect3D V) { return x * V.x + y * V.y + z * V.z; }
  };

}  // namespace axomae

#endif

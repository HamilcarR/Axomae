#include "Vector.h"

namespace math::geometry {

  /*************************************************************************************************************/
  Vect2D::Vect2D(float a, float b) : x(a), y(b) {}

  std::ostream &operator<<(std::ostream &os, const Vect2D &p) {
    os << "(" << p.x << "," << p.y << ")";
    return os;
  }

  float Vect2D::magnitude() const { return std::sqrt(x * x + y * y); }

  void Vect2D::normalize() {
    auto mag = this->magnitude();
    x = std::abs(x / mag);
    y = std::abs(y / mag);
  }

  float Vect2D::dot(const IVector &vec) const {
    const Vect2D *arg = dynamic_cast<const Vect2D *>(&vec);
    return arg->x * x + arg->y * y;
  }
  /*************************************************************************************************************/

  Vect3D::Vect3D(float a, float b, float c) : x(a), y(b), z(c) {}

  float Vect3D::magnitude() const { return std::sqrt(x * x + y * y + z * z); }

  void Vect3D::normalize() {
    auto mag = this->magnitude();
    x = std::abs(x / mag);
    y = std::abs(y / mag);
    z = std::abs(z / mag);
  }

  float Vect3D::dot(const IVector &arg) const {
    const Vect3D *V = dynamic_cast<const Vect3D *>(&arg);
    return x * V->x + y * V->y + z * V->z;
  }

  std::ostream &operator<<(std::ostream &os, const Vect3D &v) {
    os << "(" << v.x << "," << v.y << "," << v.z << ")";
    return os;
  }
}  // namespace math::geometry
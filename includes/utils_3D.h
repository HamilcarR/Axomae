#ifndef UTILS_3D_H
#define UTILS_3D_H
// clang-format off

#include <GL/glew.h>
#include "DebugGL.h"
#include "Object3D.h"
#include "constants.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <glm/common.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/projection.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/vec3.hpp>
#include <map>
// clang-format on

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

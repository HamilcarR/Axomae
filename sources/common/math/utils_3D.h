#ifndef UTILS_3D_H
#define UTILS_3D_H
#include "Vector.h"
#include "constants.h"
#include "math_utils.h"

namespace math {
  namespace geometry {

    /* Get barycentric coordinates of I in triangle P1P2P3 */
    inline Vect3D barycentric_lerp(Vect2D P1, Vect2D P2, Vect2D P3, Vect2D I) {
      float W1 = ((P2.y - P3.y) * (I.x - P3.x) + (P3.x - P2.x) * (I.y - P3.y)) / ((P2.y - P3.y) * (P1.x - P3.x) + (P3.x - P2.x) * (P1.y - P3.y));
      float W2 = ((P3.y - P1.y) * (I.x - P3.x) + (P1.x - P3.x) * (I.y - P3.y)) / ((P2.y - P3.y) * (P1.x - P3.x) + (P3.x - P2.x) * (P1.y - P3.y));
      float W3 = 1 - W1 - W2;
      Vect3D v = {W1, W2, W3};
      return v;
    }

  }  // namespace geometry
}  // namespace math
#endif

#include "ImageManager.h"
#include "Rgb.h"
#include "constants.h"
#include "utils_3D.h"

namespace exception {
  class NoNormalsException : public GenericException {
   public:
    NoNormalsException() : GenericException() { saveErrorString("This 3D model doesn't provide normals"); }
  };

  class NoTangentsException : public GenericException {
   public:
    NoTangentsException() : GenericException() { saveErrorString("This 3D model doesn't provide tangents"); }
  };

  class NoBitangentsException : public GenericException {
   public:
    NoBitangentsException() : GenericException() { saveErrorString("This 3D model doesn't provide bitangents"); }
  };

  class NoUvException : public GenericException {
   public:
    NoUvException() : GenericException() { saveErrorString("This 3D model doesn't provide UVs"); }
  };

  class NoIndicesException : public GenericException {
   public:
    NoIndicesException() : GenericException() { saveErrorString("This 3D model doesn't provide indices"); }
  };
}  // namespace exception

using namespace math::geometry;
using ImageManager = axomae::ImageManager;
float bounding_coords(float x, float y, float z, bool min) {
  if (min) {
    if (x <= y)
      return x <= z ? x : z;
    else
      return y <= z ? y : z;
  } else {
    if (x >= y)
      return x >= z ? x : z;
    else
      return y >= z ? y : z;
  }
}

/***************************************************************************************************************/
/* Get barycentric coordinates of I in triangle P1P2P3 */
inline Vect3D barycentric_lerp(Point2D P1, Point2D P2, Point2D P3, Point2D I) {
  float W1 = ((P2.y - P3.y) * (I.x - P3.x) + (P3.x - P2.x) * (I.y - P3.y)) / ((P2.y - P3.y) * (P1.x - P3.x) + (P3.x - P2.x) * (P1.y - P3.y));
  float W2 = ((P3.y - P1.y) * (I.x - P3.x) + (P1.x - P3.x) * (I.y - P3.y)) / ((P2.y - P3.y) * (P1.x - P3.x) + (P3.x - P2.x) * (P1.y - P3.y));
  float W3 = 1 - W1 - W2;
  Vect3D v = {W1, W2, W3};
  return v;
}

/***************************************************************************************************************/
inline Vect3D tan_space_transform(Vect3D T, Vect3D BT, Vect3D N, Vect3D I) {
  Vect3D result = {I.x * BT.x + I.y * BT.y + I.z * BT.z, I.x * T.x + T.y * I.y + T.z * I.z, I.x * N.x + N.y * I.y + I.z * N.z};
  return result;
}

/***************************************************************************************************************/
inline image::Rgb compute_normals_set_pixels_rgb(Point2D P1,
                                                 Point2D P2,
                                                 Point2D P3,
                                                 Vect3D N1,
                                                 Vect3D N2,
                                                 Vect3D N3,
                                                 Vect3D BT1,
                                                 Vect3D BT2,
                                                 Vect3D BT3,
                                                 Vect3D T1,
                                                 Vect3D T2,
                                                 Vect3D T3,
                                                 int x,
                                                 int y,
                                                 bool tangent_space) {

  Point2D I = {static_cast<float>(x), static_cast<float>(y)};
  Vect3D C = barycentric_lerp(P1, P2, P3, I);
  if (C.x >= 0 && C.y >= 0 && C.z >= 0) {
    auto interpolate = [&C](Vect3D N1, Vect3D N2, Vect3D N3) {
      Vect3D normal = {N1.x * C.x + N2.x * C.y + N3.x * C.z, N1.y * C.x + N2.y * C.y + N3.y * C.z, N1.z * C.x + N2.z * C.y + N3.z * C.z};
      return normal;
    };
    if (tangent_space) {
      Vect3D normal = interpolate(N1, N2, N3);
      Vect3D B = {(BT1.x + BT2.x + BT3.x) / 3, (BT1.y + BT2.y + BT3.y) / 3, (BT1.z + BT2.z + BT3.z) / 3};
      Vect3D T = {(T1.x + T2.x + T3.x) / 3, (T1.y + T2.y + T3.y) / 3, (T1.z + T2.z + T3.z) / 3};
      Vect3D N = {(N1.x + N2.x + N3.x) / 3, (N1.y + N2.y + N3.y) / 3, (N1.z + N2.z + N3.z) / 3};
      B.normalize();
      T.normalize();
      N.normalize();
      normal = tan_space_transform(T, B, N, normal);
      normal.normalize();
      image::Rgb rgb = image::Rgb((normal.x * 255 + 255) / 2, (normal.y * 255 + 255) / 2, (normal.z * 255 + 255) / 2, 0);
      return rgb;
    } else {
      N1.normalize();
      N2.normalize();
      N3.normalize();
      Vect3D normal = interpolate(N1, N2, N3);
      image::Rgb rgb = image::Rgb((normal.x * 255 + 255) / 2, (normal.y * 255 + 255) / 2, (normal.z * 255 + 255) / 2, 0);
      return rgb;
    }
  } else {
    return image::Rgb(0, 0, 0);
  }
}

void throwIfNotValid(const Object3D &object) {
  bool tangents_empty = object.tangents.empty(), bitangents_empty = object.bitangents.empty(), normals_empty = object.normals.empty(),
       uv_empty = object.uv.empty(), indices_empty = object.indices.empty();
  if (tangents_empty)
    throw exception::NoTangentsException();
  if (bitangents_empty)
    throw exception::NoBitangentsException();
  if (normals_empty)
    throw exception::NoNormalsException();
  if (uv_empty)
    throw exception::NoUvException();
  if (indices_empty)
    throw exception::NoIndicesException();
}

/***************************************************************************************************************/
/** @brief Computes the normals at the position of each texel of a 3D model and displays them on the projected UV
 * This function allows us to check the distribution of normals across the mesh.
 * Returns std::vector of size width * height * 3 as we work with an RGB image .
 */
std::vector<uint8_t> ImageManager::project_uv_normals(const Object3D &object, int width, int height, bool tangent_space) {
  try {
    throwIfNotValid(object);
  } catch (const GenericException &e) {
    throw;
  }

  std::vector<uint8_t> raw_img;
  raw_img.resize(width * height * 3);

  /*project normals on texture coordinates and interpolate them between each vertex of a face*/
  for (unsigned int i = 0; i < object.indices.size(); i += 3) {
    const std::vector<unsigned int> &index = object.indices;
    /* texture coordinates of each vertex in a face */
    Point2D P1 = {object.uv[index[i] * 2], object.uv[index[i] * 2 + 1]};
    Point2D P2 = {object.uv[index[i + 1] * 2], object.uv[index[i + 1] * 2 + 1]};
    Point2D P3 = {object.uv[index[i + 2] * 2], object.uv[index[i + 2] * 2 + 1]};
    /* Normals of each vertex*/
    Vect3D N1 = {object.normals[index[i] * 3], object.normals[index[i] * 3 + 1], object.normals[index[i] * 3 + 2]};
    Vect3D N2 = {object.normals[index[i + 1] * 3], object.normals[index[i + 1] * 3 + 1], object.normals[index[i + 1] * 3 + 2]};
    Vect3D N3 = {object.normals[index[i + 2] * 3], object.normals[index[i + 2] * 3 + 1], object.normals[index[i + 2] * 3 + 2]};
    /* bitangents */
    Vect3D BT1 = {object.bitangents[index[i] * 3], object.bitangents[index[i] * 3 + 1], object.bitangents[index[i] * 3 + 2]};
    Vect3D BT2 = {object.bitangents[index[i + 1] * 3], object.bitangents[index[i + 1] * 3 + 1], object.bitangents[index[i + 1] * 3 + 2]};
    Vect3D BT3 = {object.bitangents[index[i + 2] * 3], object.bitangents[index[i + 2] * 3 + 1], object.bitangents[index[i + 2] * 3 + 2]};
    /* tangents */
    Vect3D T1 = {object.tangents[index[i] * 3], object.tangents[index[i] * 3 + 1], object.tangents[index[i] * 3 + 2]};
    Vect3D T2 = {object.tangents[index[i + 1] * 3], object.tangents[index[i + 1] * 3 + 1], object.tangents[index[i + 1] * 3 + 2]};
    Vect3D T3 = {object.tangents[index[i + 2] * 3], object.tangents[index[i + 2] * 3 + 1], object.tangents[index[i + 2] * 3 + 2]};

    /*check if UVs are correctly set in bounds ... if not , we clamp*/
    auto clamp_uv = [&](Point2D P) {
      if (P.x > 1.f)
        P.x = std::fmod(P.x, 1.f);
      if (P.y > 1.f)
        P.y = std::fmod(P.y, 1.f);
      return P;
    };
    P1 = clamp_uv(P1);
    P2 = clamp_uv(P2);
    P3 = clamp_uv(P3);

    P1.x *= static_cast<float>(width);
    P1.y *= static_cast<float>(height);
    P2.x *= static_cast<float>(width);
    P2.y *= static_cast<float>(height);
    P3.x *= static_cast<float>(width);
    P3.y *= static_cast<float>(height);
    int x_max = static_cast<int>(bounding_coords(P1.x, P2.x, P3.x, false));
    int x_min = static_cast<int>(bounding_coords(P1.x, P2.x, P3.x, true));
    int y_max = static_cast<int>(bounding_coords(P1.y, P2.y, P3.y, false));
    int y_min = static_cast<int>(bounding_coords(P1.y, P2.y, P3.y, true));
    for (int x = x_min; x <= x_max; x++)
      for (int y = y_min; y <= y_max; y++) {
        image::Rgb val = compute_normals_set_pixels_rgb(P1, P2, P3, N1, N2, N3, BT1, BT2, BT3, T1, T2, T3, x, y, true);
        int idx = (y * width + x) * 3;
        if (!(val == image::Rgb(0, 0, 0))) {
          raw_img[idx] = static_cast<uint8_t>(val.red);
          raw_img[idx + 1] = static_cast<uint8_t>(val.green);
          raw_img[idx + 2] = static_cast<uint8_t>(val.blue);
        }
      }
  }
  return raw_img;
}
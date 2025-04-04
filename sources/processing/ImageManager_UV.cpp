#include "ImageManager.h"
#include "internal/common/exception/GenericException.h"
#include "internal/common/image/Rgb.h"
#include "internal/common/math/utils_3D.h"
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
inline float bounding_coords(float x, float y, float z, bool min) {
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
inline Vec3f tan_space_transform(Vec3f T, Vec3f BT, Vec3f N, Vec3f I) {
  Vec3f result = {I.x * BT.x + I.y * BT.y + I.z * BT.z, I.x * T.x + T.y * I.y + T.z * I.z, I.x * N.x + N.y * I.y + I.z * N.z};
  return result;
}

/***************************************************************************************************************/
inline image::Rgb compute_normals_set_pixels_rgb(Vec2f P1,
                                                 Vec2f P2,
                                                 Vec2f P3,
                                                 Vec3f N1,
                                                 Vec3f N2,
                                                 Vec3f N3,
                                                 Vec3f BT1,
                                                 Vec3f BT2,
                                                 Vec3f BT3,
                                                 Vec3f T1,
                                                 Vec3f T2,
                                                 Vec3f T3,
                                                 int x,
                                                 int y,
                                                 bool tangent_space) {

  Vec2f I = {static_cast<float>(x), static_cast<float>(y)};
  Vec3f C = math::geometry::barycentric_lerp(P1, P2, P3, I);
  if (C.x >= 0 && C.y >= 0 && C.z >= 0) {
    auto interpolate = [&C](Vec3f N1, Vec3f N2, Vec3f N3) {
      Vec3f normal = {N1.x * C.x + N2.x * C.y + N3.x * C.z, N1.y * C.x + N2.y * C.y + N3.y * C.z, N1.z * C.x + N2.z * C.y + N3.z * C.z};
      return normal;
    };
    if (tangent_space) {
      Vec3f normal = interpolate(N1, N2, N3);
      Vec3f B = {(BT1.x + BT2.x + BT3.x) / 3, (BT1.y + BT2.y + BT3.y) / 3, (BT1.z + BT2.z + BT3.z) / 3};
      Vec3f T = {(T1.x + T2.x + T3.x) / 3, (T1.y + T2.y + T3.y) / 3, (T1.z + T2.z + T3.z) / 3};
      Vec3f N = {(N1.x + N2.x + N3.x) / 3, (N1.y + N2.y + N3.y) / 3, (N1.z + N2.z + N3.z) / 3};
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
      Vec3f normal = interpolate(N1, N2, N3);
      image::Rgb rgb = image::Rgb((normal.x * 255 + 255) / 2, (normal.y * 255 + 255) / 2, (normal.z * 255 + 255) / 2, 0);
      return rgb;
    }
  } else {
    return {0, 0, 0};
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
std::vector<uint8_t> ImageManager::projectUVNormals(const Object3D &object, int width, int height, bool tangent_space) {
  try {
    throwIfNotValid(object);
  } catch (const exception::GenericException &e) {
    throw;
  }

  std::vector<uint8_t> raw_img;
  raw_img.resize(width * height * 3);

  /*project normals on texture coordinates and interpolate them between each vertex of a face*/
  for (unsigned int i = 0; i < object.indices.size(); i += 3) {
    const axstd::span<unsigned int> &index = object.indices;
    /* texture coordinates of each vertex in a face */
    Vec2f P1 = {object.uv[index[i] * 2], object.uv[index[i] * 2 + 1]};
    Vec2f P2 = {object.uv[index[i + 1] * 2], object.uv[index[i + 1] * 2 + 1]};
    Vec2f P3 = {object.uv[index[i + 2] * 2], object.uv[index[i + 2] * 2 + 1]};
    /* Normals of each vertex*/
    Vec3f N1 = {object.normals[index[i] * 3], object.normals[index[i] * 3 + 1], object.normals[index[i] * 3 + 2]};
    Vec3f N2 = {object.normals[index[i + 1] * 3], object.normals[index[i + 1] * 3 + 1], object.normals[index[i + 1] * 3 + 2]};
    Vec3f N3 = {object.normals[index[i + 2] * 3], object.normals[index[i + 2] * 3 + 1], object.normals[index[i + 2] * 3 + 2]};
    /* bitangents */
    Vec3f BT1 = {object.bitangents[index[i] * 3], object.bitangents[index[i] * 3 + 1], object.bitangents[index[i] * 3 + 2]};
    Vec3f BT2 = {object.bitangents[index[i + 1] * 3], object.bitangents[index[i + 1] * 3 + 1], object.bitangents[index[i + 1] * 3 + 2]};
    Vec3f BT3 = {object.bitangents[index[i + 2] * 3], object.bitangents[index[i + 2] * 3 + 1], object.bitangents[index[i + 2] * 3 + 2]};
    /* tangents */
    Vec3f T1 = {object.tangents[index[i] * 3], object.tangents[index[i] * 3 + 1], object.tangents[index[i] * 3 + 2]};
    Vec3f T2 = {object.tangents[index[i + 1] * 3], object.tangents[index[i + 1] * 3 + 1], object.tangents[index[i + 1] * 3 + 2]};
    Vec3f T3 = {object.tangents[index[i + 2] * 3], object.tangents[index[i + 2] * 3 + 1], object.tangents[index[i + 2] * 3 + 2]};

    /*check if UVs are correctly set in bounds ... if not , we clamp*/
    auto clamp_uv = [&](Vec2f P) {
      if (P.x > 1.f)
        P.x = std::fmod(P.x, 1.f);
      if (P.y > 1.f)
        P.y = std::fmod(P.y, 1.f);
      return P;
    };
    P1 = clamp_uv(P1);
    P2 = clamp_uv(P2);
    P3 = clamp_uv(P3);

    P1.x *= static_cast<float>(width - 1);
    P1.y *= static_cast<float>(height - 1);
    P2.x *= static_cast<float>(width - 1);
    P2.y *= static_cast<float>(height - 1);
    P3.x *= static_cast<float>(width - 1);
    P3.y *= static_cast<float>(height - 1);
    int x_max = static_cast<int>(bounding_coords(P1.x, P2.x, P3.x, false));
    int x_min = static_cast<int>(bounding_coords(P1.x, P2.x, P3.x, true));
    int y_max = static_cast<int>(bounding_coords(P1.y, P2.y, P3.y, false));
    int y_min = static_cast<int>(bounding_coords(P1.y, P2.y, P3.y, true));
    for (int x = x_min; x <= x_max; x++)
      for (int y = y_min; y <= y_max; y++) {
        image::Rgb val = compute_normals_set_pixels_rgb(P1, P2, P3, N1, N2, N3, BT1, BT2, BT3, T1, T2, T3, x, y, tangent_space);
        int idx = (y * width + x) * 3;
        AX_ASSERT(idx >= 0 && idx < width * height * 3, "");
        if (!(val == image::Rgb(0, 0, 0))) {
          raw_img[idx] = static_cast<uint8_t>(val.red);
          raw_img[idx + 1] = static_cast<uint8_t>(val.green);
          raw_img[idx + 2] = static_cast<uint8_t>(val.blue);
        }
      }
  }
  return raw_img;
}
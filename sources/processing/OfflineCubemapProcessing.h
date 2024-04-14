#ifndef OFFLINECUBEMAPPROCESSING_H
#define OFFLINECUBEMAPPROCESSING_H
#include "GenericException.h"
#include "GenericTextureProcessing.h"
#include "Mutex.h"
#include "math_utils.h"
#include <fstream>
#include <immintrin.h>
#include <sstream>

constexpr glm::dvec3 RED = glm::dvec3(1., 0, 0);
constexpr glm::dvec3 YELLOW = glm::dvec3(0, 1, 1);
constexpr glm::dvec3 GREEN = glm::dvec3(0, 1, 0);
constexpr glm::dvec3 BLUE = glm::dvec3(0, 0, 1);
constexpr glm::dvec3 BLACK = glm::dvec3(0);

class TextureInvalidDimensionsException : public exception::GenericException {
 public:
  TextureInvalidDimensionsException() : GenericException() { saveErrorString("This texture has invalid dimensions. (negative or non numerical)"); }
};

class TextureNonPowerOfTwoDimensionsException : public exception::GenericException {
 public:
  TextureNonPowerOfTwoDimensionsException() : GenericException() { saveErrorString("This texture has dimensions that are not a power of two."); }
};

template<class T>
class EnvmapProcessing final : GenericTextureProcessing {

 private:
  const std::vector<float> *data;
  unsigned width;
  unsigned height;
  unsigned channels;

 private:
  static constexpr unsigned MAX_THREADS = 8;  // Retrieve from config

 public:
  EnvmapProcessing(const std::vector<T> &_data, unsigned _width, unsigned _height, unsigned int num_channels = 3);
  TextureData computeSpecularIrradiance(double roughness);
  /**
   * @brief This method wrap around if the texture coordinates provided land beyond the texture dimensions, repeating
   * the texture values on both axes.
   */
  template<class D>
  glm::dvec2 wrapAroundTexCoords(D u, D v) const;

  /**
   * @brief Normalizes a set of pixel coordinates into texture bounds.
   */
  glm::dvec2 wrapAroundPixelCoords(int x, int y) const;

  /**
   * @brief Computes the linear interpolation of a point based on 4 of it's neighbours.
   */
  glm::dvec3 bilinearInterpolate(const glm::dvec2 &top_left,
                                 const glm::dvec2 &top_right,
                                 const glm::dvec2 &bottom_left,
                                 const glm::dvec2 &bottom_right,
                                 const glm::dvec2 &point) const;

  /**
   * @brief Sample the original texture using pixel coordinates.
   */
  glm::dvec3 discreteSample(int x, int y) const;

  /**
   * @brief In case the coordinates go beyond the bounds of the texture , we wrap around .
   * In addition , sampling texels may return an interpolated value when u,v are converted to a (x ,y) non
   * integer texture coordinate.
   */
  template<class D>
  glm::dvec3 uvSample(D u, D v) const;
  template<typename D>
  void launchAsyncDiffuseIrradianceCompute(D delta, float *f_data, unsigned width_begin, unsigned width_end, unsigned _width, unsigned _height) const;
  std::unique_ptr<TextureData> computeDiffuseIrradiance(unsigned _width, unsigned _height, unsigned delta, bool gpu) const;
  template<class D>
  glm::dvec3 computeIrradianceSingleTexel(unsigned x, unsigned y, unsigned samples, D tangent, D bitangent, D normal) const;
  template<class D>
  glm::dvec3 computeIrradianceImportanceSampling(D x, D y, D z, unsigned _width, unsigned _height, unsigned total_samples) const;
};

template<class T>
EnvmapProcessing<T>::EnvmapProcessing(const std::vector<T> &_data, const unsigned _width, const unsigned _height, const unsigned int num_channels)
    : data(&_data) {
  if (!isValidDim(_width) || !isValidDim(_height))
    throw TextureInvalidDimensionsException();
  if (!isDimPowerOfTwo(_width) || !isDimPowerOfTwo(_height))
    throw TextureNonPowerOfTwoDimensionsException();
  AX_ASSERT(num_channels == 3, "");
  width = _width;
  height = _height;
  channels = num_channels;
}

template<class T>
TextureData EnvmapProcessing<T>::computeSpecularIrradiance(double roughness) {
  return {};
}

template<class T>
template<class D>
glm::dvec2 EnvmapProcessing<T>::wrapAroundTexCoords(const D u, const D v) const {
  D u_integer = 0, v_integer = 0;
  D u_double_p = 0., v_double_p = 0.;
  u_integer = std::floor(u);
  v_integer = std::floor(v);
  if (u > 1. || u < 0.)
    u_double_p = u - u_integer;
  else
    u_double_p = u;
  if (v > 1. || v < 0.)
    v_double_p = v - v_integer;
  else
    v_double_p = v;
  return {u_double_p, v_double_p};
}

template<class T>
glm::dvec2 EnvmapProcessing<T>::wrapAroundPixelCoords(const int x, const int y) const {
  unsigned int x_coord = 0, y_coord = 0;
  int _width = static_cast<int>(width);
  int _height = static_cast<int>(height);
  if (x >= _width)
    x_coord = x % _width;
  else if (x < 0)
    x_coord = _width + (x % _width);
  else
    x_coord = x;
  if (y >= _height)
    y_coord = y % _height;
  else if (y < 0)
    y_coord = _height + (y % _height);
  else
    y_coord = y;
  return {x_coord, y_coord};
}

template<class T>
glm::dvec3 EnvmapProcessing<T>::bilinearInterpolate(const glm::dvec2 &top_left,
                                                    const glm::dvec2 &top_right,
                                                    const glm::dvec2 &bottom_left,
                                                    const glm::dvec2 &bottom_right,
                                                    const glm::dvec2 &point) const {
  const double u = (point.x - top_left.x) / (top_right.x - top_left.x);
  const double v = (point.y - top_left.y) / (bottom_left.y - top_left.y);
  const glm::dvec3 top_interp = (1 - u) * discreteSample(top_left.x, top_left.y) + u * discreteSample(top_right.x, top_right.y);
  const glm::dvec3 bot_interp = (1 - u) * discreteSample(bottom_left.x, bottom_left.y) + u * discreteSample(bottom_right.x, bottom_right.y);
  return (1 - v) * top_interp + v * bot_interp;
}

template<class T>
glm::dvec3 EnvmapProcessing<T>::discreteSample(int x, int y) const {
  const glm::dvec2 normalized = wrapAroundPixelCoords(x, y);
  const float r = (*data)[(normalized.y * width + normalized.x) * channels];
  const float g = (*data)[(normalized.y * width + normalized.x) * channels + 1];
  const float b = (*data)[(normalized.y * width + normalized.x) * channels + 2];
  return {r, g, b};
}

template<class T>
template<class D>
glm::dvec3 EnvmapProcessing<T>::uvSample(const D u, const D v) const {
  const glm::dvec2 wrap_uv = wrapAroundTexCoords(u, v);
  const glm::dvec2 pixel_coords = glm::dvec2(math::texture::uvToPixel(wrap_uv.x, width), math::texture::uvToPixel(wrap_uv.y, height));
  const glm::dvec2 top_left(std::floor(pixel_coords.x), std::floor(pixel_coords.y));
  const glm::dvec2 top_right(std::floor(pixel_coords.x) + 1, std::floor(pixel_coords.y));
  const glm::dvec2 bottom_left(std::floor(pixel_coords.x), std::floor(pixel_coords.y) + 1);
  const glm::dvec2 bottom_right(std::floor(pixel_coords.x) + 1, std::floor(pixel_coords.y) + 1);
  const glm::dvec3 texel_value = bilinearInterpolate(top_left, top_right, bottom_left, bottom_right, pixel_coords);
  return texel_value;
}

template<class T>
template<typename D>
void EnvmapProcessing<T>::launchAsyncDiffuseIrradianceCompute(
    const D delta, float *f_data, const unsigned width_begin, const unsigned width_end, const unsigned _width, const unsigned _height) const {
  for (unsigned i = width_begin; i <= width_end; i++) {
    for (unsigned j = 0; j < _height; j++) {
      glm::dvec2 uv = glm::dvec2(math::texture::pixelToUv(i, _width), math::texture::pixelToUv(j, _height));
      const glm::dvec2 sph = math::spherical::uvToSpherical(uv.x, uv.y);
      const glm::dvec3 cart = math::spherical::sphericalToCartesian(sph.x, sph.y);
      glm::dvec3 irrad = computeIrradianceImportanceSampling(cart.x, cart.y, cart.z, _width, _height, delta);
      unsigned index = (j * _width + i) * channels;
      f_data[index] = static_cast<float>(irrad.x);
      f_data[index + 1] = static_cast<float>(irrad.y);
      f_data[index + 2] = static_cast<float>(irrad.z);
    }
  }
}
template<class T>
template<class D>
glm::dvec3 EnvmapProcessing<T>::computeIrradianceSingleTexel(
    const unsigned x, const unsigned y, const unsigned samples, const D tangent, const D bitangent, const D normal) const {
  glm::dvec3 random = math::importance_sampling::pgc3d(x, y, samples);
  double phi = 2 * PI * random.x;
  double theta = asin(sqrt(random.y));
  glm::dvec3 uv_cart = math::spherical::sphericalToCartesian(phi, theta);
  uv_cart = uv_cart.x * tangent + uv_cart.y * bitangent + uv_cart.z * normal;
  auto spherical = math::spherical::cartesianToSpherical(uv_cart.x, uv_cart.y, uv_cart.z);
  glm::dvec2 uvt = math::spherical::sphericalToUv(spherical.x, spherical.y);
  return uvSample(uvt.x, uvt.y);
}

template<class T>
template<class D>
glm::dvec3 EnvmapProcessing<T>::computeIrradianceImportanceSampling(
    const D x, const D y, const D z, const unsigned _width, const unsigned _height, const unsigned total_samples) const {
  unsigned int samples = 0;
  glm::dvec3 irradiance{0};
  glm::dvec3 normal{x, y, z};
  glm::dvec3 right_vec{1.f, 0.f, 0.f};
  glm::dvec2 uv = math::spherical::sphericalToUv(math::spherical::cartesianToSpherical(x, y, z));
  glm::dvec2 pix = glm::dvec2(math::texture::uvToPixel(uv.x, _width), math::texture::uvToPixel(uv.y, _height));
  double dd = glm::dot(right_vec, normal);
  glm::dvec3 tangent = glm::dvec3(0.0, 1.0, 0.0);
  if (1.0 - abs(dd) > math::epsilon)
    tangent = glm::normalize(glm::cross(right_vec, normal));
  glm::dvec3 bitangent = glm::cross(normal, tangent);
  for (samples = 0; samples < total_samples; samples++)
    irradiance += computeIrradianceSingleTexel((unsigned)pix.x, (unsigned)pix.y, samples, tangent, bitangent, normal);
  return irradiance / static_cast<double>(total_samples);
}
#endif

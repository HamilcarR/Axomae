#ifndef OFFLINECUBEMAPPROCESSING_H
#define OFFLINECUBEMAPPROCESSING_H
#include "GenericException.h"
#include "GenericTextureProcessing.h"
#include "Math.h"
#include "Mutex.h"
#include "PerformanceLogger.h"
#include <fstream>
#include <immintrin.h>
#include <sstream>

constexpr long double epsilon = 1e-6;
constexpr glm::dvec3 RED = glm::dvec3(1., 0, 0);
constexpr glm::dvec3 YELLOW = glm::dvec3(0, 1, 1);
constexpr glm::dvec3 GREEN = glm::dvec3(0, 1, 0);
constexpr glm::dvec3 BLUE = glm::dvec3(0, 0, 1);
constexpr glm::dvec3 BLACK = glm::dvec3(0);

class TextureInvalidDimensionsException : virtual public GenericException {
 public:
  TextureInvalidDimensionsException() : GenericException() { saveErrorString("This texture has invalid dimensions. (negative or non numerical)"); }
  virtual ~TextureInvalidDimensionsException() {}
};

class TextureNonPowerOfTwoDimensionsException : virtual public GenericException {
 public:
  TextureNonPowerOfTwoDimensionsException() : GenericException() { saveErrorString("This texture has dimensions that are not a power of two."); }
  virtual ~TextureNonPowerOfTwoDimensionsException() {}
};

// TODO: [AX-51] Create generic texture processing class
template<class T>
class EnvmapProcessing : virtual public GenericTextureProcessing {
 public:
  virtual bool isDimPowerOfTwo(int dimension) const override { return (dimension & (dimension - 1)) == 0; }
  virtual bool isValidDim(int dimension) const override { return !std::isnan(dimension) && dimension > 0; }

  /**
   * @brief Construct a texture from an envmap double HDR with 3 channels
   *
   * @param _data Float data of the HDR texture.
   * @param _width Pixel width size .
   * @param _height Pixel height size .
   */
  EnvmapProcessing(const T *_data, const unsigned _width, const unsigned _height, const unsigned int num_channels = 3) {

    if (!isValidDim(_width) || !isValidDim(_height))
      throw TextureInvalidDimensionsException();
    else if (!isDimPowerOfTwo(_width) || !isDimPowerOfTwo(_height))
      throw TextureNonPowerOfTwoDimensionsException();
    width = _width;
    height = _height;
    channels = num_channels;
    for (unsigned i = 0; i < width * height * num_channels; i += num_channels) {
      data.push_back(glm::vec3(_data[i], _data[i + 1], _data[i + 2]));
    }
  }
  /**
   * @brief Destroy the Envmap Processing object
   *
   */
  virtual ~EnvmapProcessing() {}

  /**
   * @brief Returns the stored texture.
   *
   * @return const std::vector<glm::vec3>&
   */
  const std::vector<glm::vec3> &getData() { return data; }

  /**
   * @brief Computes Specular Prefiltered envmap
   *
   * @param roughness
   * @return TextureData
   */
  TextureData computeSpecularIrradiance(double roughness);

  /* Implementation of templated methods*/
 public:
  /**
   * @brief This method wrap around if the texture coordinates provided land beyond the texture dimensions, repeating
   * the texture values on both axes.
   *
   * @tparam D Data type of the coordinates.
   * @param u Horizontal UV coordinates.
   * @param v Vertical UV coordinates.
   * @return const glm::dvec2 Normalized coordinates in the [0 , 1] range.
   */
  template<class D>
  inline const glm::dvec2 wrapAroundTexCoords(const D u, const D v) const {
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
    return glm::dvec2(u_double_p, v_double_p);
  }

  /**
   * @brief Normalizes a set of pixel coordinates into texture bounds.
   *
   * @param x Horizontal coordinates.
   * @param y Vertical coordinates.
   * @return const glm::dvec2 Normalized coordinates
   */
  inline const glm::dvec2 wrapAroundPixelCoords(const int x, const int y) const {
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
    return glm::dvec2(x_coord, y_coord);
  }

  /**
   * @brief Computes the linear interpolation of a point based on 4 of it's neighbours.
   *
   * @param top_left Top left pixel .
   * @param top_right Top right pixel .
   * @param bottom_left Bottom left pixel .
   * @param bottom_right Bottom right pixel .
   * @param point Coordinates of the point being computed.
   * @return * const T
   */
  const glm::dvec3 bilinearInterpolate(const glm::dvec2 top_left,
                                       const glm::dvec2 top_right,
                                       const glm::dvec2 bottom_left,
                                       const glm::dvec2 bottom_right,
                                       const glm::dvec2 point) const {
    const double u = (point.x - top_left.x) / (top_right.x - top_left.x);
    const double v = (point.y - top_left.y) / (bottom_left.y - top_left.y);
    const glm::dvec3 top_interp = (1 - u) * discreteSample(top_left.x, top_left.y) + u * discreteSample(top_right.x, top_right.y);
    const glm::dvec3 bot_interp = (1 - u) * discreteSample(bottom_left.x, bottom_left.y) + u * discreteSample(bottom_right.x, bottom_right.y);
    return (1 - v) * top_interp + v * bot_interp;
  }

  /**
   * @brief Sample the original texture using pixel coordinates.
   *
   * @param x Width in pixels.
   * @param y Height in pixels.
   * @return const T
   */
  inline const glm::dvec3 discreteSample(int x, int y) const {
    const glm::dvec2 normalized = wrapAroundPixelCoords(static_cast<int>(x), static_cast<int>(y));
    const glm::dvec3 texel_value = data[normalized.y * width + normalized.x];
    return texel_value;
  }

  /**
   * @brief This method samples a value from the equirectangular envmap
   *! Note : In case the coordinates go beyond the bounds of the texture , we wrap around .
   *! In addition , sampling texels may return a bilinear interpolated value when u,v are converted to a (x ,y) non
   *integer texture coordinate.
   * @tparam D Type of the coordinates
   * @param u Horizontal uv coordinates
   * @param v Vertical uv coordinates
   * @return T Returns a texel value of type T
   */
  template<class D>
  inline const glm::dvec3 uvSample(const D u, const D v) const {
    const glm::dvec2 wrap_uv = wrapAroundTexCoords(u, v);
    const glm::dvec2 pixel_coords = glm::dvec2(texture_math::uvToPixel(wrap_uv.x, width), texture_math::uvToPixel(wrap_uv.y, height));
    const glm::dvec2 top_left(std::floor(pixel_coords.x), std::floor(pixel_coords.y));
    const glm::dvec2 top_right(std::floor(pixel_coords.x) + 1, std::floor(pixel_coords.y));
    const glm::dvec2 bottom_left(std::floor(pixel_coords.x), std::floor(pixel_coords.y) + 1);
    const glm::dvec2 bottom_right(std::floor(pixel_coords.x) + 1, std::floor(pixel_coords.y) + 1);
    const glm::dvec3 texel_value = bilinearInterpolate(top_left, top_right, bottom_left, bottom_right, pixel_coords);
    return texel_value;
  }

  /**
   * @brief Computes the value of the diffuse irradiance on a part of the texture.  Used with threads .
   *
   * @tparam D Type of the delta step .
   * @param delta Size of the hemisphere sampling steps , or number of samples if using importance sampling + low
   * discrep rand generators .
   * @param f_data Original texture.
   * @param width_begin X_min of the computation .
   * @param width_end X_max of the computation .
   * @param height Height of the original texture .
   * @param use_importance_sampling True if using importance sampling.
   */
  template<typename D>
  void launchAsyncDiffuseIrradianceCompute(
      const D delta, float *f_data, const unsigned width_begin, const unsigned width_end, const unsigned _width, const unsigned _height) const {
    for (unsigned i = width_begin; i <= width_end; i++) {
      for (unsigned j = 0; j < _height; j++) {
        glm::dvec2 uv = glm::dvec2(texture_math::pixelToUv(i, _width), texture_math::pixelToUv(j, _height));
        const glm::dvec2 sph = spherical_math::uvToSpherical(uv.x, uv.y);
        const glm::dvec2 sph_to_uv = spherical_math::sphericalToUv(sph);
        const glm::dvec3 cart = spherical_math::sphericalToCartesian(sph.x, sph.y);
        glm::dvec3 irrad = computeIrradianceImportanceSampling(cart.x, cart.y, cart.z, _width, _height, delta);
        unsigned index = (j * _width + i) * channels;
        f_data[index] = static_cast<float>(irrad.x);
        f_data[index + 1] = static_cast<float>(irrad.y);
        f_data[index + 2] = static_cast<float>(irrad.z);
      }
    }
  }

  /**
   * @brief Bake an equirect envmap to an irradiance map
   * @tparam D Type of the delta .
   * @param delta Size of the step if calculating irradiance using the integral all over the hemisphere , or number of
   * samples if using importance sampling.
   * @param use_importance_sampling Set to false to use a full integral over the hemisphere , using delta as step . True
   * if we want to solve the integral using importance sampling.
   * @return std::unique_ptr<TextureData> Texture data containing width , height , and double f_data about the newly
   * created map.
   */
  template<typename D>
  std::unique_ptr<TextureData> computeDiffuseIrradiance(const unsigned _width, const unsigned _height, const D delta) const {
    if (!isValidDim(_width) || !isValidDim(_height))
      throw TextureInvalidDimensionsException();
    if (!isDimPowerOfTwo(_width) || !isDimPowerOfTwo(_height))
      throw TextureNonPowerOfTwoDimensionsException();
    TextureData envmap_tex_data;
    envmap_tex_data.width = _width;
    envmap_tex_data.height = _height;
    envmap_tex_data.mipmaps = 0;
    envmap_tex_data.f_data.resize(_width * _height * channels);
    envmap_tex_data.nb_components = channels;
    std::memset(&envmap_tex_data.f_data[0], 0, _width * _height * channels * sizeof(float));
    unsigned index = 0;
    std::vector<std::shared_future<void>> futures;
    for (unsigned i = 1; i <= MAX_THREADS; i++) {
      unsigned int width_max = (_width / MAX_THREADS) * i, width_min = width_max - (_width / MAX_THREADS);
      if (i == MAX_THREADS)
        width_max += _width % MAX_THREADS - 1;
      auto lambda = [this, &envmap_tex_data](const D delta, const unsigned width_min, const unsigned width_max) {
        this->launchAsyncDiffuseIrradianceCompute(delta, envmap_tex_data.f_data, width_min, width_max, envmap_tex_data.width, envmap_tex_data.height);
      };
      futures.push_back(std::async(std::launch::async, lambda, delta, width_min, width_max));
    }
    for (auto it = futures.begin(); it != futures.end(); it++) {
      it->get();
    }
    return std::make_unique<TextureData>(envmap_tex_data);
  }

  template<class D>
  inline glm::dvec3 computeIrradianceSingleTexel(
      const unsigned x, const unsigned y, const unsigned samples, const D tangent, const D bitangent, const D normal) const {
    glm::dvec3 random = importance_sampling::pgc3d((unsigned)x, (unsigned)y, samples);
    double phi = 2 * PI * random.x;
    double theta = asin(sqrt(random.y));
    glm::dvec3 uv_cart = spherical_math::sphericalToCartesian(phi, theta);
    uv_cart = uv_cart.x * tangent + uv_cart.y * bitangent + uv_cart.z * normal;
    auto spherical = spherical_math::cartesianToSpherical(uv_cart.x, uv_cart.y, uv_cart.z);
    glm::dvec2 uvt = spherical_math::sphericalToUv(spherical.x, spherical.y);
    return uvSample(uvt.x, uvt.y);
  }

  /**
   * @brief Returns the irradiance value at a specific position of the texture using importance sampling.
   *
   * @tparam D Data type of the coordinates
   * @param x Cartesian X
   * @param y Cartesian Y
   * @param z Cartesian Z
   * @param total_samples Total number of samples.
   * @return const glm::dvec3 Irradiance value texel
   */
  template<class D>
  inline const glm::dvec3 computeIrradianceImportanceSampling(
      const D x, const D y, const D z, const unsigned _width, const unsigned _height, const unsigned total_samples) const {
    unsigned int samples = 0;
    glm::dvec3 irradiance = glm::dvec3(0.f);
    glm::dvec3 normal(x, y, z);
    glm::dvec3 someVec = glm::dvec3(1.0, 0.0, 0.0);
    glm::dvec2 uv = spherical_math::sphericalToUv(spherical_math::cartesianToSpherical(x, y, z));
    glm::dvec2 pix = glm::dvec2(texture_math::uvToPixel(uv.x, _width), texture_math::uvToPixel(uv.y, _height));
    float dd = glm::dot(someVec, normal);
    glm::dvec3 tangent = glm::dvec3(0.0, 1.0, 0.0);
    if (1.0 - abs(dd) > 1e-6)
      tangent = glm::normalize(glm::cross(someVec, normal));
    glm::dvec3 bitangent = glm::cross(normal, tangent);
    for (samples = 0; samples < total_samples; samples++)
      irradiance += computeIrradianceSingleTexel((unsigned)pix.x, (unsigned)pix.y, samples, tangent, bitangent, normal);
    return irradiance / static_cast<double>(total_samples);
  }

 protected:
  std::vector<glm::vec3> data;
  unsigned width;
  unsigned height;
  unsigned channels;

 private:
  static constexpr unsigned MAX_THREADS = 8;
  mutable Mutex mutex;
};

#endif

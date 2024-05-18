#ifndef OFFLINECUBEMAPPROCESSING_H
#define OFFLINECUBEMAPPROCESSING_H
#include "Axomae_macros.h"
#include "GenericException.h"

#include "ThreadPool.h"
#include "math_utils.h"

inline bool isDimPowerOfTwo(int dimension) { return (dimension & (dimension - 1)) == 0; }
inline bool isValidDim(int dimension) { return dimension > 0; }

constexpr glm::dvec3 RED = glm::dvec3(1., 0, 0);
constexpr glm::dvec3 YELLOW = glm::dvec3(0, 1, 1);
constexpr glm::dvec3 GREEN = glm::dvec3(0, 1, 0);
constexpr glm::dvec3 BLUE = glm::dvec3(0, 0, 1);
constexpr glm::dvec3 BLACK = glm::dvec3(0);

class TextureInvalidDimensionsException : public exception::GenericException {
 public:
  TextureInvalidDimensionsException() : GenericException() {
    saveErrorString("Provided texture has invalid dimensions. (negative or non numerical)");
  }
};

class TextureNonPowerOfTwoDimensionsException : public exception::GenericException {
 public:
  TextureNonPowerOfTwoDimensionsException() : GenericException() { saveErrorString("Provided texture has dimensions that are not a power of two."); }
};

class TextureData;

template<class T>
class TextureOperations {

 private:
  const std::vector<T> *data{};
  unsigned width{};
  unsigned height{};
  unsigned channels{};

 private:
  static constexpr unsigned MAX_THREADS = 8;  // Retrieve from config

 public:
  TextureOperations() = default;
  TextureOperations(const std::vector<T> &_data, unsigned _width, unsigned _height, unsigned int num_channels = 3);
  ~TextureOperations();
  TextureOperations(const TextureOperations &copy);
  TextureOperations(TextureOperations &&move) noexcept;
  TextureOperations &operator=(const TextureOperations &copy);
  TextureOperations &operator=(TextureOperations &&move) noexcept;

  /**
   * Applies F to each channel of the current texture "data" and copy it in "dest"
   */
  template<class F, class... Args>
  void processTexture(T *dest, int dest_width, int dest_height, F &&functor, Args &&...args) const;

  /**
   * @brief This method wrap around if the texture coordinates provided land beyond the texture dimensions, repeating
   * the texture values on both axes.
   */
  template<class D>
  glm::vec2 wrapAroundTexCoords(D u, D v) const;

  /**
   * @brief Normalizes a set of pixel coordinates into texture bounds.
   */
  [[nodiscard]] glm::vec2 wrapAroundPixelCoords(int x, int y) const;

  /**
   * @brief Computes the linear interpolation of a point based on 4 of it's neighbours.
   */
  [[nodiscard]] glm::vec3 bilinearInterpolate(const glm::vec2 &top_left,
                                              const glm::vec2 &top_right,
                                              const glm::vec2 &bottom_left,
                                              const glm::vec2 &bottom_right,
                                              const glm::vec2 &point) const;

  /**
   * @brief Sample the original texture using pixel coordinates.
   */
  [[nodiscard]] glm::vec3 discreteSample(int x, int y) const;

  /**
   * @brief In case the coordinates go beyond the bounds of the texture , we wrap around .
   * In addition , sampling texels may return an interpolated value when u,v are converted to a (x ,y) non
   * integer texture coordinate.
   */
  template<class D>
  glm::vec3 uvSample(D u, D v) const;
  template<typename D>
  void launchAsyncDiffuseIrradianceCompute(D delta, float *f_data, unsigned width_begin, unsigned width_end, unsigned _width, unsigned _height) const;
  [[nodiscard]] std::unique_ptr<TextureData> computeDiffuseIrradiance(unsigned _width, unsigned _height, unsigned delta, bool gpu) const;

  [[nodiscard]] glm::vec3 computeIrradianceSingleTexel(
      unsigned x, unsigned y, unsigned samples, const glm::vec3 &tangent, const glm::vec3 &bitangent, const glm::vec3 &normal) const;

  template<class D>
  glm::vec3 computeIrradianceImportanceSampling(D x, D y, D z, unsigned _width, unsigned _height, unsigned total_samples) const;
};
using HdrEnvmapProcessing = TextureOperations<float>;

template<class T>
TextureOperations<T>::TextureOperations(const std::vector<T> &_data, const unsigned _width, const unsigned _height, const unsigned int num_channels)
    : data(&_data) {
  if (!isValidDim(_width) || !isValidDim(_height))
    throw TextureInvalidDimensionsException();
  width = _width;
  height = _height;
  channels = num_channels;
}

template<class T>
TextureOperations<T>::TextureOperations(const TextureOperations &copy) {
  if (this != &copy) {
    data = copy.data;
    width = copy.width;
    height = copy.height;
    channels = copy.channels;
  }
}
template<class T>
TextureOperations<T>::TextureOperations(TextureOperations &&move) noexcept {
  if (this != &move) {
    data = move.data;
    width = move.width;
    height = move.height;
    channels = move.channels;
  }
}

template<class T>
TextureOperations<T> &TextureOperations<T>::operator=(const TextureOperations &copy) {
  if (this != &copy) {
    data = copy.data;
    width = copy.width;
    height = copy.height;
    channels = copy.channels;
  }
  return *this;
}

template<class T>
TextureOperations<T> &TextureOperations<T>::operator=(TextureOperations &&move) noexcept {
  if (this != &move) {
    data = move.data;
    width = move.width;
    height = move.height;
    channels = move.channels;
  }
  return *this;
}

template<class T>
template<class F, class... Args>
void TextureOperations<T>::processTexture(T *dest, int dest_width, int dest_height, F &&func, Args &&...args) const {
  for (int j = 0; j < dest_height; j++)
    for (int i = 0; i < dest_width; i++) {
      double u = math::texture::pixelToUv(i, dest_width);
      double v = math::texture::pixelToUv(j, dest_height);
      glm::vec3 col = uvSample(u, v);
      int idx = (j * dest_width + i) * channels;
      col.r = func(col.r, std::forward<Args>(args)...);
      col.g = func(col.g, std::forward<Args>(args)...);
      col.b = func(col.b, std::forward<Args>(args)...);
      dest[idx] = col.r;
      dest[idx + 1] = col.g;
      dest[idx + 2] = col.b;
    }
}

template<class T>
glm::vec3 TextureOperations<T>::discreteSample(int x, int y) const {
  const glm::vec2 normalized = wrapAroundPixelCoords(x, y);
  int index = (static_cast<int>(normalized.y) * width + static_cast<int>(normalized.x)) * channels;
  const float r = (*data)[index];
  const float g = (*data)[index + 1];
  const float b = (*data)[index + 2];
  return {r, g, b};
}

template<class T>
glm::vec3 TextureOperations<T>::bilinearInterpolate(const glm::vec2 &top_left,
                                                    const glm::vec2 &top_right,
                                                    const glm::vec2 &bottom_left,
                                                    const glm::vec2 &bottom_right,
                                                    const glm::vec2 &point) const {
  const float u = (point.x - top_left.x) / (top_right.x - top_left.x);
  const float v = (point.y - top_left.y) / (bottom_left.y - top_left.y);
  const float horizontal_diff = (1 - u);
  const glm::vec3 tl = discreteSample(top_left.x, top_left.y);
  const glm::vec3 tr = discreteSample(top_right.x, top_right.y);
  const glm::vec3 bl = discreteSample(bottom_left.x, bottom_left.y);
  const glm::vec3 br = discreteSample(bottom_right.x, bottom_right.y);
  const glm::vec3 top_interp = horizontal_diff * tl + u * tr;
  const glm::vec3 bot_interp = horizontal_diff * bl + u * br;
  return (1 - v) * top_interp + v * bot_interp;
}

template<class T>
template<class D>
glm::vec3 TextureOperations<T>::uvSample(const D u, const D v) const {
  const glm::vec2 wrap_uv = wrapAroundTexCoords(u, v);
  const glm::vec2 pixel_coords = glm::dvec2(math::texture::uvToPixel(wrap_uv.x, width), math::texture::uvToPixel(wrap_uv.y, height));
  const glm::vec2 top_left(std::floor(pixel_coords.x), std::floor(pixel_coords.y));
  const glm::vec2 top_right(std::floor(pixel_coords.x) + 1, std::floor(pixel_coords.y));
  const glm::vec2 bottom_left(std::floor(pixel_coords.x), std::floor(pixel_coords.y) + 1);
  const glm::vec2 bottom_right(std::floor(pixel_coords.x) + 1, std::floor(pixel_coords.y) + 1);
  const glm::vec3 texel_value = bilinearInterpolate(top_left, top_right, bottom_left, bottom_right, pixel_coords);
  return texel_value;
}

template<class T>
template<class D>
glm::vec2 TextureOperations<T>::wrapAroundTexCoords(const D u, const D v) const {
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
glm::vec2 TextureOperations<T>::wrapAroundPixelCoords(const int x, const int y) const {
  int x_coord = 0, y_coord = 0;
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
template<typename D>
void TextureOperations<T>::launchAsyncDiffuseIrradianceCompute(
    const D delta, float *f_data, const unsigned width_begin, const unsigned width_end, const unsigned _width, const unsigned _height) const {
  for (unsigned i = width_begin; i <= width_end; i++) {
    for (unsigned j = 0; j < _height; j++) {
      glm::vec2 uv = glm::vec2(math::texture::pixelToUv(i, _width), math::texture::pixelToUv(j, _height));
      const glm::vec2 sph = math::spherical::uvToSpherical(uv.x, uv.y);
      const glm::vec3 cart = math::spherical::sphericalToCartesian(sph.x, sph.y);
      glm::vec3 irrad = computeIrradianceImportanceSampling(cart.x, cart.y, cart.z, _width, _height, delta);
      unsigned index = (j * _width + i) * channels;
      f_data[index] = irrad.x;
      f_data[index + 1] = irrad.y;
      f_data[index + 2] = irrad.z;
    }
  }
}
template<class T>
glm::vec3 TextureOperations<T>::computeIrradianceSingleTexel(
    const unsigned x, const unsigned y, const unsigned samples, const glm::vec3 &tangent, const glm::vec3 &bitangent, const glm::vec3 &normal) const {
  glm::vec3 random = math::importance_sampling::pgc3d(x, y, samples);
  float phi = 2 * PI * random.x;
  float theta = asin(sqrt(random.y));
  glm::vec3 uv_cart = math::spherical::sphericalToCartesian(phi, theta);
  uv_cart = uv_cart.x * tangent + uv_cart.y * bitangent + uv_cart.z * normal;
  auto spherical = math::spherical::cartesianToSpherical(uv_cart.x, uv_cart.y, uv_cart.z);
  glm::vec2 uvt = math::spherical::sphericalToUv(spherical.x, spherical.y);
  return uvSample(uvt.x, uvt.y);
}

template<class T>
template<class D>
glm::vec3 TextureOperations<T>::computeIrradianceImportanceSampling(
    const D x, const D y, const D z, const unsigned _width, const unsigned _height, const unsigned total_samples) const {
  unsigned int samples = 0;
  glm::vec3 irradiance{0};
  glm::vec3 normal{x, y, z};
  glm::vec3 right_vec{1.f, 0.f, 0.f};
  glm::vec2 uv{math::spherical::sphericalToUv(math::spherical::cartesianToSpherical(x, y, z))};
  const unsigned pixel_x = math::texture::uvToPixel(uv.x, _width);
  const unsigned pixel_y = math::texture::uvToPixel(uv.y, _height);
  float dd = glm::dot(right_vec, normal);
  glm::vec3 tangent{0.0, 1.0, 0.0};
  if (1.0 - abs(dd) > math::epsilon)
    tangent = glm::normalize(glm::cross(right_vec, normal));
  glm::vec3 bitangent = glm::cross(normal, tangent);
  for (samples = 0; samples < total_samples; samples++)
    irradiance += computeIrradianceSingleTexel(pixel_x, pixel_y, samples, tangent, bitangent, normal);
  return irradiance / static_cast<float>(total_samples);
}

template<class T>
TextureOperations<T>::~TextureOperations() = default;

#endif

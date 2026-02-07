#ifndef NOVATEXTUREINTERFACE_H
#define NOVATEXTUREINTERFACE_H
#include "TextureContext.h"
#include "texture_datastructures.h"
#include <internal/common/math/math_utils.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/macro/project_macros.h>
#include <internal/memory/tag_ptr.h>
namespace nova::texturing {

  /* returns uv in [0 , 1] interval*/
  ax_device_callable_inlined float wrap(float uv) {
    if (uv < 0.f) {
      float float_part = ceilf(uv) - uv;
      return 1.f - float_part;
    }
    if (uv > 1.f) {
      float float_part = uv - floorf(uv);
      return float_part;
    }
    return uv;
  }

  class ConstantTexture {
    glm::vec4 albedo{};

   public:
    ax_device_callable ConstantTexture() = default;
    ax_device_callable ConstantTexture(const glm::vec4 &albedo_) : albedo(albedo_) {}
    ax_device_callable glm::vec4 sample(float /*u*/, float /*v*/, const texture_data_aggregate_s & /*sample_data*/) const { return albedo; }
  };

  template<class T>
  class ImageTexture;

  template<>
  class ImageTexture<float> {
    std::size_t texture_index{0};

   public:
    ax_device_callable ImageTexture() = default;

    ax_device_callable ImageTexture(std::size_t index) : texture_index(index) {}

    ax_device_callable glm::vec4 sample(float u, float v, const texture_data_aggregate_s &sample_data) const {
      u = wrap(u);
      v = wrap(v);

      return sample_data.texture_ctx->f32pixel(texture_index, u, v);
    }

    std::size_t getTextureIndex() const { return texture_index; }
  };

  template<>
  class ImageTexture<uint32_t> {
    std::size_t texture_index{0};

    ax_device_callable_inlined glm::vec4 getPixelFromFormat(uint32_t pixel_value, bool is_rgba, unsigned channels) const {
      union FORMAT {
        uint32_t rgba;
        struct {
          uint8_t r;
          uint8_t g;
          uint8_t b;
          uint8_t a;
        };
      };
      FORMAT pixel{};
      pixel.rgba = pixel_value;

      uint8_t r;
      uint8_t g;
      uint8_t b;
      uint8_t a;

      if (is_rgba) {
        r = pixel.r;
        g = pixel.g;
        b = pixel.b;
        a = pixel.a;
      } else {
        r = pixel.b;
        g = pixel.g;
        b = pixel.r;
        a = pixel.a;
      }

      switch (channels) {
        case 1:
          return {r, 0, 0, 1.f};
        case 2:
          return {r, g, 0, 1.f};
        case 3:
          return {r, g, b, 1.f};
        case 4:
          return {r, g, b, a};
        default:
          AX_UNREACHABLE
          break;
      }
      AX_UNREACHABLE;
    }

    ax_device_callable_inlined glm::vec4 bilinearInterpolate(const uint32_t pixels[4],
                                                             float tx,
                                                             float ty,
                                                             const texture_data_aggregate_s &sample_data) const {

      bool is_rgba = sample_data.texture_ctx->u32IsRGBA(texture_index);
      int channels = sample_data.texture_ctx->u32channels(texture_index);
      glm::vec4 v0 = getPixelFromFormat(pixels[0], is_rgba, channels);
      glm::vec4 v1 = getPixelFromFormat(pixels[1], is_rgba, channels);
      glm::vec4 v2 = getPixelFromFormat(pixels[2], is_rgba, channels);
      glm::vec4 v3 = getPixelFromFormat(pixels[3], is_rgba, channels);

      float horizontal_diff = 1 - tx;
      glm::vec4 top_interp = horizontal_diff * v0 + tx * v1;
      glm::vec4 bot_interp = horizontal_diff * v2 + tx * v3;
      return (1 - ty) * top_interp + ty * bot_interp;
    }

   public:
    ax_device_callable_inlined ImageTexture() = default;

    ax_device_callable_inlined ImageTexture(std::size_t index) : texture_index(index) {}

    ax_device_callable_inlined glm::vec4 sample(float u, float v, const texture_data_aggregate_s &sample_data) const {
      u = wrap(u);
      v = wrap(v);
#ifndef __CUDA_ARCH__
      uint32_t pixels[4];
      float tx{}, ty{};
      sample_data.texture_ctx->u32pixel(texture_index, u, v, tx, ty, pixels);
      return bilinearInterpolate(pixels, tx, ty, sample_data);
#else
      return getPixelFromFormat(sample_data.texture_ctx->u32pixel(texture_index, u, v),
                                sample_data.texture_ctx->u32IsRGBA(texture_index),
                                sample_data.texture_ctx->u32channels(texture_index));
#endif
    }

    ax_device_callable_inlined size_t getTextureIndex() const { return texture_index; }
  };

  ax_device_callable_inlined glm::vec3 sample_cubemap_plane(const glm::vec3 &sample_vector, const glm::vec3 &up_vector) {
    const glm::vec3 same_plane_vector = glm::dot(sample_vector, up_vector) * sample_vector;
    return same_plane_vector;
  }

  class EnvmapTexture {
    std::size_t texture_index{0};

   public:
    ax_device_callable EnvmapTexture() = default;
    ax_device_callable EnvmapTexture(std::size_t id_) : texture_index(id_) {}

    /* Pass the direction vector for sampling inside sample_data.
     * parameters u,v are not used here. */
    ax_device_callable glm::vec4 sample(float /*u*/, float /*v*/, const texture_data_aggregate_s &data) const {
      /* Compute UV coordinates from the cartesian sampling vector. */
      glm::vec3 normalized = glm::normalize(data.geometric_data.sampling_vector);
      const glm::vec2 sph = math::spherical::cartesianToSpherical(normalized);
      const glm::vec2 uv = math::spherical::sphericalToUv(sph);
      const float u = wrap(uv.x);
      const float v = wrap(uv.y);
      const TextureCtx *texture_ctx = data.texture_ctx;
      AX_ASSERT_NOTNULL(texture_ctx);
      return texture_ctx->f32pixel(texture_index, u, v);
    }

    std::size_t getTextureIndex() const { return texture_index; }
  };

  class NovaTextureInterface : public core::tag_ptr<ConstantTexture, ImageTexture<uint32_t>, ImageTexture<float>, EnvmapTexture> {
   public:
    using tag_ptr::tag_ptr;
    ax_device_callable glm::vec4 sample(float u, float v, const texture_data_aggregate_s &sample_data) const {
      auto disp = [&](auto texture) { return texture->sample(u, v, sample_data); };
      return dispatch(disp);
    }
  };

  using TYPELIST = core::type_list<ConstantTexture, ImageTexture<uint32_t>, ImageTexture<float>, EnvmapTexture>;

}  // namespace nova::texturing

using EnvMapCollection = axstd::span<const nova::texturing::EnvmapTexture>;
using ImgTexCollection = axstd::span<const nova::texturing::ImageTexture<uint32_t>>;
using ImgF32TexCollection = axstd::span<const nova::texturing::ImageTexture<float>>;
using CstTexCollection = axstd::span<const nova::texturing::ConstantTexture>;
using IntfTexCollection = axstd::span<const nova::texturing::NovaTextureInterface>;

#endif  // NOVATEXTUREINTERFACE_H

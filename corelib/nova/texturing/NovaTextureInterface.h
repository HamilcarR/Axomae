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

  class ConstantTexture;
  class ImageTexture;
  class EnvmapTexture;

  class ConstantTexture {
    glm::vec4 albedo{};

   public:
    ax_device_callable ConstantTexture() = default;
    ax_device_callable ConstantTexture(const glm::vec4 &albedo_) : albedo(albedo_) {}
    ax_device_callable glm::vec4 sample(float /*u*/, float /*v*/, const texture_data_aggregate_s & /*sample_data*/) const { return albedo; }
  };

  class ImageTexture {
    std::size_t texture_index{0};

   public:
    ax_device_callable ImageTexture() = default;

    ax_device_callable ImageTexture(std::size_t index) : texture_index(index) {}

    ax_device_callable glm::vec4 sample(float u, float v, const texture_data_aggregate_s &sample_data) const {
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
      pixel.rgba = sample_data.texture_ctx->u32pixel(texture_index, u, v);

      uint8_t r;
      uint8_t g;
      uint8_t b;
      uint8_t a;
      bool is_rgba = sample_data.texture_ctx->isRgba();

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
      int channels = 4;  // sample_data.texture_ctx->u32channels(texture_index);
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
      AX_UNREACHABLE
      return glm::vec4(0);
    }
  };

  /* returns uv in [0 , 1] interval*/
  ax_device_callable_inlined float normalize_uv(float uv) {
    if (uv < 0) {
      float float_part = std::ceil(uv) - uv;
      return 1.f - float_part;
    }
    return uv;
  }

  ax_device_callable_inlined glm::vec3 sample_cubemap_plane(const glm::vec3 &sample_vector, const glm::vec3 &up_vector) {
    const glm::vec3 same_plane_vector = glm::dot(sample_vector, up_vector) * sample_vector;
    return same_plane_vector;
  }

  class EnvmapTexture {
    const float *image_buffer{};
    int width{}, height{}, channels{};

   public:
    ax_device_callable EnvmapTexture() = default;
    ax_device_callable EnvmapTexture(const float *buf, int w, int h, int channels) : image_buffer(buf), width(w), height(h), channels(channels) {}

    /* Pass the direction vector for sampling inside sample_data.
     * parameters u,v are not used here. */
    ax_device_callable glm::vec4 sample(float /*u*/, float /*v*/, const texture_data_aggregate_s &data) const {
      glm::vec3 sample_vector = data.geometric_data.sampling_vector;
      glm::vec3 normalized = glm::normalize(sample_vector);
      std::swap(normalized.y, normalized.z);
      normalized.z = -normalized.z;
      const glm::vec2 sph = math::spherical::cartesianToSpherical(normalized);
      const glm::vec2 uv = math::spherical::sphericalToUv(sph);
      const float u = normalize_uv(uv.x);
      const float v = normalize_uv(uv.y);

      const int x = (int)math::texture::uvToPixel(u, width - 1);
      const int y = (int)math::texture::uvToPixel(1 - v, height - 1);

      int index = (y * width + x) * channels;
      AX_ASSERT_LT(index + 2, width * height * channels);
      return {image_buffer[index], image_buffer[index + 1], image_buffer[index + 2], 1.f};
    }

    ax_device_callable F32Texture getRawData() const {
      F32Texture data{};
      data.width = width;
      data.height = height;
      data.channels = channels;
      data.raw_data = axstd::span(image_buffer, width * height * channels);
      return data;
    }
  };

  class NovaTextureInterface : public core::tag_ptr<ConstantTexture, ImageTexture, EnvmapTexture> {
   public:
    using tag_ptr::tag_ptr;
    ax_device_callable glm::vec4 sample(float u, float v, const texture_data_aggregate_s &sample_data) const {
      auto disp = [&](auto texture) { return texture->sample(u, v, sample_data); };
      return dispatch(disp);
    }
  };

  using TYPELIST = core::type_list<ConstantTexture, ImageTexture, EnvmapTexture>;

}  // namespace nova::texturing

#endif  // NOVATEXTUREINTERFACE_H

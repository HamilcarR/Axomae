#ifndef NOVATEXTUREINTERFACE_H
#define NOVATEXTUREINTERFACE_H
#include "internal/device/gpgpu/device_utils.h"
#include "math_utils.h"
#include "project_macros.h"
#include "tag_ptr.h"

namespace nova::texturing {
  struct texture_sample_data {
    glm::vec3 p;
  };

  struct TextureRawData {
    float *raw_data;  // TODO : const?
    int width;
    int height;
    int channels;
  };

  class ConstantTexture;
  class ImageTexture;

  class NovaTextureInterface : public core::tag_ptr<ConstantTexture, ImageTexture> {
   public:
    using tag_ptr::tag_ptr;
    AX_DEVICE_CALLABLE [[nodiscard]] glm::vec4 sample(float u, float v, const texture_sample_data &sample_data) const;
  };

  class ConstantTexture {
   private:
    glm::vec4 albedo{};

   public:
    AX_DEVICE_CALLABLE ConstantTexture() = default;
    AX_DEVICE_CALLABLE explicit ConstantTexture(const glm::vec4 &albedo);
    AX_DEVICE_CALLABLE [[nodiscard]] glm::vec4 sample(float u, float v, const texture_sample_data &sample_data) const;
  };

  class ImageTexture {
    const uint32_t *image_buffer{};
    int width{}, height{}, channels{};
    bool is_rgba{false};  // or bgra

   public:
    AX_DEVICE_CALLABLE ImageTexture() = default;
    AX_DEVICE_CALLABLE ImageTexture(const uint32_t *raw_image, int w, int h, int chans, bool isrgba = false);
    AX_DEVICE_CALLABLE [[nodiscard]] glm::vec4 sample(float u, float v, const texture_sample_data &sample_data) const;
  };

  using TYPELIST = core::type_list<ConstantTexture, ImageTexture>;

}  // namespace nova::texturing
#endif  // NOVATEXTUREINTERFACE_H

#ifndef NOVATEXTUREINTERFACE_H
#define NOVATEXTUREINTERFACE_H
#include "texture_datastructures.h"
#include <internal/common/math/math_utils.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/macro/project_macros.h>
#include <internal/memory/tag_ptr.h>
namespace nova::texturing {

  class ConstantTexture;
  class ImageTexture;
  class EnvmapTexture;

  class NovaTextureInterface : public core::tag_ptr<ConstantTexture, ImageTexture, EnvmapTexture> {
   public:
    using tag_ptr::tag_ptr;
    ax_device_callable glm::vec4 sample(float u, float v, const texture_data_aggregate_s &sample_data) const;
  };

  class ConstantTexture {
    glm::vec4 albedo{};

   public:
    ax_device_callable ConstantTexture() = default;
    ax_device_callable explicit ConstantTexture(const glm::vec4 &albedo);
    ax_device_callable glm::vec4 sample(float u, float v, const texture_data_aggregate_s &sample_data) const;
  };

  class ImageTexture {
    std::size_t texture_index{0};

   public:
    ax_device_callable ImageTexture() = default;
    ax_device_callable ImageTexture(std::size_t id);
    ax_device_callable glm::vec4 sample(float u, float v, const texture_data_aggregate_s &sample_data) const;
  };

  class EnvmapTexture {
    const float *image_buffer{};
    int width{}, height{}, channels{};

   public:
    ax_device_callable EnvmapTexture() = default;
    ax_device_callable EnvmapTexture(const float *buffer, int w, int h, int channels);
    /* Pass the direction vector for sampling inside sample_data.
     * parameters u,v are not used here. */
    ax_device_callable glm::vec4 sample(float /*u*/, float /*v*/, const texture_data_aggregate_s &sample_data) const;
    ax_device_callable F32Texture getRawData() const;
  };

  using TYPELIST = core::type_list<ConstantTexture, ImageTexture, EnvmapTexture>;

}  // namespace nova::texturing

#endif  // NOVATEXTUREINTERFACE_H

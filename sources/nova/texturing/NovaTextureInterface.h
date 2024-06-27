#ifndef NOVATEXTUREINTERFACE_H
#define NOVATEXTUREINTERFACE_H
#include "Axomae_macros.h"
#include "math_utils.h"
namespace nova::texturing {
  struct texture_sample_data {
    glm::vec3 p;
  };

  struct TextureRawData {
    std::vector<float> *raw_data;
    int width;
    int height;
    int channels;
  };
  class NovaTextureInterface {
   public:
    virtual ~NovaTextureInterface() = default;
    [[nodiscard]] virtual glm::vec4 sample(float u, float v, const texture_sample_data &sample_data) const = 0;
  };
}  // namespace nova::texturing
#endif  // NOVATEXTUREINTERFACE_H

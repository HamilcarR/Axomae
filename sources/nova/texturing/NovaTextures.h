#ifndef NOVATEXTURES_H
#define NOVATEXTURES_H
#include "Axomae_macros.h"
#include "math_utils.h"

namespace nova::texturing {

  struct texture_sample_data {
    glm::vec3 p;
  };

  class NovaTextureInterface {
   public:
    virtual ~NovaTextureInterface() = default;
    [[nodiscard]] virtual glm::vec4 sample(float u, float v, const texture_sample_data &sample_data) const = 0;
  };

  class ConstantTexture : public NovaTextureInterface {
   private:
    glm::vec4 albedo{};

   public:
    CLASS_OCM(ConstantTexture)

    explicit ConstantTexture(const glm::vec4 &albedo);
    [[nodiscard]] glm::vec4 sample(float u, float v, const texture_sample_data &sample_data) const override;
  };

  class CheckerTexture : public NovaTextureInterface {
   private:
  };

}  // namespace nova::texturing
#endif  // NOVATEXTURES_H

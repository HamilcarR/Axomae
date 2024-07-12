#ifndef NOVATEXTURES_H
#define NOVATEXTURES_H
#include "NovaTextureInterface.h"
#include "math_utils.h"
#include "project_macros.h"

namespace nova::texturing {

  class ConstantTexture final : public NovaTextureInterface {
   private:
    glm::vec4 albedo{};

   public:
    CLASS_OCM(ConstantTexture)

    explicit ConstantTexture(const glm::vec4 &albedo);
    [[nodiscard]] glm::vec4 sample(float u, float v, const texture_sample_data &sample_data) const override;
  };

}  // namespace nova::texturing
#endif  // NOVATEXTURES_H

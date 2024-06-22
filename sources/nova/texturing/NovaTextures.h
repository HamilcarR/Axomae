#ifndef NOVATEXTURES_H
#define NOVATEXTURES_H
#include "nova_texturing.h"

namespace nova::texturing {
  class ConstantTexture : public NovaTextureInterface {
   private:
    glm::vec4 albedo{};

   public:
    ConstantTexture() = default;
    explicit ConstantTexture(const glm::vec4 &albedo);
    ~ConstantTexture() override = default;
    ConstantTexture(const ConstantTexture &other) = default;
    ConstantTexture(ConstantTexture &&other) noexcept = default;
    ConstantTexture &operator=(const ConstantTexture &other) = default;
    ConstantTexture &operator=(ConstantTexture &&other) noexcept = default;
    [[nodiscard]] glm::vec4 sample(float u, float v, const texture_sample_data &sample_data) const override;
  };
}  // namespace nova::texturing
#endif  // NOVATEXTURES_H

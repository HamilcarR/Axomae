#ifndef NOVATEXTURES_H
#define NOVATEXTURES_H
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

#ifndef NOVA_MATERIAL_H
#define NOVA_MATERIAL_H
#include "Axomae_macros.h"
#include "Ray.h"
#include "nova_utils.h"
#include "ray/Hitable.h"
#include "texturing/nova_texturing.h"

#include <memory>

/* Lambertian shading */

namespace nova {
  class Ray;
  namespace material {
    class NovaMaterialInterface {
     public:
      virtual ~NovaMaterialInterface() = default;
      virtual bool scatter(const Ray &in, Ray &out, hit_data &hit_d) const = 0;

      template<class SUBTYPE, class... Args>
      static std::unique_ptr<SUBTYPE> create(Args &&...args) {
        ASSERT_SUBTYPE(NovaMaterialInterface, SUBTYPE);
        return std::make_unique<SUBTYPE>(std::forward<Args>(args)...);
      }
    };

    class NovaDiffuseMaterial final : public NovaMaterialInterface {
     private:
      std::unique_ptr<texturing::NovaTextureInterface> albedo;

     public:
      explicit NovaDiffuseMaterial(const glm::vec4 &col);
      ~NovaDiffuseMaterial() override = default;
      NovaDiffuseMaterial(const NovaDiffuseMaterial &other) = delete;
      NovaDiffuseMaterial(NovaDiffuseMaterial &&other) noexcept = default;
      NovaDiffuseMaterial &operator=(const NovaDiffuseMaterial &other) = delete;
      NovaDiffuseMaterial &operator=(NovaDiffuseMaterial &&other) noexcept = default;
      bool scatter(const Ray &in, Ray &out, hit_data &hit_d) const override;
    };

    class NovaConductorMaterial final : public NovaMaterialInterface {
     private:
      std::unique_ptr<texturing::NovaTextureInterface> albedo;
      float fuzz{};

     public:
      explicit NovaConductorMaterial(const glm::vec4 &color);
      NovaConductorMaterial(const glm::vec4 &color, float fuzz_);
      ~NovaConductorMaterial() override = default;
      NovaConductorMaterial(const NovaConductorMaterial &other) = delete;
      NovaConductorMaterial(NovaConductorMaterial &&other) noexcept = default;
      NovaConductorMaterial &operator=(const NovaConductorMaterial &other) = delete;
      NovaConductorMaterial &operator=(NovaConductorMaterial &&other) noexcept = default;
      bool scatter(const Ray &in, Ray &out, hit_data &hit_d) const override;
    };

    class NovaDielectricMaterial final : public NovaMaterialInterface {
     private:
      std::unique_ptr<texturing::NovaTextureInterface> albedo;
      float eta;  // ior

     public:
      explicit NovaDielectricMaterial(const glm::vec4 &color);
      NovaDielectricMaterial(const glm::vec4 &color, float ior);
      ~NovaDielectricMaterial() override = default;
      NovaDielectricMaterial(const NovaDielectricMaterial &other) = delete;
      NovaDielectricMaterial(NovaDielectricMaterial &&other) noexcept = default;
      NovaDielectricMaterial &operator=(const NovaDielectricMaterial &other) = delete;
      NovaDielectricMaterial &operator=(NovaDielectricMaterial &&other) noexcept = default;

      bool scatter(const Ray &in, Ray &out, hit_data &hit_d) const override;
    };
  }  // namespace material
}  // namespace nova
#endif

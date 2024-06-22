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
      texturing::NovaTextureInterface *albedo;

     public:
      explicit NovaDiffuseMaterial(texturing::NovaTextureInterface *texture);
      ~NovaDiffuseMaterial() override = default;
      NovaDiffuseMaterial(const NovaDiffuseMaterial &other) = default;
      NovaDiffuseMaterial(NovaDiffuseMaterial &&other) noexcept = default;
      NovaDiffuseMaterial &operator=(const NovaDiffuseMaterial &other) = default;
      NovaDiffuseMaterial &operator=(NovaDiffuseMaterial &&other) noexcept = default;
      bool scatter(const Ray &in, Ray &out, hit_data &hit_d) const override;
    };

    class NovaConductorMaterial final : public NovaMaterialInterface {
     private:
      texturing::NovaTextureInterface *albedo;
      float fuzz{};

     public:
      explicit NovaConductorMaterial(texturing::NovaTextureInterface *texture);
      NovaConductorMaterial(texturing::NovaTextureInterface *texture, float fuzz_);
      ~NovaConductorMaterial() override = default;
      NovaConductorMaterial(const NovaConductorMaterial &other) = default;
      NovaConductorMaterial(NovaConductorMaterial &&other) noexcept = default;
      NovaConductorMaterial &operator=(const NovaConductorMaterial &other) = default;
      NovaConductorMaterial &operator=(NovaConductorMaterial &&other) noexcept = default;
      bool scatter(const Ray &in, Ray &out, hit_data &hit_d) const override;
    };

    class NovaDielectricMaterial final : public NovaMaterialInterface {
     private:
      texturing::NovaTextureInterface *albedo;
      float eta;  // ior

     public:
      explicit NovaDielectricMaterial(texturing::NovaTextureInterface *texture);
      NovaDielectricMaterial(texturing::NovaTextureInterface *texture, float ior);
      ~NovaDielectricMaterial() override = default;
      NovaDielectricMaterial(const NovaDielectricMaterial &other) = default;
      NovaDielectricMaterial(NovaDielectricMaterial &&other) noexcept = default;
      NovaDielectricMaterial &operator=(const NovaDielectricMaterial &other) = default;
      NovaDielectricMaterial &operator=(NovaDielectricMaterial &&other) noexcept = default;

      bool scatter(const Ray &in, Ray &out, hit_data &hit_d) const override;
    };
  }  // namespace material
}  // namespace nova
#endif

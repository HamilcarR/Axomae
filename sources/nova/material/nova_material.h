#ifndef NOVA_MATERIAL_H
#define NOVA_MATERIAL_H
#include "Axomae_macros.h"
#include "Hitable.h"
#include "Ray.h"
#include "nova_utils.h"
#include <memory>

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
      glm::vec4 albedo;

     public:
      explicit NovaDiffuseMaterial(const glm::vec4 &col) : albedo(col) {}
      ~NovaDiffuseMaterial() override = default;
      NovaDiffuseMaterial(const NovaDiffuseMaterial &other) = default;
      NovaDiffuseMaterial(NovaDiffuseMaterial &&other) noexcept = default;
      NovaDiffuseMaterial &operator=(const NovaDiffuseMaterial &other) = default;
      NovaDiffuseMaterial &operator=(NovaDiffuseMaterial &&other) noexcept = default;
      bool scatter(const Ray &in, Ray &out, hit_data &hit_d) const override;
    };

    class NovaConductorMaterial final : public NovaMaterialInterface {
     private:
      glm::vec4 albedo;
      float fuzz{};

     public:
      explicit NovaConductorMaterial(const glm::vec4 &color) : albedo(color) {}
      NovaConductorMaterial(const glm::vec4 &color, float fuzz_) : albedo(color), fuzz(fuzz_) {}
      ~NovaConductorMaterial() override = default;
      NovaConductorMaterial(const NovaConductorMaterial &other) = default;
      NovaConductorMaterial(NovaConductorMaterial &&other) noexcept = default;
      NovaConductorMaterial &operator=(const NovaConductorMaterial &other) = default;
      NovaConductorMaterial &operator=(NovaConductorMaterial &&other) noexcept = default;
      bool scatter(const Ray &in, Ray &out, hit_data &hit_d) const override;
    };

    class NovaDielectricMaterial final : public NovaMaterialInterface {
     private:
      glm::vec4 albedo;
      float n1, n2;  // ior
    };
  }  // namespace material
}  // namespace nova
#endif

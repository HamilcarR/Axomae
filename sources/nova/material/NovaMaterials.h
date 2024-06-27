#ifndef NOVAMATERIALS_H
#define NOVAMATERIALS_H
#include "Axomae_macros.h"
#include "ray/Hitable.h"
#include "texturing/nova_texturing.h"
namespace nova {
  class Ray;
}

namespace nova::material {
  class NovaMaterialInterface {
   public:
    virtual ~NovaMaterialInterface() = default;
    virtual bool scatter(const Ray &in, Ray &out, hit_data &hit_d) const = 0;
  };

  class NovaDiffuseMaterial final : public NovaMaterialInterface {
   private:
    texturing::NovaTextureInterface *albedo{};

   public:
    CLASS_OCM(NovaDiffuseMaterial)

    explicit NovaDiffuseMaterial(texturing::NovaTextureInterface *texture);
    bool scatter(const Ray &in, Ray &out, hit_data &hit_d) const override;
  };

  class NovaConductorMaterial final : public NovaMaterialInterface {
   private:
    texturing::NovaTextureInterface *albedo{};
    float fuzz{};

   public:
    CLASS_OCM(NovaConductorMaterial)

    explicit NovaConductorMaterial(texturing::NovaTextureInterface *texture);
    NovaConductorMaterial(texturing::NovaTextureInterface *texture, float fuzz_);
    bool scatter(const Ray &in, Ray &out, hit_data &hit_d) const override;
  };

  class NovaDielectricMaterial final : public NovaMaterialInterface {
   private:
    texturing::NovaTextureInterface *albedo{};
    float eta{};  // ior

   public:
    CLASS_OCM(NovaDielectricMaterial)

    explicit NovaDielectricMaterial(texturing::NovaTextureInterface *texture);
    NovaDielectricMaterial(texturing::NovaTextureInterface *texture, float ior);
    bool scatter(const Ray &in, Ray &out, hit_data &hit_d) const override;
  };
}  // namespace nova::material
#endif  // NOVAMATERIALS_H

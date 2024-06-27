#ifndef NOVAMATERIALS_H
#define NOVAMATERIALS_H
#include "Axomae_macros.h"
#include "ray/Hitable.h"
#include "texturing/nova_texturing.h"
namespace nova {
  class Ray;
}

namespace nova::material {
  struct texture_pack {
    const texturing::ImageTexture *albedo;
    const texturing::ImageTexture *metallic;
    const texturing::ImageTexture *roughness;
    const texturing::ImageTexture *ao;
    const texturing::ImageTexture *normalmap;
    const texturing::ImageTexture *emissive;
  };
  class NovaMaterialInterface {
   public:
    virtual ~NovaMaterialInterface() = default;
    virtual bool scatter(const Ray &in,
                         Ray &out,
                         hit_data &hit_d) const = 0;  // replace by eval() , check metallic , roughness , emissive value and use correct function
  };

  class NovaDiffuseMaterial final : public NovaMaterialInterface {
   private:
    texture_pack t_pack{};

   public:
    CLASS_OCM(NovaDiffuseMaterial)

    explicit NovaDiffuseMaterial(const texture_pack &textures);
    bool scatter(const Ray &in, Ray &out, hit_data &hit_d) const override;
  };

  class NovaConductorMaterial final : public NovaMaterialInterface {
   private:
    texture_pack t_pack{};
    float fuzz{};

   public:
    CLASS_OCM(NovaConductorMaterial)

    explicit NovaConductorMaterial(const texture_pack &textures);
    NovaConductorMaterial(const texture_pack &textures, float fuzz_);
    bool scatter(const Ray &in, Ray &out, hit_data &hit_d) const override;
  };

  class NovaDielectricMaterial final : public NovaMaterialInterface {
   private:
    texture_pack t_pack{};
    float eta{};  // ior

   public:
    CLASS_OCM(NovaDielectricMaterial)

    explicit NovaDielectricMaterial(const texture_pack &textures);
    NovaDielectricMaterial(const texture_pack &textures, float ior);
    bool scatter(const Ray &in, Ray &out, hit_data &hit_d) const override;
  };
}  // namespace nova::material
#endif  // NOVAMATERIALS_H

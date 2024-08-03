#ifndef NOVAMATERIALS_H
#define NOVAMATERIALS_H
#include "project_macros.h"
#include "ray/Hitable.h"
#include "sampler/Sampler.h"
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

    CLASS_CM(texture_pack)
  };
  class NovaDiffuseMaterial;
  class NovaDielectricMaterial;
  class NovaConductorMaterial;

  class NovaMaterialInterface : public core::tag_ptr<NovaDiffuseMaterial, NovaDielectricMaterial, NovaConductorMaterial> {
   public:
    using tag_ptr::tag_ptr;
    /*replace by eval() , check metallic , roughness , emissive value and use correct function*/
    AX_DEVICE_CALLABLE bool scatter(const Ray &in, Ray &out, hit_data &hit_d, sampler::SamplerInterface &sampler) const;
  };

  class NovaDiffuseMaterial {
   private:
    texture_pack t_pack{};

   public:
    CLASS_CM(NovaDiffuseMaterial)

    explicit NovaDiffuseMaterial(const texture_pack &textures);
    AX_DEVICE_CALLABLE bool scatter(const Ray &in, Ray &out, hit_data &hit_d, sampler::SamplerInterface &sampler) const;
  };

  class NovaConductorMaterial {
   private:
    texture_pack t_pack{};
    float fuzz{};

   public:
    CLASS_CM(NovaConductorMaterial)

    explicit NovaConductorMaterial(const texture_pack &textures);
    NovaConductorMaterial(const texture_pack &textures, float fuzz_);
    AX_DEVICE_CALLABLE bool scatter(const Ray &in, Ray &out, hit_data &hit_d, sampler::SamplerInterface &sampler) const;
  };

  class NovaDielectricMaterial {
   private:
    texture_pack t_pack{};
    float eta{};  // ior

   public:
    CLASS_CM(NovaDielectricMaterial)

    explicit NovaDielectricMaterial(const texture_pack &textures);
    NovaDielectricMaterial(const texture_pack &textures, float ior);
    AX_DEVICE_CALLABLE bool scatter(const Ray &in, Ray &out, hit_data &hit_d, sampler::SamplerInterface &sampler) const;
  };

  using TYPELIST = NovaMaterialInterface::type_pack;

}  // namespace nova::material
#endif  // NOVAMATERIALS_H

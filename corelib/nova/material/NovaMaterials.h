#ifndef NOVAMATERIALS_H
#define NOVAMATERIALS_H
#include "material_datastructures.h"
#include <internal/common/axstd/span.h>
#include <internal/macro/project_macros.h>
#include <internal/memory/tag_ptr.h>

namespace nova {
  class Ray;
  struct hit_data;
  namespace sampler {
    class SamplerInterface;
  }
  namespace texturing {
    class ImageTexture;
    struct texture_data_aggregate_s;
  }  // namespace texturing
}  // namespace nova

namespace nova::material {
  struct texture_pack {
    const texturing::ImageTexture *albedo;
    const texturing::ImageTexture *metallic;
    const texturing::ImageTexture *roughness;
    const texturing::ImageTexture *ao;
    const texturing::ImageTexture *normalmap;
    const texturing::ImageTexture *emissive;
    const texturing::ImageTexture *specular;
    const texturing::ImageTexture *opacity;
    CLASS_DCM(texture_pack)
  };

  struct shading_data_s {
    texturing::texture_data_aggregate_s *texture_aggregate{nullptr};
  };

  class NovaDiffuseMaterial;
  class NovaDielectricMaterial;
  class NovaConductorMaterial;

  class NovaMaterialInterface : public core::tag_ptr<NovaDiffuseMaterial, NovaDielectricMaterial, NovaConductorMaterial> {
   public:
    using tag_ptr::tag_ptr;
    /*replace by eval() , check metallic , roughness , emissive value and use correct function*/
    ax_device_callable bool scatter(const Ray &in, Ray &out, hit_data &hit_d, sampler::SamplerInterface &sampler, shading_data_s &mat_ctx) const;
  };

  using NovaMatIntfView = axstd::span<NovaMaterialInterface>;
  using CstNovaMatIntfView = axstd::span<const NovaMaterialInterface>;

  class NovaDiffuseMaterial {
   private:
    texture_pack t_pack{};

   public:
    CLASS_CM(NovaDiffuseMaterial)

    explicit NovaDiffuseMaterial(const texture_pack &textures);
    ax_device_callable bool scatter(const Ray &in, Ray &out, hit_data &hit_d, sampler::SamplerInterface &sampler, shading_data_s &mat_ctx) const;
  };

  class NovaConductorMaterial {
   private:
    texture_pack t_pack{};
    float fuzz{};

   public:
    CLASS_CM(NovaConductorMaterial)

    explicit NovaConductorMaterial(const texture_pack &textures);
    NovaConductorMaterial(const texture_pack &textures, float fuzz_);
    ax_device_callable bool scatter(const Ray &in, Ray &out, hit_data &hit_d, sampler::SamplerInterface &sampler, shading_data_s &mat_ctx) const;
  };

  class NovaDielectricMaterial {
   private:
    texture_pack t_pack{};
    float eta{};  // ior

   public:
    CLASS_CM(NovaDielectricMaterial)

    explicit NovaDielectricMaterial(const texture_pack &textures);
    NovaDielectricMaterial(const texture_pack &textures, float ior);
    ax_device_callable bool scatter(const Ray &in, Ray &out, hit_data &hit_d, sampler::SamplerInterface &sampler, shading_data_s &mat_ctx) const;
  };

  using TYPELIST = NovaMaterialInterface::type_pack;

}  // namespace nova::material
#endif  // NOVAMATERIALS_H

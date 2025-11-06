#ifndef NOVAMATERIALS_H
#define NOVAMATERIALS_H
#include "BxDF.h"
#include "TexturePackSampler.h"
#include "glm/fwd.hpp"
#include "material/BSDF.h"
#include "material/BxDF_flags.h"
#include "material_datastructures.h"
#include "ray/Hitable.h"
#include "ray/IntersectFrame.h"
#include "ray/Ray.h"
#include "sampler/Sampler.h"
#include "spectrum/Spectrum.h"
#include "texturing/texture_datastructures.h"
#include <internal/common/axstd/span.h>
#include <internal/common/math/math_includes.h>
#include <internal/common/math/utils_3D.h>
#include <internal/debug/debug_utils.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/macro/project_macros.h>
#include <internal/memory/Allocator.h>
#include <internal/memory/tag_ptr.h>

namespace nova::material {

  struct shading_data_s {
    texturing::texture_data_aggregate_s *texture_aggregate{nullptr};
    // TODO: Add object Transform for World -> Object conversion ?
  };

  constexpr float INTERSECT_OFFSET = 1e-4f;

  ax_device_callable_inlined glm::vec3 hemi_sample(sampler::SamplerInterface &sampler) {
    float ret[2] = {};
    sampler.sample2D(ret);
    return bxdf::hemisphere_sample_uniform(ret);
  }

  ax_device_callable_inlined glm::vec3 compute_local_normal(float u,
                                                            float v,
                                                            const TexturePackSampler &t_pack,
                                                            texturing::texture_data_aggregate_s &sample_data) {
    return t_pack.normal(u, v, sample_data) * 2.f - 1.f;
  }
  ax_device_callable_inlined glm::vec3 compute_world_normal(const glm::vec3 &local_normal, const IntersectFrame &shading_frame) {
    return glm::normalize(shading_frame.localToWorld(local_normal));
  }

  class NovaDiffuseMaterial {
    texture_pack t_pack{};

   public:
    ax_device_callable_inlined NovaDiffuseMaterial() = default;

    ax_device_callable_inlined NovaDiffuseMaterial(const texture_pack &texture) : t_pack(texture) {}

    ax_device_callable_inlined bool scatter(const Ray &in,
                                            Ray & /*out*/,
                                            const intersection_record_s &hit_d,
                                            material_record_s &mat_rec,
                                            sampler::SamplerInterface &sampler,
                                            axstd::StaticAllocator64kb &allocator,
                                            shading_data_s &mat_ctx) const {
      AX_ASSERT_NOTNULL(mat_ctx.texture_aggregate);
      TexturePackSampler texture_pack_sampler(t_pack);
      const IntersectFrame &shading_frame = hit_d.shading_frame;
      glm::vec3 local_normal = compute_local_normal(hit_d.u, hit_d.v, texture_pack_sampler, *mat_ctx.texture_aggregate);
      glm::vec3 geometric_normal = shading_frame.getNormal();
      mat_rec = {};

      mat_rec.normal = compute_world_normal(local_normal, shading_frame);
      Spectrum R(glm::vec3(texture_pack_sampler.albedo(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate)));
      Spectrum E(glm::vec3(texture_pack_sampler.emissive(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate)));

      DiffuseBxDF bxdf(R);
      BSDF bsdf(&bxdf, mat_rec.normal, shading_frame.getTangent());
      BSDFSample lobe;
      float uc = sampler.sample1D();
      float u[2]{};
      sampler.sample2D(u);
      if (!bsdf.sample_f(in.direction, uc, u, &lobe))
        return false;
      mat_rec.attenuation = lobe.f;
      mat_rec.emissive = E;
      mat_rec.lobe = lobe;
      return true;
    }
  };

  class NovaConductorMaterial {
   private:
    texture_pack t_pack{};
    glm::vec3 tint, reflectivity;

   public:
    CLASS_CM(NovaConductorMaterial)

    ax_device_callable_inlined NovaConductorMaterial(const texture_pack &texture) : t_pack(texture) {
      reflectivity = glm::vec3(0.18, 0.42, 1.37);
      tint = glm::vec3(3.42, 2.35, 1.77);
    }

    ax_device_callable_inlined NovaConductorMaterial(const texture_pack &texture, const glm::vec3 &reflectivity, const glm::vec3 &tint)
        : t_pack(texture), reflectivity(reflectivity), tint(tint) {}

    ax_device_callable_inlined bool scatter(const Ray &in,
                                            Ray &out,
                                            const intersection_record_s &hit_d,
                                            material_record_s &mat_rec,
                                            sampler::SamplerInterface &sampler,
                                            axstd::StaticAllocator64kb &allocator,
                                            shading_data_s &mat_ctx) const {

      AX_ASSERT_NOTNULL(mat_ctx.texture_aggregate);
      TexturePackSampler texture_pack_sampler(t_pack);
      const IntersectFrame &shading_frame = hit_d.shading_frame;

      glm::vec3 perturbed_shading_normal = compute_local_normal(hit_d.u, hit_d.v, texture_pack_sampler, *mat_ctx.texture_aggregate);
      glm::vec3 perturbed_world_normal = compute_world_normal(perturbed_shading_normal, shading_frame);
      glm::vec3 geometric_world_normal = shading_frame.getNormal();
      mat_rec = {};
      mat_rec.normal = perturbed_world_normal;

      Spectrum R(glm::vec3(texture_pack_sampler.albedo(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate)));
      Spectrum E(glm::vec3(texture_pack_sampler.emissive(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate)));

      BSDFSample lobe;
      float uc = sampler.sample1D();
      float u[2]{};
      sampler.sample2D(u);
      ConductorBxDF bxdf(reflectivity, tint, 0.f);
      BSDF bsdf(&bxdf, mat_rec.normal, shading_frame.getTangent());
      if (!bsdf.sample_f(in.direction, uc, u, &lobe))
        return false;
      mat_rec.attenuation = lobe.f * R;
      mat_rec.emissive = E;
      mat_rec.lobe = lobe;

      return true;
    }
  };

  class NovaDielectricMaterial {
    texture_pack t_pack{};
    float eta{};  // ior

   public:
    ax_device_callable_inlined NovaDielectricMaterial(const texture_pack &texture) : t_pack(texture), eta(1.f) {}

    ax_device_callable_inlined NovaDielectricMaterial(const texture_pack &texture, float ior) : t_pack(texture), eta(ior) {}

    ax_device_callable_inlined bool scatter(const Ray &in,
                                            Ray &out,
                                            const intersection_record_s &hit_d,
                                            material_record_s &mat_rec,
                                            sampler::SamplerInterface &sampler,
                                            axstd::StaticAllocator64kb &allocator,
                                            shading_data_s &mat_ctx) const {
      TexturePackSampler texture_pack_sampler(t_pack);
      const IntersectFrame &shading_frame = hit_d.shading_frame;

      glm::vec3 perturbed_shading_normal = compute_local_normal(hit_d.u, hit_d.v, texture_pack_sampler, *mat_ctx.texture_aggregate);
      glm::vec3 perturbed_world_normal = compute_world_normal(perturbed_shading_normal, shading_frame);
      glm::vec3 geometric_world_normal = shading_frame.getNormal();

      DielectricBxDF bxdf(eta, 0.f);
      BSDF bsdf(&bxdf, perturbed_world_normal, shading_frame.getTangent());
      BSDFSample lobe;
      float uc = sampler.sample1D();
      float u[2];
      sampler.sample2D(u);
      if (!bsdf.sample_f(in.direction, uc, u, &lobe, TRANSPORT::RADIANCE, REFLTRANSFLAG::TRANSMISSION))
        return false;

      Spectrum R(texture_pack_sampler.albedo(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate));
      Spectrum E(texture_pack_sampler.emissive(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate));
      mat_rec = {};
      mat_rec.attenuation = lobe.f * R;
      mat_rec.emissive = E;
      mat_rec.lobe = lobe;
      mat_rec.normal = perturbed_world_normal;

      return true;
    }
  };

  class NovaMaterialInterface : public core::tag_ptr<NovaDiffuseMaterial, NovaDielectricMaterial, NovaConductorMaterial> {
   public:
    using tag_ptr::tag_ptr;
    ax_device_callable_inlined bool scatter(const Ray &in,
                                            Ray &out,
                                            const intersection_record_s &hit_d,
                                            material_record_s &mat_rec,
                                            sampler::SamplerInterface &sampler,
                                            axstd::StaticAllocator64kb &allocator,
                                            shading_data_s &mat_ctx) const {
      auto disp = [&](auto material) { return material->scatter(in, out, hit_d, mat_rec, sampler, allocator, mat_ctx); };
      return dispatch(disp);
    }
  };

  using NovaMatIntfView = axstd::span<NovaMaterialInterface>;
  using CstNovaMatIntfView = axstd::span<const NovaMaterialInterface>;

  using TYPELIST = NovaMaterialInterface::type_pack;
}  // namespace nova::material
#endif  // NOVAMATERIALS_H

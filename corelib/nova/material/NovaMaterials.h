#ifndef NOVAMATERIALS_H
#define NOVAMATERIALS_H
#include "BxDF.h"
#include "TexturePackSampler.h"
#include "glm/fwd.hpp"
#include "material/BSDF.h"
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
                                            Ray &out,
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
    float fuzz{};

   public:
    CLASS_CM(NovaConductorMaterial)

    ax_device_callable_inlined NovaConductorMaterial(const texture_pack &texture) : t_pack(texture) {}

    ax_device_callable_inlined NovaConductorMaterial(const texture_pack &texture, float fuzz_) : t_pack(texture), fuzz(fuzz_) {}

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
      glm::vec3 reflected = glm::reflect(in.direction, perturbed_world_normal);
      AX_ASSERT(!ISNAN(reflected), "");

      out.origin = hit_d.position + geometric_world_normal * INTERSECT_OFFSET;
      glm::vec3 generated_direction_vector = shading_frame.localToWorld(hemi_sample(sampler));
      out.direction = DENAN(glm::normalize(reflected + generated_direction_vector * fuzz), glm::normalize(reflected));

      mat_ctx.texture_aggregate->geometric_data.sampling_vector = hit_d.position;

      mat_rec = {};
      mat_rec.attenuation = texture_pack_sampler.albedo(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate);
      mat_rec.emissive = texture_pack_sampler.emissive(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate);
      mat_rec.normal = perturbed_world_normal;

      return glm::dot(geometric_world_normal, out.direction) > 0;
    }
  };

  class NovaDielectricMaterial {
   private:
    texture_pack t_pack{};
    float eta{};  // ior

    ax_device_callable_inlined bool refract(const glm::vec3 &v, const glm::vec3 &n, float eta, glm::vec3 &refracted) const {
      glm::vec3 inc = glm::normalize(v);
      float dt = glm::dot(inc, n);
      float discriminant = 1.f - eta * eta * (1 - dt * dt);
      if (discriminant > 1e-4f) {
        refracted = glm::refract(v, n, eta);
        return true;
      }
      return false;
    }

    ax_device_callable_inlined float schlick(float cosine, float eta) const {
      float r0 = (1 - eta) / (1 + eta);
      r0 *= r0;
      return r0 + (1 - r0) * AX_GPU_FASTPOW((1.f - cosine), 5.f);
    }

   public:
    CLASS_CM(NovaDielectricMaterial)

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

      glm::vec3 original_perturbed_world_normal = perturbed_world_normal;
      glm::vec3 direction = glm::normalize(in.direction);
      const glm::vec3 reflected = glm::reflect(direction, perturbed_world_normal);
      out.origin = hit_d.position;
      AX_ASSERT_NOTNULL(mat_ctx.texture_aggregate);
      mat_ctx.texture_aggregate->geometric_data.sampling_vector = out.origin;

      float index = eta;
      float reflect_prob = 0.f;
      float cosine = 0.f;
      if (glm::dot(direction, perturbed_world_normal) > 0) {
        original_perturbed_world_normal = -original_perturbed_world_normal;
        cosine = glm::dot(direction, perturbed_world_normal) * eta / glm::length(direction);
      } else {
        index = 1.f / eta;
        cosine = -glm::dot(direction, perturbed_world_normal) / glm::length(direction);
      }
      glm::vec3 refracted;
      if (refract(direction, original_perturbed_world_normal, index, refracted))
        reflect_prob = schlick(cosine, eta);
      else
        reflect_prob = 1.f;
      float sampler_random = sampler.sample1D();
      if (sampler_random < reflect_prob)
        out.direction = glm::normalize(reflected);
      else
        out.direction = glm::normalize(refracted);
      AX_ASSERT(!ISNAN(out.direction), "");

      mat_rec = {};
      mat_rec.attenuation = texture_pack_sampler.albedo(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate);
      mat_rec.emissive = texture_pack_sampler.emissive(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate);
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

#ifndef NOVAMATERIALS_H
#define NOVAMATERIALS_H
#include "TexturePackSampler.h"
#include "glm/fwd.hpp"
#include "material_datastructures.h"
#include "ray/Hitable.h"
#include "ray/Ray.h"
#include "sampler/Sampler.h"
#include <internal/common/axstd/span.h>
#include <internal/common/math/math_includes.h>
#include <internal/common/math/utils_3D.h>
#include <internal/debug/debug_utils.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/macro/project_macros.h>
#include <internal/memory/tag_ptr.h>

namespace nova::material {

  struct shading_data_s {
    texturing::texture_data_aggregate_s *texture_aggregate{nullptr};
    // TODO: Add object Transform for World -> Object conversion ?
  };

  constexpr float INTERSECT_OFFSET = 1e-4f;

  ax_device_callable_inlined glm::vec3 compute_map_normal(const hit_data &hit_d,
                                                          const TexturePackSampler &t_pack,
                                                          const glm::mat3 &tbn,
                                                          texturing::texture_data_aggregate_s &sample_data) {
    const glm::vec3 map_normal = t_pack.normal(hit_d.u, hit_d.v, sample_data) * 2.f - 1.f;
    return glm::normalize(tbn * map_normal);
  }

  ax_device_callable_inlined glm::vec3 hemi_sample(const glm::mat3 &tbn, sampler::SamplerInterface &sampler) {
    float ret[2] = {};
    sampler.sample2D(ret);
    return tbn * bxdf::hemisphere_sample_uniform(ret);
  }

  class NovaDiffuseMaterial {
   private:
    texture_pack t_pack{};

   public:
    CLASS_CM(NovaDiffuseMaterial)

    ax_device_callable NovaDiffuseMaterial(const texture_pack &texture) : t_pack(texture) {}

    ax_device_callable bool scatter(
        const Ray & /*in*/, Ray &out, hit_data &hit_d, sampler::SamplerInterface &sampler, shading_data_s &mat_ctx) const {
      AX_ASSERT_NOTNULL(mat_ctx.texture_aggregate);
      TexturePackSampler texture_pack_sampler(t_pack);
      const glm::mat3 tbn = math::geometry::construct_tbn(hit_d.normal, hit_d.tangent, hit_d.bitangent);
      glm::vec3 normal = compute_map_normal(hit_d, texture_pack_sampler, tbn, *mat_ctx.texture_aggregate);
      out.direction = hemi_sample(tbn, sampler);
      out.origin = hit_d.position + hit_d.normal * INTERSECT_OFFSET;

      mat_ctx.texture_aggregate->geometric_data.sampling_vector = hit_d.position;
      hit_d.attenuation = texture_pack_sampler.albedo(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate);
      hit_d.normal = normal;
      hit_d.emissive = texture_pack_sampler.emissive(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate);
      return glm::dot(normal, out.direction) > 0;
    }
  };

  class NovaConductorMaterial {
   private:
    texture_pack t_pack{};
    float fuzz{};

   public:
    CLASS_CM(NovaConductorMaterial)

    ax_device_callable NovaConductorMaterial(const texture_pack &texture) : t_pack(texture) {}

    ax_device_callable NovaConductorMaterial(const texture_pack &texture, float fuzz_) : t_pack(texture), fuzz(fuzz_) {}

    ax_device_callable bool scatter(const Ray &in, Ray &out, hit_data &hit_d, sampler::SamplerInterface &sampler, shading_data_s &mat_ctx) const {
      AX_ASSERT_NOTNULL(mat_ctx.texture_aggregate);

      TexturePackSampler texture_pack_sampler(t_pack);
      const glm::mat3 tbn = math::geometry::construct_tbn(hit_d.normal, hit_d.tangent, hit_d.bitangent);
      glm::vec3 normal = compute_map_normal(hit_d, texture_pack_sampler, tbn, *mat_ctx.texture_aggregate);
      glm::vec3 reflected = glm::reflect(in.direction, normal);
      AX_ASSERT(!ISNAN(reflected), "");

      out.origin = hit_d.position + hit_d.normal * INTERSECT_OFFSET;
      out.direction = DENAN(glm::normalize(reflected + hemi_sample(tbn, sampler) * fuzz), glm::normalize(reflected));

      mat_ctx.texture_aggregate->geometric_data.sampling_vector = hit_d.position;
      hit_d.attenuation = texture_pack_sampler.albedo(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate);
      hit_d.emissive = texture_pack_sampler.emissive(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate);
      hit_d.normal = normal;

      return glm::dot(normal, out.direction) > 0;
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

    ax_device_callable NovaDielectricMaterial(const texture_pack &texture) : t_pack(texture), eta(1.f) {}

    ax_device_callable NovaDielectricMaterial(const texture_pack &texture, float ior) : t_pack(texture), eta(ior) {}

    ax_device_callable bool scatter(const Ray &in, Ray &out, hit_data &hit_d, sampler::SamplerInterface &sampler, shading_data_s &mat_ctx) const {
      TexturePackSampler texture_pack_sampler(t_pack);
      const glm::mat3 tbn = math::geometry::construct_tbn(hit_d.normal, hit_d.tangent, hit_d.bitangent);
      glm::vec3 perturbed_normal = compute_map_normal(hit_d, texture_pack_sampler, tbn, *mat_ctx.texture_aggregate);
      glm::vec3 original_perturbed_normal = perturbed_normal;
      glm::vec3 direction = glm::normalize(in.direction);
      const glm::vec3 reflected = glm::reflect(direction, perturbed_normal);
      out.origin = hit_d.position;
      AX_ASSERT_NOTNULL(mat_ctx.texture_aggregate);
      mat_ctx.texture_aggregate->geometric_data.sampling_vector = out.origin;
      hit_d.attenuation = texture_pack_sampler.albedo(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate);
      hit_d.emissive = texture_pack_sampler.emissive(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate);
      float index = eta;
      float reflect_prob = 0.f;
      float cosine = 0.f;
      if (glm::dot(direction, perturbed_normal) > 0) {
        original_perturbed_normal = -original_perturbed_normal;
        cosine = glm::dot(direction, perturbed_normal) * eta / glm::length(direction);
      } else {
        index = 1.f / eta;
        cosine = -glm::dot(direction, perturbed_normal) / glm::length(direction);
      }
      glm::vec3 refracted;
      if (refract(direction, original_perturbed_normal, index, refracted))
        reflect_prob = schlick(cosine, eta);
      else
        reflect_prob = 1.f;
      float sampler_random = sampler.sample1D();
      if (sampler_random < reflect_prob)
        out.direction = glm::normalize(reflected);
      else
        out.direction = glm::normalize(refracted);
      hit_d.normal = perturbed_normal;
      AX_ASSERT(!ISNAN(out.direction), "");
      return true;
    }
  };

  class NovaMaterialInterface : public core::tag_ptr<NovaDiffuseMaterial, NovaDielectricMaterial, NovaConductorMaterial> {
   public:
    using tag_ptr::tag_ptr;
    ax_device_callable bool scatter(const Ray &in, Ray &out, hit_data &hit_d, sampler::SamplerInterface &sampler, shading_data_s &mat_ctx) const {
      auto disp = [&](auto material) { return material->scatter(in, out, hit_d, sampler, mat_ctx); };
      return dispatch(disp);
    }
  };

  using NovaMatIntfView = axstd::span<NovaMaterialInterface>;
  using CstNovaMatIntfView = axstd::span<const NovaMaterialInterface>;

  using TYPELIST = NovaMaterialInterface::type_pack;
}  // namespace nova::material
#endif  // NOVAMATERIALS_H

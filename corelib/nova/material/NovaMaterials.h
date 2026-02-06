#ifndef NOVAMATERIALS_H
#define NOVAMATERIALS_H
#include "BxDF.h"
#include "TexturePackSampler.h"
#include "material/BSDF.h"
#include "material/BxDF_flags.h"
#include "material/BxDF_math.h"
#include "material_datastructures.h"
#include "ray/Hitable.h"
#include "ray/IntersectFrame.h"
#include "ray/Ray.h"
#include "sampler/Sampler.h"
#include "spectrum/Spectrum.h"
#include "texturing/texture_datastructures.h"
#include "utils/aliases.h"
#include <climits>
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

  ax_device_callable_inlined glm::vec3 compute_local_normal(float u,
                                                            float v,
                                                            const TexturePackSampler &t_pack,
                                                            const texturing::texture_data_aggregate_s &sample_data) {
    return t_pack.normal(u, v, sample_data) * 2.f - 1.f;
  }

  ax_device_callable_inlined glm::vec3 compute_world_normal(const glm::vec3 &local_normal, const glm::vec3 &world_normal, const glm::vec3 &binormal) {
    IntersectFrame shading_frame(world_normal, binormal);
    return glm::normalize(shading_frame.localToWorld(local_normal));
  }

  class PrincipledMaterial {
    texture_pack t_pack{};
    Spectrum eta{0.f}, k{0.f};
    float anisotropy_factor{0.f};

    struct shading_derivatives_s {
      glm::vec3 dpdu;
      glm::vec3 dpdv;
      glm::vec3 ns;
    };

    ax_device_callable_inlined bool sampleBSDF(const glm::vec3 &wo,
                                               const hit_geometry_s &derivatives,
                                               bsdf_params_s &bsdf_params,
                                               BSDFSample &lobe,
                                               sampler::SamplerInterface &sampler,
                                               StackAllocator &allocator) const {

      float uc, u0[2];
      uc = sampler.sample1D();
      sampler.sample2D(u0);
      PrincipledBSDF bsdf(bsdf_params, uc, allocator);

      if (bsdf.flags() & GLOSSY_REFLECTION) {
        glm::vec3 ns = bsdf_params.ns;
        glm::vec3 ng = bsdf_params.ng;
        if (bsdf_params.wo_dot_ng >= 0) {
          ns = glm::normalize(bsdf::correct_specular_shading_normal(ng, wo, ns));

          bsdf_params.ns = ns;
          bsdf_params.wo_dot_ns = glm::dot(wo, ns);
          bsdf.configureBSDFParams(bsdf_params);
        }
      }
      uc = sampler.sample1D();
      bool valid = bsdf.sample_f(wo, uc, u0, &lobe);
      return valid;
    }

   public:
    ax_device_callable_inlined PrincipledMaterial(const texture_pack &texture) : t_pack(texture) {}
    ax_device_callable_inlined PrincipledMaterial(const texture_pack &texture, float eta[3], float k[3], float anisotropy = 0)
        : t_pack(texture), eta(eta), k(k), anisotropy_factor(anisotropy) {}

    ax_device_callable_inlined bool scatter(const Ray &in,
                                            const intersection_record_s &hit_d,
                                            material_record_s &mat_rec,
                                            sampler::SamplerInterface &sampler,
                                            StackAllocator &allocator,
                                            shading_data_s &mat_ctx) const {
      TexturePackSampler texture_pack_sampler(t_pack);
      bsdf_params_s bsdf_params{};
      const hit_geometry_s &geometry = hit_d.geometry;
      const hit_shading_s &shading = hit_d.shading;
      float u = geometry.u, v = geometry.v;
      IntersectFrame shading_frame = shading.frame;

      glm::vec3 local_shading_normal = compute_local_normal(u, v, texture_pack_sampler, *mat_ctx.texture_aggregate);
      glm::vec3 ns = glm::normalize(shading_frame.localToWorld(local_shading_normal));
      glm::vec3 ng = geometry.ng;
      bsdf_params.dpdu = geometry.dpdu;
      bsdf_params.dpdv = geometry.dpdv;
      bsdf_params.ns = ns;
      bsdf_params.ng = ng;

      glm::vec3 wo = glm::normalize(-in.direction);
      bsdf_params.wo_dot_ng = glm::dot(wo, ng);
      bsdf_params.wo_dot_ns = glm::dot(wo, ns);
      bsdf_params.albedo = texture_pack_sampler.albedo(u, v, *mat_ctx.texture_aggregate);

      // TODO for test : change this to sample roughness texture.
      bsdf_params.roughness = texture_pack_sampler.metallic(u, v, *mat_ctx.texture_aggregate).g;
      bsdf_params.metal = texture_pack_sampler.metallic(u, v, *mat_ctx.texture_aggregate).b;
      if (eta && !k)
        bsdf_params.transmission = 1.f;
      else
        bsdf_params.transmission = 0.f;

      bsdf_params.anisotropy_ratio = anisotropy_factor;
      bsdf_params.thin_surface = false;
      bsdf_params.eta = eta;
      bsdf_params.k = k;

      BSDFSample bsdf_lobe{};

      bool valid = sampleBSDF(wo, hit_d.geometry, bsdf_params, bsdf_lobe, sampler, allocator);
      mat_rec = {};
      mat_rec.attenuation = bsdf_lobe.f;
      mat_rec.emissive = texture_pack_sampler.emissive(u, v, *mat_ctx.texture_aggregate);
      mat_rec.lobe = bsdf_lobe;
      mat_rec.normal = ns;
      return valid;
    }
  };

  class NovaMaterialInterface : public core::tag_ptr<PrincipledMaterial> {
   public:
    using tag_ptr::tag_ptr;
    ax_device_callable_inlined bool scatter(const Ray &in,
                                            const intersection_record_s &hit_d,
                                            material_record_s &mat_rec,
                                            sampler::SamplerInterface &sampler,
                                            StackAllocator &allocator,
                                            shading_data_s &mat_ctx) const {
      auto disp = [&](auto material) { return material->scatter(in, hit_d, mat_rec, sampler, allocator, mat_ctx); };
      return dispatch(disp);
    }
  };

  using NovaMatIntfView = axstd::span<NovaMaterialInterface>;
  using CstNovaMatIntfView = axstd::span<const NovaMaterialInterface>;

  using TYPELIST = NovaMaterialInterface::type_pack;
}  // namespace nova::material
#endif  // NOVAMATERIALS_H

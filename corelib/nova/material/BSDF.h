#ifndef BSDF_H
#define BSDF_H

#include "material/BxDF.h"
#include "material/BxDF_flags.h"
#include "ray/IntersectFrame.h"
#include "spectrum/Spectrum.h"
#include "utils/aliases.h"
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/macro/project_macros.h>

namespace nova {

  /*
   * Wrapper around a BxDF instance.
   * Converts direction vectors between shading and world spaces.
   * All vectors in input must be in world space , and will be returned in world space.
   * wo and wi points outwards.
   */

  class PrincipledBSDF {

    struct lobe_params_s {
      Spectrum f{};
      Spectrum eta{};
      glm::vec3 wi{};
      float pdf{0.f};
      BXDFFLAGS flag{NONE};
      bool cosine_weighted{true};
      bool evaluation_success{true};
    };

    struct lobe_weights_s {
      float W, w_diffuse, w_sheen, w_clearcoat, w_specular, w_transmission, p_diffuse, p_sheen, p_clearcoat, p_transmission, p_specular;
    };

    bsdf_params_s params{};
    lobe_weights_s weights{};
    IntersectFrame local_frame{};
    BxDF bxdf;

    ax_device_callable_inlined void generateWeightsFromParam() {
      float abscostheta_o = fabsf(params.wo_dot_ng);

      weights.w_diffuse = (1 - params.transmission) * (1 - params.metal);
      weights.w_sheen = (1 - params.metal) * params.sheen * (1.f - params.transmission);
      weights.w_clearcoat = 0.25f * params.clearcoat;
      weights.w_specular = params.specular;
      weights.w_transmission = (1.f - params.metal) * params.transmission;

      float F = Fresnel::schlick(abscostheta_o, params.specular * 0.04f);
      float p_diffuse = weights.w_diffuse * (1.f - F);
      float p_specular = weights.w_specular * F;
      weights.W = p_diffuse + p_specular + weights.w_sheen + weights.w_clearcoat + weights.w_transmission;

      if (weights.W == 0) {
        weights.p_specular = 1.f;
        weights.W = 1.f;
      }

      weights.p_diffuse = p_diffuse / weights.W;
      weights.p_specular = p_specular / weights.W;
      weights.p_sheen = weights.w_sheen / weights.W;
      weights.p_clearcoat = weights.w_clearcoat / weights.W;
      weights.p_transmission = weights.w_transmission / weights.W;
    }

    ax_device_callable_inlined BxDF constructBXDF(float uc, StackAllocator &allocator) const {
      float idx = 0.f;
      if (uc < (idx += weights.p_diffuse)) {
        return allocator.construct<DisneyDiffuseBxDF>(params, weights.w_diffuse, weights.p_diffuse);
      } else if (uc < (idx += weights.p_specular)) {
        return allocator.construct<SpecularBxDF>(params, weights.w_specular, weights.p_specular);
      } else if (uc < (idx += weights.p_transmission)) {
        return allocator.construct<DielectricBxDF>(params, weights.w_transmission, weights.p_transmission);
      }
      return nullptr;
    }

   public:
    ax_device_callable_inlined PrincipledBSDF(const bsdf_params_s &p_params, float uc, StackAllocator &allocator) {
      configureBSDFParams(p_params);
      bxdf = constructBXDF(uc, allocator);
    }

    ax_device_callable_inlined void configureBSDFParams(const bsdf_params_s &p_params) {
      params = p_params;
      local_frame = IntersectFrame(params.dpdu, params.dpdv, params.ns, true);
      if (params.wo_dot_ng < 0) {
        local_frame.flipFrame();
        if (!params.thin_surface && params.eta != 0.f)
          params.eta = 1.f / params.eta;
      }
      params.ng = glm::normalize(local_frame.worldToLocal(params.ng));
      generateWeightsFromParam();
    }

    ax_device_callable_inlined Spectrum f(const glm::vec3 &wo, const glm::vec3 &wi, TRANSPORT transport_mode = TRANSPORT::RADIANCE) const {
      if (!bxdf.get())
        return 0.f;
      glm::vec3 local_wi = glm::normalize(local_frame.worldToLocal(wi));
      glm::vec3 local_wo = glm::normalize(local_frame.worldToLocal(wo));
      return bxdf.f(local_wo, local_wi, transport_mode);
    }

    ax_device_callable_inlined bool sample_f(const glm::vec3 &wo,
                                             float uc,
                                             const float u[2],
                                             BSDFSample *sample,
                                             TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                             REFLTRANSFLAG sample_flag = REFLTRANSFLAG::ALL) const {
      AX_ASSERT_NOTNULL(sample);
      if (!bxdf.get())
        return false;

      glm::vec3 local_wo = glm::normalize(local_frame.worldToLocal(wo));

      if (!(bxdf.flags() & (unsigned)sample_flag) || local_wo.z == 0.f)
        return false;

      if (!bxdf.sample_f(local_wo, uc, u, sample, transport_mode, sample_flag))
        return false;

      if (!sample->f || sample->pdf == 0.f || sample->wi.z == 0.f)
        return false;

      sample->wi = glm::normalize(local_frame.localToWorld(sample->wi));
      return true;
    }

    ax_device_callable_inlined float pdf(const glm::vec3 &wo,
                                         const glm::vec3 &wi,
                                         TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                         REFLTRANSFLAG flag = REFLTRANSFLAG::ALL) const {
      if (!bxdf.get())
        return 0.f;
      glm::vec3 local_wo = glm::normalize(local_frame.worldToLocal(wo));
      glm::vec3 local_wi = glm::normalize(local_frame.localToWorld(wi));

      return bxdf.pdf(local_wo, local_wi, transport_mode, flag);
    }

    ax_device_callable_inlined unsigned flags() const {
      if (!bxdf.get())
        return NONE;
      return bxdf.flags();
    }

    ax_device_callable_inlined Spectrum rho(const glm::vec3 &wo, axstd::cspan<float> &uc, axstd::cspan<uniform_sample2d> &u) const {
      if (!bxdf.get())
        return 0.f;
      glm::vec3 local_wo = local_frame.worldToLocal(wo);
      return bxdf.rho(local_wo, uc, u);
    }

    ax_device_callable_inlined Spectrum rho(axstd::cspan<uniform_sample2d> &u0, axstd::cspan<float> &uc, axstd::cspan<uniform_sample2d> &u1) const {
      if (!bxdf.get())
        return 0.f;
      return bxdf.rho(u0, uc, u1);
    }
  };

}  // namespace nova
#endif

#ifndef BSDF_H
#define BSDF_H

#include "material/BxDF.h"
#include "material/BxDF_flags.h"
#include "ray/IntersectFrame.h"
#include "spectrum/Spectrum.h"
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/macro/project_macros.h>

namespace nova::material {

  /*
   * Wrapper around a BxDF instance.
   * Converts direction vectors between shading and world spaces.
   * All vectors in input must be in world space , and will be returned in world space.
   * wo is entrant (eye -> sample).
   * wi is outgoing (sample -> eye).
   */
  class BSDF {
    IntersectFrame local_frame;
    BxDF bxdf;
    float wo_dot_ng;  // Are we entering a medium or exiting.

   public:
    ax_device_callable_inlined BSDF() = default;

    template<class BXDF>
    ax_device_callable_inlined BSDF(BXDF *bxdf_instance, const glm::vec3 &shading_normal, const glm::vec3 &dpdu, float wo_dot_ng) {
      local_frame = IntersectFrame(shading_normal, dpdu);
      bxdf = bxdf_instance;
      this->wo_dot_ng = wo_dot_ng;
      if (wo_dot_ng < 0) {  // TODO: Only valid for non thin materials.
        local_frame.flipFrame();
        bxdf.invertEta();
      }
    }
    ax_device_callable_inlined Spectrum f(const glm::vec3 &wo, const glm::vec3 &wi, TRANSPORT transport_mode = TRANSPORT::RADIANCE) const {
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
      glm::vec3 local_wo = glm::normalize(local_frame.worldToLocal(wo));

      if (!(bxdf.flags() & (unsigned)sample_flag) && local_wo.z == 0.f)
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

      glm::vec3 local_wo = glm::normalize(local_frame.worldToLocal(wo));
      glm::vec3 local_wi = glm::normalize(local_frame.localToWorld(wi));

      return bxdf.pdf(local_wo, local_wi, transport_mode, flag);
    }

    ax_device_callable_inlined unsigned flags() const { return bxdf.flags(); }

    ax_device_callable_inlined Spectrum rho(const glm::vec3 &wo, axstd::cspan<float> &uc, axstd::cspan<uniform_sample2d> &u) const {
      glm::vec3 local_wo = local_frame.worldToLocal(wo);
      return bxdf.rho(local_wo, uc, u);
    }

    ax_device_callable_inlined Spectrum rho(axstd::cspan<uniform_sample2d> &u0, axstd::cspan<float> &uc, axstd::cspan<uniform_sample2d> &u1) const {
      return bxdf.rho(u0, uc, u1);
    }
  };

}  // namespace nova::material
#endif

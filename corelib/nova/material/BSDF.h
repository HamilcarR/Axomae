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
   * This class serves as a wrapper around a BxDF instance.
   * Its goal is about converting direction vectors between shading and world spaces.
   */

  class BSDF {
    IntersectFrame local_frame;
    BxDF bxdf;

   public:
    ax_device_callable_inlined BSDF() = default;

    template<class BXDF>
    ax_device_callable_inlined BSDF(const BXDF *bxdf_instance, const glm::vec3 &shading_normal, const glm::vec3 &dpdu)
        : local_frame(shading_normal, dpdu), bxdf(bxdf_instance) {}

    ax_device_callable_inlined Spectrum f(const glm::vec3 &wo, const glm::vec3 &wi, TRANSPORT transport_mode = TRANSPORT::RADIANCE) const {
      glm::vec3 local_wi = local_frame.worldToLocal(wi);
      glm::vec3 local_wo = local_frame.worldToLocal(wo);
      return bxdf.f(local_wo, local_wi, transport_mode);
    }

    ax_device_callable_inlined bool sample_f(const glm::vec3 &wo,
                                             float uc,
                                             const float u[2],
                                             BSDFSample *sample,
                                             TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                             REFLTRANSFLAG sample_flag = REFLTRANSFLAG::ALL) const {
      AX_ASSERT_NOTNULL(sample);
      glm::vec3 local_wo = local_frame.worldToLocal(wo);
      if (bxdf.sample_f(local_wo, uc, u, sample)) {
        sample->wi = local_frame.localToWorld(sample->wi);
        return true;
      }
      return false;
    }

    ax_device_callable_inlined float pdf(const glm::vec3 &wo,
                                         const glm::vec3 &wi,
                                         TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                         REFLTRANSFLAG flag = REFLTRANSFLAG::ALL) const {

      glm::vec3 local_wo = local_frame.worldToLocal(wo);
      glm::vec3 local_wi = local_frame.localToWorld(wi);

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
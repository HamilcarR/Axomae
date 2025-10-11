#ifndef BXDF_H
#define BXDF_H
#include "BxDF_flags.h"
#include "BxDF_math.h"
#include "spectrum/Spectrum.h"
#include <internal/common/axstd/span.h>
#include <internal/common/math/math_spherical.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/macro/project_macros.h>
#include <internal/memory/tag_ptr.h>

struct uniform_sample2d {
  float u[2];
};

/*
 * Based on "Physically Based Rendering : From Theory To Implementation" , vol 4
 * We will use here the same methods and naming conventions as the books.
 * Other than that , we assume view vectors to always be outgoing from the hemisphere.
 * wi = sample -> light if exitant , light -> sample if entrant.
 * wo = sample -> eye.
 * All vectors are expressed in tagent space and returned in tangent space.
 */

namespace nova::material {

  ax_device_callable_inlined bool same_hemisphere(const glm::vec3 &w1, const glm::vec3 &w2) { return w1.z * w2.z > 0; }

  class CoatedDiffuseBxDF;
  class CoatedConductorBxDF;
  class DielectricBxDF;
  class ThinDielectricBxDF;
  class HairBxDF;
  class DiffuseTransmissionBxDF;
  class ConductorBxDF;

  class DiffuseBxDF {
    Spectrum R;

   public:
    ax_device_callable_inlined DiffuseBxDF(Spectrum RR) : R(RR) {}

    ax_device_callable_inlined Spectrum f(const glm::vec3 &wo, const glm::vec3 &wi, TRANSPORT mode = TRANSPORT::RADIANCE) const {
      if (!same_hemisphere(wi, wo))
        return Spectrum(0.f);
      return R * INV_PI;
    }

    ax_device_callable_inlined float pdf(const glm::vec3 &wo,
                                         const glm::vec3 &wi,
                                         TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                         REFLTRANSFLAG flag = REFLTRANSFLAG::ALL) const {
      if (!same_hemisphere(wi, wo))
        return 0;
      if (!(flag & REFLTRANSFLAG::REFLECTION))
        return 0;
      return bxdf::costheta(wi) * (float)INV_PI;
    }

    /*
     * uc randomly choses between scattering types (reflection / transmission)
     */
    ax_device_callable_inlined bool sample_f(const glm::vec3 &wo,
                                             const float uc,
                                             const float u[2],
                                             BSDFSample *sample,
                                             TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                             REFLTRANSFLAG sample_flag = REFLTRANSFLAG::ALL) const {
      if (!(sample_flag & REFLTRANSFLAG::REFLECTION))
        return false;
      AX_ASSERT_NOTNULL(sample);
      glm::vec3 wi = bxdf::hemisphere_sample_uniform(u);
      if (wo.z < 0)
        wi.z *= -1.f;

      sample->wi = wi;
      sample->pdf = bxdf::costheta(wi) * (float)INV_PI;
      sample->flags = BXDFFLAGS::DIFFUSE_REFLECTION;
      sample->is_pdf_proportionnal = false;
      sample->f = R * INV_PI;
      sample->eta = 1.f;

      return true;
    }

    ax_device_callable_inlined unsigned flags() const { return R ? BXDFFLAGS::DIFFUSE_REFLECTION : BXDFFLAGS::NONE; }
  };

  /*
  #define LISTBxDF \
    DiffuseBxDF, DiffuseTransmissionBxDF, CoatedConductorBxDF, CoatedConductorBxDF, DielectricBxDF, ThinDielectricBxDF, HairBxDF, ConductorBxDF
  */

#define LISTBxDF DiffuseBxDF

  class BxDF : public core::tag_ptr<LISTBxDF> {
   public:
    using tag_ptr::tag_ptr;

    ax_device_callable_inlined Spectrum f(const glm::vec3 &wo, const glm::vec3 &wi, TRANSPORT transport_mode = TRANSPORT::RADIANCE) const {
      auto d = [&](auto bxdf) { return bxdf->f(wo, wi, transport_mode); };
      return dispatch(d);
    }

    ax_device_callable_inlined float pdf(const glm::vec3 &wo,
                                         const glm::vec3 &wi,
                                         TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                         REFLTRANSFLAG flag = REFLTRANSFLAG::ALL) const {
      auto d = [&](auto bxdf) { return bxdf->pdf(wo, wi, transport_mode, flag); };
      return dispatch(d);
    }

    ax_device_callable_inlined bool sample_f(const glm::vec3 &wo,
                                             const float uc,
                                             const float u[2],
                                             BSDFSample *sample,
                                             TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                             REFLTRANSFLAG sample_flag = REFLTRANSFLAG::ALL) const {
      auto d = [&](auto bxdf) { return bxdf->sample_f(wo, uc, u, sample, transport_mode, sample_flag); };
      return dispatch(d);
    }

    ax_device_callable_inlined unsigned flags() const {
      auto d = [&](auto bxdf) { return bxdf->flags(); };
      return dispatch(d);
    }

    /**
     * @brief Computes total reflectance over the hemisphere due to one direction. (Rho_h_d(wo) / hemisphere-direction).
     *
     * @param wo Direction vector.
     * @param uc 1D Low discrepancy uniforms.
     * @param u 2D Low discrepancy uniforms.
     * @return Spectrum
     */

    ax_device_callable_inlined Spectrum rho(const glm::vec3 &wo, axstd::cspan<float> &uc, axstd::cspan<float[2]> u) const {
      AX_ASSERT_EQ(uc.size(), u2.size());
      Spectrum R(0.f);
      for (size_t i = 0; i < u.size(); i++) {
        BSDFSample sample;
        bool is_sampled = sample_f(wo, uc[i], u[i], &sample);
        if (is_sampled && sample.pdf > 0)
          R += sample.f * bxdf::costheta(sample.wi) / sample.pdf;
      }
      return R / (float)uc.size();
    }

    /**
     * @brief Computes the fraction of incident light when the incident light is the same over the hemisphere. (Rho_h_h(wo) / hemisphere-hemisphere).
     *
     * @param u0 2D Low discrepancy uniforms.
     * @param uc 1D Low discrepancy uniforms.
     * @param u1 2D Low discrepancy uniforms.
     * @return Spectrum
     */
    ax_device_callable_inlined Spectrum rho(axstd::cspan<uniform_sample2d> &u0, axstd::cspan<float> &uc, axstd::cspan<uniform_sample2d> &u1) const {
      AX_ASSERT_EQ(u0.size(), uc.size());
      AX_ASSERT_EQ(uc.size(), u1.size());
      Spectrum R(0.f);
      for (size_t i = 0; i < uc.size(); i++) {
        glm::vec3 wo = bxdf::hemisphere_sample_uniform(u0[0].u);
        if (wo.z == 0)
          continue;
        constexpr float pdf = bxdf::hemisphere_pdf();
        BSDFSample sample;
        bool is_sampled = sample_f(wo, uc[i], u1[i].u, &sample);
        if (is_sampled && sample.pdf > 0)
          R += sample.f * bxdf::costheta(wo) * bxdf::costheta(sample.wi) / (pdf * sample.pdf);
      }
      return R / (float)uc.size() * PI;
    }
  };

}  // namespace nova::material
#endif

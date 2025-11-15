#ifndef BXDF_H
#define BXDF_H
#include "BxDF_flags.h"
#include "BxDF_math.h"
#include "glm/geometric.hpp"
#include "spectrum/Spectrum.h"
#include <internal/common/axstd/span.h>
#include <internal/common/math/math_spherical.h>
#include <internal/common/math/math_utils.h>
#include <internal/debug/debug_utils.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/macro/project_macros.h>
#include <internal/memory/tag_ptr.h>

struct uniform_sample2d {
  float u[2];
};

namespace nova::material {

  class CoatedDiffuseBxDF;
  class CoatedConductorBxDF;
  class ThinDielectricBxDF;
  class HairBxDF;
  class DiffuseTransmissionBxDF;

  class DielectricBxDF {
    float eta{1.f};  // External medium refraction index.
    NDF ggx;

    ax_device_callable_inlined bool perfectSpecularReflection(
        glm::vec3 wo, float R, float pr, float pt, float eta, BSDFSample *sample, TRANSPORT transport, REFLTRANSFLAG flag) const {

      glm::vec3 wi = glm::reflect(wo, glm::vec3(0.f, 0.f, 1.f));
      float costheta_i = bxdf::abscostheta(wi);
      sample->costheta = costheta_i;
      sample->f = R / costheta_i;
      sample->pdf = pr / (pr + pt);
      sample->eta = eta;
      sample->flags = BXDFFLAGS::SPECULAR_REFLECTION;
      sample->pdf_cosine_weighted = true;
      sample->wi = wi;
      return true;
    }

    ax_device_callable_inlined bool perfectSpecularTransmission(
        glm::vec3 wo, float T, float pr, float pt, float eta, BSDFSample *sample, TRANSPORT transport, REFLTRANSFLAG flag) const {
      glm::vec3 n(0.f, 0.f, 1.f);
      float costheta_o = bxdf::costheta(wo);
      if (costheta_o > 0) {
        n = -n;
        eta = 1.f / eta;
      }

      glm::vec3 wi = glm::refract(wo, n, eta);
      float costheta_i = bxdf::abscostheta(wi);
      sample->costheta = costheta_i;
      sample->f = (transport == TRANSPORT::RADIANCE) ? (T / costheta_i) / math::sqr(eta) : (T / costheta_i);
      sample->pdf = pt / (pt + pr);
      sample->eta = eta;
      sample->flags = BXDFFLAGS::SPECULAR_TRANSMISSION;
      sample->pdf_cosine_weighted = true;
      sample->wi = wi;
      return true;
    }

   public:
    ax_device_callable_inlined DielectricBxDF(float n, float roughness) : eta(1.f / n), ggx(roughness) {}

    ax_device_callable_inlined DielectricBxDF(float roughness) : ggx(roughness) {}

    ax_device_callable_inlined DielectricBxDF(float n1, float n2, float roughness) : ggx(roughness) {
      AX_ASSERT_NEQ(n2, 0.f);
      eta = n1 / n2;
    }

    ax_device_callable_inlined Spectrum f(const glm::vec3 &wo, const glm::vec3 &wi, TRANSPORT mode = TRANSPORT::RADIANCE) const {
      if (!bxdf::same_hemisphere(wo, wi))
        return Spectrum(0.f);
      if (ggx.isFullSpecular())
        return Spectrum(0.f);
    }

    ax_device_callable_inlined float pdf(const glm::vec3 &wo,
                                         const glm::vec3 &wi,
                                         TRANSPORT mode = TRANSPORT::RADIANCE,
                                         REFLTRANSFLAG sample_flag = REFLTRANSFLAG::ALL) const {}

    ax_device_callable_inlined bool sample_f(const glm::vec3 &wo,
                                             float uc,
                                             const float u[2],
                                             BSDFSample *sample,
                                             TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                             REFLTRANSFLAG sample_flag = REFLTRANSFLAG::ALL) const {

      float costheta_o = bxdf::costheta(wo);

      if (ggx.isFullSpecular() || eta == 1.f) {
        Fresnel fresnel;
        float R = fresnel.realIndex(costheta_o, eta), T = 1 - R;
        float pr = R, pt = T;
        if (!(sample_flag & REFLTRANSFLAG::REFLECTION))
          pr = 0;
        if (!(sample_flag & REFLTRANSFLAG::TRANSMISSION))
          pt = 0;
        if (pr == 0.f && pt == 0.f)
          return false;

        if (uc < (pr / (pr + pt))) {  // Specular reflection
          return perfectSpecularReflection(wo, R, pr, pt, eta, sample, transport_mode, sample_flag);
        } else {  // Specular Transmission
          return perfectSpecularTransmission(wo, T, pr, pt, eta, sample, transport_mode, sample_flag);
        }
      }
      return false;
    }

    ax_device_callable_inlined unsigned flags() const {
      unsigned flag = (eta / eta) == 1.f ? BXDFFLAGS::TRANSMISSION : BXDFFLAGS::REFLECTION | BXDFFLAGS::TRANSMISSION;
      if (ggx.isFullSpecular())
        return flag | BXDFFLAGS::SPECULAR;
      return flag | BXDFFLAGS::GLOSSY;
    }
  };

  class ConductorBxDF {
    Spectrum k, eta;
    NDF ggx;

    ax_device_callable_inlined Spectrum computeFresnel(float costheta_i) const {
      Spectrum ret(0.f);
      Fresnel fresnel;
      for (unsigned i = 0; i < Spectrum::SPECTRUM_SAMPLES; i++) {
        ret.samples[i] = fresnel.complexIndex(costheta_i, math::fcomplex(eta[i], k[i]));
      }
      return ret;
    }

   public:
    ax_device_callable_inlined ConductorBxDF(Spectrum eta, Spectrum k, float roughness) : eta(eta), k(k), ggx(roughness) {}

    ax_device_callable_inlined Spectrum f(const glm::vec3 &wo, const glm::vec3 &wi, TRANSPORT mode = TRANSPORT::RADIANCE) const {
      if (!bxdf::same_hemisphere(wo, wi))
        return Spectrum(0.f);
      if (ggx.isFullSpecular())
        return Spectrum(0.f);

      // Implement rough specular
    }

    ax_device_callable_inlined float pdf(const glm::vec3 &wo,
                                         const glm::vec3 &wi,
                                         TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                         REFLTRANSFLAG sample_flag = REFLTRANSFLAG::ALL) const {
      if (!(sample_flag & REFLTRANSFLAG::REFLECTION))
        return 0.f;
      if (!bxdf::same_hemisphere(wi, wo))
        return 0.f;
      if (ggx.isFullSpecular())
        return 0.f;
    }

    ax_device_callable_inlined bool sample_f(const glm::vec3 &wo,
                                             float uc,
                                             const float u[2],
                                             BSDFSample *sample,
                                             TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                             REFLTRANSFLAG sample_flag = REFLTRANSFLAG::ALL) const {

      AX_ASSERT_NOTNULL(sample);
      if (!(sample_flag & REFLTRANSFLAG::REFLECTION))
        return false;
      // Models a dirac delta distribution , returns 1.f for pdf as convention.
      if (ggx.isFullSpecular()) {
        glm::vec3 wi = glm::reflect(wo, glm::vec3(0.f, 0.f, 1.f));
        float costheta_i = bxdf::abscostheta(wi);
        Spectrum fresnel = computeFresnel(costheta_i) / costheta_i;
        sample->costheta = costheta_i;
        sample->eta = eta;
        sample->f = fresnel;
        sample->pdf = 1.f;
        sample->pdf_cosine_weighted = true;
        sample->wi = wi;
        sample->flags = BXDFFLAGS::SPECULAR_REFLECTION;
        return true;
      }
      return false;
    }

    ax_device_callable_inlined unsigned flags() const { return ggx.isFullSpecular() ? BXDFFLAGS::SPECULAR_REFLECTION : BXDFFLAGS::GLOSSY_REFLECTION; }
  };

  class DiffuseBxDF {
    Spectrum R;

   public:
    ax_device_callable_inlined DiffuseBxDF(Spectrum RR) : R(RR) {}

    ax_device_callable_inlined Spectrum f(const glm::vec3 &wo, const glm::vec3 &wi, TRANSPORT mode = TRANSPORT::RADIANCE) const {
      if (!bxdf::same_hemisphere(wi, wo))
        return Spectrum(0.f);
      return R * INV_PI;
    }

    ax_device_callable_inlined float pdf(const glm::vec3 &wo,
                                         const glm::vec3 &wi,
                                         TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                         REFLTRANSFLAG flag = REFLTRANSFLAG::ALL) const {
      if (!bxdf::same_hemisphere(wi, wo))
        return 0;
      if (!(flag & REFLTRANSFLAG::REFLECTION))
        return 0;
      return bxdf::abscostheta(wi) * (float)INV_PI;
    }

    ax_device_callable_inlined bool sample_f(const glm::vec3 & /*wo*/,
                                             float uc,  // randomly choses between scattering types (reflection / transmission)
                                             const float u[2],
                                             BSDFSample *sample,
                                             TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                             REFLTRANSFLAG sample_flag = REFLTRANSFLAG::ALL) const {
      if (!(sample_flag & REFLTRANSFLAG::REFLECTION))
        return false;
      AX_ASSERT_NOTNULL(sample);
      glm::vec3 wi = bxdf::hemisphere_cosine_sample(u);
      float costheta = bxdf::abscostheta(wi);
      sample->wi = wi;
      sample->pdf = costheta * (float)INV_PI;
      sample->flags = BXDFFLAGS::DIFFUSE_REFLECTION;
      sample->pdf_cosine_weighted = true;
      sample->f = R * INV_PI;
      sample->eta = 1.f;
      sample->costheta = costheta;
      return true;
    }

    ax_device_callable_inlined unsigned flags() const { return R ? BXDFFLAGS::DIFFUSE_REFLECTION : BXDFFLAGS::NONE; }
  };

#define LISTBxDF DiffuseBxDF, ConductorBxDF, DielectricBxDF

  class BxDF : public core::tag_ptr<LISTBxDF> {
   public:
    using tag_ptr::tag_ptr;

    /**
     * @brief Evalutes BxDF
     *
     * @param wo view direction, Always incoming (eye->sample).
     * @param wi radiance direction, Always outgoing (sample->light).
     * @param transport_mode Transport Mode.
     * @return Spectrum
     */
    ax_device_callable_inlined Spectrum f(const glm::vec3 &wo, const glm::vec3 &wi, TRANSPORT transport_mode = TRANSPORT::RADIANCE) const {
      auto d = [&](auto bxdf) { return bxdf->f(wo, wi, transport_mode); };
      return dispatch(d);
    }

    /**
     * @brief Computes the PDF of the underlying BXDF
     *
     * @param wo view direction . Always incoming (eye->sample)
     * @param wi radiance direction. Always outgoing (sample-> light).
     * @param transport_mode Transport Mode.
     * @param flag Reflection/Transmission flag for rendering passes.
     * @return float: PDF of the BxDF.
     */
    ax_device_callable_inlined float pdf(const glm::vec3 &wo,
                                         const glm::vec3 &wi,
                                         TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                         REFLTRANSFLAG flag = REFLTRANSFLAG::ALL) const {
      auto d = [&](auto bxdf) { return bxdf->pdf(wo, wi, transport_mode, flag); };
      return dispatch(d);
    }

    /**
     * @brief Generates a valid BxDF lobe when possible.
     *
     * @param wo view direction. Always incoming (eye->sample).
     * @param uc 1D random float.
     * @param u 2D random floats.
     * @param sample Computed BXDF lobe.
     * @param transport_mode Transport Mode.
     * @param sample_flag Reflection/Transmission flag.
     * @return bool: If computation of the lobe is possible.
     */
    ax_device_callable_inlined bool sample_f(const glm::vec3 &wo,
                                             float uc,
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

    ax_device_callable_inlined Spectrum rho(const glm::vec3 &wo, axstd::cspan<float> &uc, axstd::cspan<uniform_sample2d> u) const {
      AX_ASSERT_EQ(uc.size(), u.size());
      Spectrum R(0.f);
      for (size_t i = 0; i < u.size(); i++) {
        BSDFSample sample;
        bool is_sampled = sample_f(wo, uc[i], u[i].u, &sample);
        if (is_sampled && sample.pdf > 0)
          R += sample.f * bxdf::abscostheta(sample.wi) / sample.pdf;
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
          R += sample.f * bxdf::abscostheta(wo) * bxdf::abscostheta(sample.wi) / (pdf * sample.pdf);
      }
      return R / (float)uc.size() * PI;
    }
  };

}  // namespace nova::material
#endif

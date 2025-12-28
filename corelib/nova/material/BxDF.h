#ifndef BXDF_H
#define BXDF_H
#include "BxDF_flags.h"
#include "BxDF_math.h"
#include "spectrum/Spectrum.h"
#include <cmath>
#include <internal/common/axstd/span.h>
#include <internal/common/math/math_includes.h>
#include <internal/common/math/math_spherical.h>
#include <internal/common/math/math_utils.h>
#include <internal/debug/debug_utils.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/macro/project_macros.h>
#include <internal/memory/tag_ptr.h>

namespace nova {

  struct bsdf_params_s {
    glm::vec3 shading_normal;
    glm::vec3 dpdu;

    float wo_dot_ng;
    float roughness;
    float metal;
    float transmission{0.f};      // 0 = opaque , 1 = transparent dielectric.
    float anisotropy_ratio{1.f};  // [-1,1]
    float subsurface_scattering{0.8f};
    float sheen;
    float clearcoat;
    float specular{1.f};
    bool thin_surface{false};

    Spectrum eta, k;
    Spectrum albedo;
  };

  struct uniform_sample2d {
    float u[2];
  };

  class CoatedDiffuseBxDF;
  class CoatedConductorBxDF;
  class ThinDielectricBxDF;
  class HairBxDF;
  class DiffuseTransmissionBxDF;

  class DielectricBxDF {
    Spectrum ior{1.f};
    VNDF ggx;

    ax_device_callable_inlined bool roughSpecularReflection(
        glm::vec3 wo, const glm::vec3 &wm, float R, float pr, float pt, float eta, BSDFSample *sample, TRANSPORT transport, REFLTRANSFLAG flag)
        const {
      glm::vec3 wi = glm::reflect(-wo, wm);
      if (!bxdf::same_hemisphere(wo, wi))
        return false;
      Spectrum f = ggx.D(wm) * ggx.G(wo, wi) * R / (4 * bxdf::costheta(wi) * bxdf::costheta(wo));
      if (!f.isValid())
        return false;
      float pdf = ggx.pdf(wo, wm) / (4 * bxdf::absdot(wo, wm)) * pr / (pr + pt);
      sample->f = f;
      sample->pdf = pdf;
      sample->wi = wi;
      sample->costheta = bxdf::abscostheta(wi);
      sample->pdf_cosine_weighted = false;
      sample->eta = eta;
      sample->flags = BXDFFLAGS::GLOSSY_REFLECTION;
      return true;
    }

    ax_device_callable_inlined bool roughSpecularTransmission(
        glm::vec3 wo, glm::vec3 wm, float T, float pr, float pt, float eta, BSDFSample *sample, TRANSPORT transport, REFLTRANSFLAG flag) const {

      glm::vec3 wi;
      bool no_tir = true;
      eta = bxdf::refract(wo, wm, eta, no_tir, wi);

      if (bxdf::same_hemisphere(wi, wo) || wi.z == 0.f || !no_tir)
        return false;
      float costheta_i = bxdf::costheta(wi);
      float costheta_o = bxdf::costheta(wo);

      float denom = math::sqr(glm::dot(wi, wm) + glm::dot(wo, wm) / eta);
      float jacobian = bxdf::absdot(wi, wm) / denom;
      float pdf = ggx.pdf(wo, wm) * jacobian * pt / (pt + pr);
      AX_ASSERT(!ISNAN(pdf) && !ISINF(pdf), "Rough dielectric transmission pdf returned nan or inf.");
      Spectrum f = T * ggx.D(wm) * ggx.G(wo, wi) * fabsf(glm::dot(wi, wm) * glm::dot(wo, wm) / (costheta_i * costheta_o * denom));
      if (transport == TRANSPORT::RADIANCE)
        f /= math::sqr(eta);
      if (!f.isValid())
        return false;
      sample->f = f;
      sample->eta = eta;
      sample->costheta = bxdf::abscostheta(wi);
      sample->pdf = pdf;
      sample->wi = wi;
      sample->pdf_cosine_weighted = false;
      sample->flags = BXDFFLAGS::GLOSSY_TRANSMISSION;
      return true;
    }

    ax_device_callable_inlined bool perfectSpecularTransmission(
        glm::vec3 wo, float T, float pr, float pt, float eta, BSDFSample *sample, TRANSPORT transport, REFLTRANSFLAG flag) const {
      bool no_tir = true;
      glm::vec3 wi;
      bxdf::refract(wo, glm::vec3(0.f, 0.f, 1.f), eta, no_tir, wi);
      if (!no_tir)
        return false;
      float costheta_i = bxdf::abscostheta(wi);
      Spectrum f = T / costheta_i;
      if (transport == RADIANCE)
        f /= math::sqr(eta);
      sample->f = f;
      sample->costheta = costheta_i;
      sample->pdf = pt / (pr + pt);
      sample->eta = eta;
      sample->flags = BXDFFLAGS::SPECULAR_TRANSMISSION;
      sample->pdf_cosine_weighted = false;
      sample->wi = wi;
      return true;
    }

    ax_device_callable_inlined bool perfectSpecularReflection(
        glm::vec3 wo, float R, float pr, float pt, float eta, BSDFSample *sample, TRANSPORT transport, REFLTRANSFLAG flag) const {
      glm::vec3 wi = glm::reflect(-wo, glm::vec3(0, 0, 1.f));
      float costheta_i = bxdf::abscostheta(wi);
      sample->costheta = costheta_i;
      sample->f = R / costheta_i;
      sample->pdf = pr / (pr + pt);
      sample->eta = eta;
      sample->flags = BXDFFLAGS::SPECULAR_REFLECTION;
      sample->pdf_cosine_weighted = false;
      sample->wi = wi;
      return true;
    }

   public:
    ax_device_callable_inlined DielectricBxDF(Spectrum eta, float roughness) : ior(eta), ggx(roughness) {}

    ax_device_callable_inlined DielectricBxDF(float roughness) : ggx(roughness) {}

    ax_device_callable_inlined DielectricBxDF(Spectrum eta, float roughness, float anisotropy) : ior(eta), ggx(anisotropy, roughness) {}

    ax_device_callable_inlined Spectrum f(const glm::vec3 &wo, const glm::vec3 &wi, TRANSPORT mode = TRANSPORT::RADIANCE) const {
      if (ggx.isFullSpecular() || ior == 1.f)
        return Spectrum(0.f);

      return 0.f;
    }

    ax_device_callable_inlined float pdf(const glm::vec3 &wo,
                                         const glm::vec3 &wi,
                                         TRANSPORT mode = TRANSPORT::RADIANCE,
                                         REFLTRANSFLAG sample_flag = REFLTRANSFLAG::ALL) const {

      if (ior == 1.f || ggx.isFullSpecular())
        return 0.f;
    }

    ax_device_callable_inlined bool sample_f(const glm::vec3 &wo_incident,
                                             float uc,
                                             const float u[2],
                                             BSDFSample *sample,
                                             TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                             REFLTRANSFLAG sample_flag = REFLTRANSFLAG::ALL) const {

      float eta_computed = 0.f;
      float cste_ior = ior[0];  // TODO : Make dielectric wavelength dependent
      Fresnel fresnel(cste_ior);
      if (ggx.isFullSpecular() || ior == 1.f) {

        float R = fresnel.real(bxdf::costheta(wo_incident), eta_computed), T = 1 - R;
        float pr = R, pt = T;
        if (!(sample_flag & REFLTRANSFLAG::REFL))
          pr = 0;
        if (!(sample_flag & REFLTRANSFLAG::TRAN))
          pt = 0;
        if (pr == 0.f && pt == 0.f)
          return false;

        if (uc < (pr / (pr + pt)))  // Specular reflection
          return perfectSpecularReflection(wo_incident, R, pr, pt, cste_ior, sample, transport_mode, sample_flag);
        else  // Specular Transmission
          return perfectSpecularTransmission(wo_incident, T, pr, pt, cste_ior, sample, transport_mode, sample_flag);
      }
      // Sample rough dielectric.
      glm::vec3 wm = ggx.sampleGGXVNDF(wo_incident, u);
      float wo_dot_wm = glm::dot(wo_incident, wm);
      float R = fresnel.real(wo_dot_wm, eta_computed), T = 1 - R;
      float pr = R, pt = T;
      if (!(sample_flag & REFLTRANSFLAG::REFL))
        pr = 0;
      if (!(sample_flag & REFLTRANSFLAG::TRAN))
        pt = 0;
      if (pr == 0.f && pt == 0.f)
        return false;
      if (uc < (pr / (pr + pt)))
        return roughSpecularReflection(wo_incident, wm, R, pr, pt, cste_ior, sample, transport_mode, sample_flag);
      else
        return roughSpecularTransmission(wo_incident, wm, T, pr, pt, cste_ior, sample, transport_mode, sample_flag);
    }

    ax_device_callable_inlined unsigned flags() const {
      unsigned flag = ior == 1.f ? BXDFFLAGS::TRANSMISSION : BXDFFLAGS::REFLECTION | BXDFFLAGS::TRANSMISSION;
      if (ggx.isFullSpecular())
        return flag | BXDFFLAGS::SPECULAR;
      return flag | BXDFFLAGS::GLOSSY;
    }

    ax_device_callable_inlined void invertEta() { ior = 1.f / ior; }
  };

  template<class T>
  class BaseConductorBxDF {
   protected:
    VNDF ggx;

   public:
    ax_device_callable_inlined BaseConductorBxDF(float roughness) : ggx(roughness) {}

    ax_device_callable_inlined BaseConductorBxDF(float anisotropy, float roughness) : ggx(anisotropy, roughness) {}

    ax_device_callable_inlined Spectrum f(const glm::vec3 &wo, const glm::vec3 &wi, TRANSPORT mode = TRANSPORT::RADIANCE) const {
      if (!bxdf::same_hemisphere(wo, wi))
        return 0.f;
      if (ggx.isFullSpecular())
        return 0.f;

      // Don't process grazing angles
      float costheta_wo = bxdf::abscostheta(wo);
      float costheta_wi = bxdf::abscostheta(wi);

      if (costheta_wi == 0.f || costheta_wo == 0.f)
        return 0.f;

      glm::vec3 wm = glm::normalize(wi + wo);
      if (glm::length2(wm) == 0.f)
        return 0.f;

      Spectrum fresnel = static_cast<const T *>(this)->computeFresnel(bxdf::absdot(wo, wm));
      return ggx.D(wm) * fresnel * ggx.G(wo, wi) / (4 * costheta_wi * costheta_wo);
    }

    ax_device_callable_inlined float pdf(const glm::vec3 &wo,
                                         const glm::vec3 &wi,
                                         TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                         REFLTRANSFLAG sample_flag = REFLTRANSFLAG::ALL) const {
      if (!(sample_flag & REFLTRANSFLAG::REFL))
        return 0.f;
      if (!bxdf::same_hemisphere(wi, wo))
        return 0.f;
      if (ggx.isFullSpecular())
        return 0.f;

      glm::vec3 wm = glm::normalize(wi + wo);  // Gets microfacet's normal
      if (glm::length2(wm) == 0.f)
        return 0.f;
      if (!bxdf::same_hemisphere(wm, glm::vec3(0.f, 0.f, 1.f)))
        wm.z = -wm.z;

      /* From the law of specular reflection :
       * w_r = -w_o + 2(w_m.w_o)w_m
       * dw_m/dw_i = sintheta_m * dtheta_m * dphi_m/sin2theta_m * 2 * dtheta_m * dphi_m
       * = 1 / 4(w_o . w_m)
       */
      return ggx.pdf(wo, wm) / 4 * bxdf::absdot(wo, wm);
    }

    ax_device_callable_inlined bool sample_f(const glm::vec3 &wo,
                                             float uc,
                                             const float u[2],
                                             BSDFSample *sample,
                                             TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                             REFLTRANSFLAG sample_flag = REFLTRANSFLAG::ALL) const {

      AX_ASSERT_NOTNULL(sample);
      if (!(sample_flag & REFLTRANSFLAG::REFL))
        return false;
      // Models a dirac delta distribution , returns 1.f for pdf as convention.
      if (ggx.isFullSpecular()) {
        glm::vec3 wi = glm::reflect(-wo, glm::vec3(0.f, 0.f, 1.f));
        float costheta_i = bxdf::abscostheta(wi);
        if (costheta_i == 0.f)
          return false;
        float costheta_o = bxdf::abscostheta(wo);
        Spectrum fresnel = static_cast<const T *>(this)->computeFresnel(costheta_o) / costheta_i;
        sample->costheta = costheta_i;
        sample->eta = 0.f;
        sample->f = fresnel;
        sample->pdf = 1.f;
        sample->pdf_cosine_weighted = false;
        sample->wi = wi;
        sample->flags = BXDFFLAGS::SPECULAR_REFLECTION;
        return true;
      }

      // Models specular roughness.
      if (bxdf::costheta(wo) == 0.f)
        return false;
      glm::vec3 wm = ggx.sampleGGXVNDF(wo, u);
      glm::vec3 wi = glm::reflect(-wo, wm);
      if (!bxdf::same_hemisphere(wi, wo))
        return false;

      float costheta_i = bxdf::abscostheta(wi);
      float costheta_o = bxdf::abscostheta(wo);
      float wm_dot_wo = bxdf::absdot(wm, wo);
      if (costheta_i == 0.f || costheta_o == 0.f || wm_dot_wo == 0.f)
        return false;

      Spectrum fresnel = static_cast<const T *>(this)->computeFresnel(wm_dot_wo);
      float pdf = ggx.pdf(wo, wm) / (4 * bxdf::absdot(wo, wm));
      if (ISNAN(pdf) || ISINF(pdf))
        return false;
      sample->costheta = costheta_i;
      sample->eta = 0.f;
      Spectrum f = ggx.D(wm) * fresnel * ggx.G(wo, wi) / (4 * costheta_i * costheta_o);
      sample->f = f;
      sample->pdf = pdf;
      sample->pdf_cosine_weighted = false;
      sample->wi = wi;
      sample->flags = BXDFFLAGS::GLOSSY_REFLECTION;
      return true;
    }

    ax_device_callable_inlined unsigned flags() const { return ggx.isFullSpecular() ? BXDFFLAGS::SPECULAR_REFLECTION : BXDFFLAGS::GLOSSY_REFLECTION; }

    ax_device_callable_inlined void invertEta() {}
  };

  class SchlickConductorBxDF : public BaseConductorBxDF<SchlickConductorBxDF> {
    Spectrum albedo;
    float metallic{1.f};

   public:
    ax_device_callable_inlined SchlickConductorBxDF(Spectrum albedo, float metal, float roughness = 0.f, float anisotropy = 1.f)
        : albedo(albedo), BaseConductorBxDF(anisotropy, roughness) {}

    ax_device_callable_inlined Spectrum computeFresnel(float abscostheta_i) const { return Fresnel::schlick(abscostheta_i, albedo); }
  };

  class ConductorBxDF : public BaseConductorBxDF<ConductorBxDF> {
    Spectrum k, eta;

   public:
    ax_device_callable_inlined ConductorBxDF(Spectrum eta, Spectrum k, float roughness) : BaseConductorBxDF(roughness), eta(eta), k(k) {}
    ax_device_callable_inlined ConductorBxDF(Spectrum eta, Spectrum k, float anisotropy, float roughness)
        : eta(eta), k(k), BaseConductorBxDF(anisotropy, roughness) {}

    ax_device_callable_inlined Spectrum computeFresnel(float abscostheta_i) const {
      Spectrum ret(0.f);
      for (unsigned i = 0; i < Spectrum::SPECTRUM_SAMPLES; i++) {
        Fresnel fresnel(eta[i], k[i]);
        ret.samples[i] = fresnel.complex(abscostheta_i);
      }
      return ret;
    }
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
      if (!(flag & REFLTRANSFLAG::REFL))
        return 0;
      return bxdf::abscostheta(wi) * (float)INV_PI;
    }

    ax_device_callable_inlined bool sample_f(const glm::vec3 & /*wo*/,
                                             float uc,  // randomly choses between scattering types (reflection / transmission)
                                             const float u[2],
                                             BSDFSample *sample,
                                             TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                             REFLTRANSFLAG sample_flag = REFLTRANSFLAG::ALL) const {
      if (!(sample_flag & REFLTRANSFLAG::REFL))
        return false;
      AX_ASSERT_NOTNULL(sample);
      glm::vec3 wi = bxdf::hemisphere_cosine_sample(u);
      float costheta = bxdf::abscostheta(wi);
      sample->wi = wi;
      sample->pdf = costheta * (float)INV_PI;
      sample->flags = BXDFFLAGS::DIFFUSE_REFLECTION;
      sample->pdf_cosine_weighted = false;
      sample->f = R * INV_PI;
      sample->eta = 1.f;
      sample->costheta = costheta;
      return true;
    }

    ax_device_callable_inlined unsigned flags() const { return R ? BXDFFLAGS::DIFFUSE_REFLECTION : BXDFFLAGS::NONE; }

    ax_device_callable_inlined void invertEta() {}
  };

  class DisneyPrincipledBxDF {
    bsdf_params_s params;

    struct lobe_params_s {
      Spectrum f{};
      Spectrum eta{};
      glm::vec3 wi;
      float pdf{1.f};
      BXDFFLAGS flag;
      bool cosine_weighted{true};
      bool sampled_success{true};
    };
    struct lobe_weights_s {
      float diffuse, sheen, clearcoat, transmission, specular;
    };

    ax_device_callable_inlined lobe_params_s diffuse(const glm::vec3 &wo, const float u[2], const lobe_weights_s &weights) const {

      auto fresnel = [](const glm::vec3 &w, float value) { return 1.f + (value - 1) * std::powf(1.f - bxdf::abscostheta(w), 5); };

      glm::vec3 wi = glm::normalize(bxdf::hemisphere_cosine_sample(u));
      glm::vec3 h = glm::normalize(wi + wo);

      float fd90 = 0.5f + 2 * params.roughness * math::sqr(bxdf::absdot(wi, h));
      Spectrum fd = params.albedo * M_1_PIf * fresnel(wi, fd90) * fresnel(wo, fd90);

      // TODO: Could replace this part with a subsurf lobe, but needs volumetrics.
      float fss90 = params.roughness * math::sqr(bxdf::absdot(h, wi));

      Spectrum fss = 1.25f * params.albedo * M_1_PIf *
                     (0.5f + fresnel(wi, fss90) * fresnel(wo, fss90) * (-0.5f + 1 / (bxdf::abscostheta(wo) + bxdf::abscostheta(wi))));

      Spectrum f_diffuse = (1 - params.subsurface_scattering) * fd + params.subsurface_scattering * fss;
      float pdf = bxdf::abscostheta(wi) * M_1_PIf;
      return {f_diffuse, params.eta, wi, pdf, DIFFUSE, false, true};
    }

    ax_device_callable_inlined lobe_params_s specular(const glm::vec3 &wo, const float u[2], const lobe_weights_s &weights) const {
      NDF ggx(params.anisotropy_ratio, params.roughness);

      auto fresnel = [](float costheta, float metal, const Spectrum &albedo) {
        Spectrum F0 = 0.04;
        F0 = math::lerp(F0, albedo, Spectrum(metal));
        return F0 + (1 - F0) * std::powf(1 - costheta, 5);
      };

      glm::vec3 wm = ggx.sampleGGXVNDF(wo, u);
      glm::vec3 wi = glm::reflect(-wo, wm);
      float costheta_i = bxdf::abscostheta(wi);
      float costheta_o = bxdf::abscostheta(wo);
      float wm_dot_wo = bxdf::absdot(wm, wo);

      if (!bxdf::same_hemisphere(wo, wi) || costheta_i < 1e-4f || costheta_o < 1e-4f)
        return failedLobe();

      Spectrum F0 = fresnel(bxdf::absdot(wo, wm), params.metal, params.albedo);
      Spectrum f = F0 * ggx.D(wm) * ggx.G1(wi) * ggx.G1(wo) / (4 * costheta_o * costheta_i);
      float pdf = ggx.D(wm) * ggx.G1(wo) / (4 * costheta_o);
      if (!f.isValid() || ISNAN(pdf) || ISINF(pdf))
        return failedLobe();
      return {f, params.eta, wi, pdf, GLOSSY_REFLECTION, false, true};
    }

    ax_device_callable_inlined lobe_params_s transmission(const glm::vec3 &wo, float uc, const float u[2], const lobe_weights_s &weights) const {

      NDF ggx(params.anisotropy_ratio, params.roughness);
      glm::vec3 wm = ggx.sampleGGXVNDF(wo, u);

      float eta = params.eta[0];
      Fresnel fresnel(eta);

      glm::vec3 wi;
      bool no_tir = true;
      eta = bxdf::refract(wo, wm, eta, no_tir, wi);
      if (bxdf::same_hemisphere(wi, wo) || wi.z == 0.f || !no_tir)
        return failedLobe();

      float costheta_i = bxdf::costheta(wi);
      float costheta_o = bxdf::costheta(wo);
      float refract_eta = 0.f;
      Spectrum T = (1.f - fresnel.real(glm::dot(wo, wm), refract_eta)) * params.albedo;

      float denom = math::sqr(glm::dot(wi, wm) + glm::dot(wo, wm) / eta);
      float jacobian = bxdf::absdot(wi, wm) / denom;
      float pdf = ggx.pdf(wo, wm) * jacobian;
      AX_ASSERT(!ISNAN(pdf) && !ISINF(pdf), "Rough dielectric transmission pdf returned nan or inf.");
      Spectrum f = T * ggx.D(wm) * ggx.G(wo, wi) * fabsf(glm::dot(wi, wm) * glm::dot(wo, wm) / (costheta_i * costheta_o * denom));
      f /= math::sqr(eta);

      if (!f.isValid())
        return failedLobe();

      return {f, params.eta, wi, pdf, GLOSSY_TRANSMISSION, false, true};
    }

    ax_device_callable_inlined lobe_weights_s generateWeights() const {
      float w_diffuse = (1 - params.transmission) * (1 - params.metal);
      float w_sheen = (1 - params.metal) * params.sheen;
      float w_clearcoat = 0.25f * params.clearcoat;
      float w_specular = params.specular;
      float w_transmission = (1.f - params.metal) * params.transmission;

      float W = w_diffuse + w_sheen + w_clearcoat + w_specular + w_transmission;

      lobe_weights_s weights{};
      if (W == 0) {
        weights.specular = 1.f;
        return weights;
      }

      weights.diffuse = w_diffuse / W;
      weights.sheen = w_sheen / W;
      weights.clearcoat = w_clearcoat / W;
      weights.transmission = w_transmission / W;
      weights.specular = w_specular / W;
      return weights;
    }

    ax_device_callable_inlined unsigned evalLobeType(float uc, const lobe_weights_s &weights) const {
      float weight_index = 0.f;
      if (uc < (weight_index += weights.diffuse))
        return DIFFUSE;
      if (uc < (weight_index += weights.specular))
        return SPECULAR;
      if (uc < (weight_index += weights.clearcoat))
        return DIFFUSE | SPECULAR | GLOSSY;
      if (uc < (weight_index += weights.sheen))
        return GLOSSY | REFLECTION;
      if (uc < (weight_index += weights.transmission))
        return TRANSMISSION;
      AX_UNREACHABLE;
      return -1;
    }

    ax_device_callable_inlined lobe_params_s sampleLobe(const glm::vec3 &wo, float uc, const float u[2], const lobe_weights_s &weights) const {
      unsigned flag = evalLobeType(uc, weights);
      switch (flag) {
        case DIFFUSE:
          return diffuse(wo, u, weights);
        case SPECULAR:
          return specular(wo, u, weights);
        case TRANSMISSION:
          return transmission(wo, uc, u, weights);
      }
      AX_UNREACHABLE;
      return {};
    }

    ax_device_callable_inlined lobe_params_s failedLobe() const {
      lobe_params_s fail;
      fail.sampled_success = false;
      return fail;
    }

   public:
    ax_device_callable_inlined DisneyPrincipledBxDF(const bsdf_params_s &bsdf_parameters) : params(bsdf_parameters) {}

    ax_device_callable_inlined Spectrum f(const glm::vec3 &wo, const glm::vec3 &wi, TRANSPORT mode = TRANSPORT::RADIANCE) const {}

    ax_device_callable_inlined float pdf(const glm::vec3 &wo,
                                         const glm::vec3 &wi,
                                         TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                         REFLTRANSFLAG flag = REFLTRANSFLAG::ALL) const {}

    ax_device_callable_inlined bool sample_f(const glm::vec3 &wo,
                                             float uc,
                                             const float u[2],
                                             BSDFSample *sample,
                                             TRANSPORT transport_mode = TRANSPORT::RADIANCE,
                                             REFLTRANSFLAG sample_flag = REFLTRANSFLAG::ALL) const {
      lobe_weights_s weights = generateWeights();
      lobe_params_s sampled_lobe = sampleLobe(wo, uc, u, weights);
      if (!sampled_lobe.sampled_success)
        return false;
      sample->f = sampled_lobe.f;
      sample->pdf = sampled_lobe.pdf;
      sample->wi = sampled_lobe.wi;
      sample->costheta = bxdf::abscostheta(sampled_lobe.wi);
      sample->pdf_cosine_weighted = sampled_lobe.cosine_weighted;
      sample->eta = sampled_lobe.eta;
      sample->flags = sampled_lobe.flag;
      return true;
    }

    // Special case , since sampling decision happens inside this bsdf for simplicity.
    ax_device_callable_inlined unsigned flags() const { return ALL; }

    ax_device_callable_inlined void invertEta() {}
  };

#define LISTBxDF DiffuseBxDF, DisneyPrincipledBxDF, ConductorBxDF, SchlickConductorBxDF, DielectricBxDF

  class BxDF : public core::tag_ptr<LISTBxDF> {
   public:
    using tag_ptr::tag_ptr;

    ax_device_callable_inlined void invertEta() {  // TODO: Remove asap.
      auto d = [&](auto bxdf) { bxdf->invertEta(); };
      dispatch(d);
    }

    /**
     * @brief Evalutes BxDF
     *
     * @param wo view direction. Points outside.
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
     * @param wo view direction . Points outside.
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
     * @param wo view direction, points outside.
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
     * @param wo Direction vector, points outwards.
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
}  // namespace nova
#endif

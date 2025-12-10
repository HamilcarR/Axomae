#include "Test.h"
#include "material/BxDF_flags.h"
#include "material/BxDF_math.h"
#include "sampler/Sampler.h"
#include "spectrum/Spectrum.h"
#include <gtest/gtest.h>
#include <internal/common/math/math_random.h>
#include <internal/common/math/math_spherical.h>
#include <nova/material/BxDF.h>

namespace nm = nova;

static constexpr unsigned RAND_NUM = 10;
static constexpr float uc_samples[RAND_NUM][2] = {{0.0f, 0.0f},
                                                  {0.1f, 0.5f},
                                                  {0.2f, 0.25f},
                                                  {0.3f, 0.75f},
                                                  {0.4f, 0.125f},
                                                  {0.5f, 0.625f},
                                                  {0.6f, 0.375f},
                                                  {0.7f, 0.875f},
                                                  {0.8f, 0.0625f},
                                                  {0.9f, 0.5625f}};

static constexpr float roughness_values[6] = {1e-4f, 0.1f, 0.45f, 0.75f, 1.f};

constexpr float IOR_VACUUM = 1.00000f;
constexpr float IOR_AIR = 1.00028f;
constexpr float IOR_ICE = 1.309f;
constexpr float IOR_WATER = 1.333f;
constexpr float IOR_ETHANOL = 1.361f;
constexpr float IOR_GLASS = 1.50f;
constexpr float IOR_ACRYLIC = 1.490f;
constexpr float IOR_POLYCARB = 1.585f;
constexpr float IOR_DIAMOND = 2.417f;

/********************** Fresnel **********************/

struct fresnel_result {
  float eta;
  float R;
};

static fresnel_result test_fresnel_real(float ior, float angle_radian) {
  Fresnel real_fresnel(ior);
  float new_eta = 0.f;
  float R = real_fresnel.real(angle_radian, new_eta);
  return {new_eta, R};
}

TEST(FresnelTest, real) {
  fresnel_result result = test_fresnel_real(IOR_GLASS, 0.5);
  ASSERT_EQ(IOR_GLASS, result.eta);     // IOR should stay the same since we're on the upper hemisphere.
  EXPECT_GT(1.f - result.R, result.R);  // at 45° , transmission component should be greater than reflection for glass.

  result = test_fresnel_real(IOR_GLASS, 0.001);
  ASSERT_EQ(IOR_GLASS, result.eta);
  EXPECT_GT(result.R, 1.f - result.R);

  result = test_fresnel_real(IOR_GLASS, -0.001);
  ASSERT_NE(IOR_GLASS, result.eta);

  result = test_fresnel_real(IOR_VACUUM, 0.5f);
  ASSERT_EQ(IOR_VACUUM, result.eta);
  EXPECT_GT(1.f - result.R, result.R);  // Vacuum has only transmission.

  result = test_fresnel_real(IOR_DIAMOND, 0.1f);
  EXPECT_GT(result.R, 1.f - result.R);
}

/********************** GGX / VNDF **********************/
TEST(GGXTest, SampleWm) {

  glm::vec3 wo(1.2f, 0.4f, 1.f);

  for (float roughness : roughness_values)
    for (int i = 0; i < 5; i++) {
      NDF test_ggx(roughness);
      const float *uc = uc_samples[i];
      glm::vec3 wm = test_ggx.sampleGGXVNDF(wo, uc);

      // Sampled direction should be normalized.
      EXPECT_NEAR(glm::length2(wm), 1.f, 1e-4f);

      // Sampled microfacet normal should always be in the same hemisphere as the view direction.
      ASSERT_TRUE(bxdf::same_hemisphere(wm, wo));
    }
}

/* Tests the behavior of the masking function G1().
 * Should return values close to 1 with a vector colinear with the facet's normal , and close to 0 for grazing angles.
 */
TEST(GGXTest, G1_bounds) {
  NDF test_ggx(0.75f);
  // Visible micrafacets from top view.
  glm::vec3 sample_vector(0.f, 0.f, 1.f);
  float visible_micrafacets_probability = test_ggx.G1(sample_vector);
  EXPECT_GT(visible_micrafacets_probability, 0);
  EXPECT_LE(visible_micrafacets_probability, 1);
  EXPECT_NEAR(visible_micrafacets_probability, 1, 1e-3f);

  // Visible microfacets at grazing angle.
  sample_vector = glm::vec3(1.f, 1.f, 1e-4f);
  visible_micrafacets_probability = test_ggx.G1(sample_vector);
  EXPECT_GE(visible_micrafacets_probability, 0);
  EXPECT_LT(visible_micrafacets_probability, 1.f);
  EXPECT_NEAR(visible_micrafacets_probability, 0.f, 1e-2f);
}

/********************** BSDFs **********************/

/************* Diffuse **************/

/* Tests f(wo , wi , p) = f(wi , wo , p). */
template<class BxDF>
void test_reciprocity(const BxDF &bxxdf) {
  glm::vec3 wo(0.5f, 0.5f, 0.5f);
  glm::vec3 wi(0.f, 0.f, 1.f);
  nova::Spectrum R = bxxdf.f(wo, wi);
  nova::Spectrum RR = bxxdf.f(wi, wo);
  ASSERT_EQ(R, RR);
}

/* Tests for energy conservation :
 * $ \int f(wo , wi , p).costheta dw <= 1 $
 */
template<class BxDF>
void test_energy_conservation(const BxDF &bxxdf) {
  nova::sampler::SobolSampler sobol;
  std::vector<nova::uniform_sample2d> u0, u1;
  std::vector<float> uc;

  for (unsigned i = 0; i < 100; i++) {
    sobol.reset(i);
    nova::uniform_sample2d v0{}, v2{};
    sobol.sample2D(v0.u);
    float v1 = sobol.sample1D();
    sobol.sample2D(v2.u);
    u0.push_back(v0);
    u1.push_back(v2);
    uc.push_back(v1);
  }
  axstd::cspan<nova::uniform_sample2d> s0(u0);
  axstd::cspan<nova::uniform_sample2d> s1(u1);
  axstd::cspan<float> sc(uc);
  nova::Spectrum R = bxxdf.rho(s0, sc, s1);

  ASSERT_LT(R.max(), 1.f) << "Spectrum: " << R.max();
}

TEST(DiffuseBxDFTest, f) {
  nm::DiffuseBxDF diffuse(1.f);
  glm::vec3 wo(0.5f, 0.5f, 0.5f);
  glm::vec3 wi(0.f, 0.f, 1.f);
  nova::Spectrum R = diffuse.f(wo, wi);
  ASSERT_TRUE(R);

  wo = glm::vec3(0.5f, 0.5f, 0.5f);
  wi = glm::vec3(0.f, 0.f, -1.f);
  R = diffuse.f(wo, wi);
  ASSERT_FALSE(R);
}

TEST(DiffuseBxDFTest, reciprocity) {
  nm::DiffuseBxDF diffuse(1.f);
  test_reciprocity(diffuse);
}

TEST(DiffuseBxDFTest, energy_conservation) {
  nm::DiffuseBxDF diffuse(0.f);
  nm::BxDF bxxdf = &diffuse;
  test_energy_conservation(bxxdf);
}

TEST(DiffuseBxDFTest, pdf) {
  nm::DiffuseBxDF diffuse(1.f);
  glm::vec3 wo(0.5f, 0.5f, 0.5f);
  glm::vec3 wi(0.f, 0.f, 1.f);
  float pdf = diffuse.pdf(wo, wi);
  ASSERT_NEAR(pdf, bxdf::abscostheta(wi) * (float)INV_PI, 1e-6f);

  wo = glm::vec3(0.5f, 0.5f, 0.5f);
  wi = glm::vec3(0.f, 0.f, -1.f);
  pdf = diffuse.pdf(wi, wo);
  ASSERT_FALSE(pdf);
}

TEST(DiffuseBxDF, sample_f) {
  nm::DiffuseBxDF diffuse(1.f);
  glm::vec3 wo(0.5f, 0.5f, 0.5f);
  float uc = 0.533f;
  float u[2] = {0.33f, 1.35f};
  nm::BSDFSample sample;
  bool is_sampled = diffuse.sample_f(wo, uc, u, &sample, nm::RADIANCE, nm::REFLTRANSFLAG::NONE);
  ASSERT_FALSE(is_sampled);
}

/************* Dielectric **************/
TEST(DielectricBxDF, sample_f) {
  // Tests specular dielectric transmission.
  nm::DielectricBxDF dielectric(1.f, 0.f);
  glm::vec3 wo(0, 0, 1.f);
  float uc = 0.533f;
  float u[2] = {0.33f, 1.35f};
  nm::BSDFSample sample;
  bool is_sampled = dielectric.sample_f(wo, uc, u, &sample, nm::RADIANCE, nm::REFLTRANSFLAG::TRAN);
  ASSERT_TRUE(is_sampled);
  EXPECT_NEAR(sample.costheta, 1.f, 1e-5f);  // Transmission in a medium with IOR of vacuum shouldn't have deviation from the normal.
  EXPECT_LE(sample.f, 1.f);
}

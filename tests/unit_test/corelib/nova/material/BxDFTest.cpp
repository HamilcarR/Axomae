#include "Test.h"
#include "material/BxDF_flags.h"
#include "sampler/Sampler.h"
#include "spectrum/Spectrum.h"
#include <gtest/gtest.h>
#include <internal/common/math/math_random.h>
#include <internal/common/math/math_spherical.h>
#include <nova/material/BxDF.h>

namespace nm = nova::material;

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
  std::vector<uniform_sample2d> u0, u1;
  std::vector<float> uc;

  for (unsigned i = 0; i < 100; i++) {
    sobol.reset(i);
    uniform_sample2d v0{}, v2{};
    sobol.sample2D(v0.u);
    float v1 = sobol.sample1D();
    sobol.sample2D(v2.u);
    u0.push_back(v0);
    u1.push_back(v2);
    uc.push_back(v1);
  }
  axstd::cspan<uniform_sample2d> s0(u0);
  axstd::cspan<uniform_sample2d> s1(u1);
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
  ASSERT_NEAR(pdf, bxdf::costheta(wi) * (float)INV_PI, 1e-6f);

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
#include "sampler/Sampler.h"
#include "Test.h"

const int MAX_ITER = 100;

TEST(SamplerTest, cpu_sample) {
  math::random::CPUQuasiRandomGenerator generator;
  nova::sampler::SobolSampler sobol(generator);
  bool all_zero = true;
  for (int i = 0; i < MAX_ITER; i++) {
    glm::vec3 sampled = sobol.sample();
    EXPECT_GT(sampled.x, -1);
    EXPECT_GT(sampled.y, -1);
    EXPECT_GT(sampled.z, -1);

    EXPECT_LT(sampled.x, 1);
    EXPECT_LT(sampled.y, 1);
    EXPECT_LT(sampled.z, 1);
    if (sampled.x != 0 || sampled.y != 0 || sampled.z != 0)
      all_zero = false;
  }
  EXPECT_FALSE(all_zero);
  all_zero = true;
  math::random::CPUPseudoRandomGenerator pgenerator;
  nova::sampler::RandomSampler rand(pgenerator);
  for (int i = 0; i < MAX_ITER; i++) {
    glm::vec3 sampled = rand.sample();
    EXPECT_GT(sampled.x, -1);
    EXPECT_GT(sampled.y, -1);
    EXPECT_GT(sampled.z, -1);

    EXPECT_LT(sampled.x, 1);
    EXPECT_LT(sampled.y, 1);
    EXPECT_LT(sampled.z, 1);
    if (sampled.x != 0 || sampled.y != 0 || sampled.z != 0)
      all_zero = false;
  }
  EXPECT_FALSE(all_zero);
}

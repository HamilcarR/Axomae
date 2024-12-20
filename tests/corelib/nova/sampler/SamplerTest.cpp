#include "sampler/Sampler.h"
#include "Test.h"

const int MAX_ITER = 100;

TEST(SamplerTest, cpu_sample) {
  math::random::CPUQuasiRandomGenerator generator;
  nova::sampler::SobolSampler sobol(generator);
  for (int i = 0; i < MAX_ITER; i++) {
    glm::vec3 sampled = sobol.sample();
    EXPECT_TRUE(sampled.x > 0 || sampled.y > 0 || sampled.z > 0);
    EXPECT_GT(sampled.x, -1);
    EXPECT_GT(sampled.y, -1);
    EXPECT_GT(sampled.z, -1);

    EXPECT_LT(sampled.x, 1);
    EXPECT_LT(sampled.y, 1);
    EXPECT_LT(sampled.z, 1);
  }

  math::random::CPUPseudoRandomGenerator pgenerator;
  nova::sampler::RandomSampler rand(pgenerator);
  for (int i = 0; i < MAX_ITER; i++) {
    glm::vec3 sampled = rand.sample();
    EXPECT_TRUE(sampled.x > 0 || sampled.y > 0 || sampled.z > 0);
    EXPECT_GT(sampled.x, -1);
    EXPECT_GT(sampled.y, -1);
    EXPECT_GT(sampled.z, -1);

    EXPECT_LT(sampled.x, 1);
    EXPECT_LT(sampled.y, 1);
    EXPECT_LT(sampled.z, 1);
  }
}

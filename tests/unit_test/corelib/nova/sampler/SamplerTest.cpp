#include "sampler/Sampler.h"
#include <unit_test/Test.h>

const int MAX_ITER = 100;

// Tests that sobol returns values [-1 , 1) , ie  [0 , 1)
TEST(SobolSamplerTest, sampler3D) {
  math::random::SobolGenerator generator;
  nova::sampler::SobolSampler sobol(generator);
  bool all_zero = true;
  for (int i = 0; i < MAX_ITER; i++) {
    sobol.reset(i);

    float out[3] = {};
    sobol.sample3D(out);

    glm::vec3 sampled;
    std::memcpy(&sampled, out, sizeof(out));

    EXPECT_GE(sampled.x, -1) << sampled.x;
    EXPECT_GE(sampled.y, -1) << sampled.y;
    EXPECT_GE(sampled.z, -1) << sampled.z;

    EXPECT_LT(sampled.x, 1) << sampled.x;
    EXPECT_LT(sampled.y, 1) << sampled.y;
    EXPECT_LT(sampled.z, 1) << sampled.z;
    if (sampled.x != 0 || sampled.y != 0 || sampled.z != 0)
      all_zero = false;
  }
  EXPECT_FALSE(all_zero);
}

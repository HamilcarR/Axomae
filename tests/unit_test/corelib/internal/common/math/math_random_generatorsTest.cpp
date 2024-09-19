#include "Test.h"
#include <internal/common/math/math_random.h>

namespace mr = math::random;

TEST(math_random_generatorsTest, generate) {
  mr::SobolGenerator test_generator;
  for (unsigned i = 0; i < 10; i++) {
    float random = test_generator.generate(i, 0);
    ASSERT_LT(random, 1.f);
    ASSERT_GE(random, 0.f);
  }
}

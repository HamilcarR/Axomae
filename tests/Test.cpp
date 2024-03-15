#include "Test.h"
#include "math_utils.h"
#include <gtest/gtest.h>

int main(int argv, char **argc) {
  math::random::init_rand();
  ::testing::InitGoogleTest(&argv, argc);
  auto a = RUN_ALL_TESTS();
  return a;
}
#include "Test.h"
#include <cstring>
#include <internal/common/axstd/span.h>
#include <internal/common/math/math_random.h>

#if __cplusplus < 202002L

static constexpr char test_string[] = "hello world";

TEST(span_test , copy_container) {
  std::vector<float> float_vec;
  math::random::CPUPseudoRandomGenerator prng;
  for (std::size_t i = 0 ; i < 100 ; i++)
    float_vec.push_back(prng.nrandf(-300.f , 300.f));
  axstd::span<float> float_span(float_vec);
  for (std::size_t i = 0 ; i < 100 ; i++)
    ASSERT_EQ(float_span[i], float_vec[i]);
  axstd::span<float> float_span2 = float_vec;
  for (std::size_t i = 0 ; i < 100 ; i++)
    ASSERT_EQ(float_span2[i], float_vec[i]);
}

TEST(span_test , front) {
  constexpr axstd::span s(test_string, std::size(test_string));
  ASSERT_EQ(s.front(), 'h');
}

TEST(span_test , back) {
  constexpr axstd::span s(test_string, std::size(test_string));
  ASSERT_EQ(s.back(), 'd');
}

TEST(span_test , data) {
  constexpr axstd::span s(test_string, std::size(test_string));
  ASSERT_EQ(s.data() , test_string);
}

TEST(span_test , index_operator) {
  constexpr axstd::span s(test_string, std::size(test_string));
  for (int i = 0; i < s.size(); ++i) {
    ASSERT_EQ(test_string[i] , s[i]);
  }
}

TEST(span_test , size) {
  constexpr axstd::span s(test_string, std::size(test_string));
  ASSERT_EQ(s.size(), std::size(test_string));
}

TEST(span_test , size_bytes) {
  const int seq[] = {1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9};
  const axstd::span s(seq, std::size(seq));
  ASSERT_EQ(s.size_bytes() , std::size(seq) * sizeof(int));
}

TEST(span_test , empty ) {
  axstd::span<char> s;
  ASSERT_EQ(s.empty(), true);
}

TEST(span_test , subspan) {
  constexpr std::size_t size_str = std::size(test_string) - 1;
  constexpr axstd::span s = axstd::span(test_string, size_str);
  constexpr axstd::span sub = s.subspan(6, size_str-6);
  ASSERT_TRUE(std::strncmp(sub.data(), "world" , 5) == 0);
  constexpr axstd::span static_sub = s.subspan<6 , size_str-6>();
  ASSERT_TRUE(std::strncmp(static_sub.data(), "world" , 5) == 0);
  constexpr axstd::span sub2 = s.subspan(0 , 6);
  ASSERT_TRUE(std::strncmp(sub2.data(), "hello" , 5) == 0);
}


TEST(span_test , first ) {
  constexpr axstd::span s(test_string, strlen(test_string));
  constexpr axstd::span first = s.first(5);
  ASSERT_TRUE(std::strncmp(first.data(), "hello" , 5) == 0);
  constexpr axstd::span first2 = s.first<5>();
  ASSERT_TRUE(std::strncmp(first2.data(), "hello" , 5) == 0);
}


TEST(span_test , last ) {
  constexpr axstd::span s(test_string, strlen(test_string));
  constexpr axstd::span last = s.last(5);
  ASSERT_TRUE(std::strncmp(last.data(), "world" , 5) == 0);
  constexpr axstd::span last2 = s.last<5>();
  ASSERT_TRUE(std::strncmp(last2.data(), "world" , 5) == 0);
}

#endif
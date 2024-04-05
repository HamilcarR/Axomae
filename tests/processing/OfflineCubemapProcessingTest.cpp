#include "OfflineCubemapProcessing.h"
#include "Test.h"
#include "math.h"

#define f_rand rand() / ((float)RAND_MAX)

constexpr unsigned int NUM_CHANNELS = 3;
template<typename T>
class TextureBuilder {
 public:
  struct TexStruct {
    std::vector<T> data;
    unsigned width;
    unsigned height;
  };

 public:
  TextureBuilder(unsigned _width, unsigned _height, bool random = true, T value_if_not_random = 1.f) {
    srand(time(nullptr));
    width = _width;
    height = _height;
    if (random)
      for (unsigned i = 0; i < width * height * NUM_CHANNELS; i++)
        data.push_back(f_rand);
    else
      for (unsigned i = 0; i < width * height * NUM_CHANNELS; i++)
        data.push_back(value_if_not_random);
  }

  virtual ~TextureBuilder() {}

  const TexStruct getData() {
    TexStruct tex;
    tex.data = data;
    tex.width = width;
    tex.height = height;
    return tex;
  }

 public:
  std::vector<T> data;
  unsigned width;
  unsigned height;
};

template<class T>
class RandomVecBuilder {
 public:
  RandomVecBuilder() { srand(time(nullptr)); }
  virtual ~RandomVecBuilder() {}
  T generate() { return T(f_rand); }
};

TEST(EnvmapComputation, WrapPixelCoords) {
  TextureBuilder<float> builder(128, 256);
  EnvmapProcessing process(builder.getData().data, builder.getData().width, builder.getData().height);
  std::vector<std::pair<glm::dvec2, glm::dvec2>> test_cases = {std::pair(glm::dvec2(0, 0), glm::dvec2(0, 0)),
                                                               std::pair(glm::dvec2(128, 256), glm::dvec2(0, 0)),
                                                               std::pair(glm::dvec2(129, 0), glm::dvec2(1, 0)),
                                                               std::pair(glm::dvec2(129, -257), glm::dvec2(1, 255))};
  for (auto A : test_cases) {
    glm::dvec2 case_test = A.first;
    glm::dvec2 expected_result = A.second;
    glm::dvec2 wrap_result = process.wrapAroundPixelCoords(case_test.x, case_test.y);
    EXPECT_EQ(wrap_result, expected_result);
  }
}

TEST(EnvmapComputation, discreteSample) {
  TextureBuilder<float> builder(16, 16, false, 1.f);
  EnvmapProcessing process(builder.getData().data, builder.getData().width, builder.getData().height);
  for (unsigned i = 0; i < builder.width; i++)
    for (unsigned j = 0; j < builder.height; j++) {
      ASSERT_EQ(process.discreteSample(i, j), glm::dvec3(1.f));
    }
}

TEST(EnvmapComputation, uvSphericalCohesion) {
  RandomVecBuilder<glm::dvec2> builder;
  for (unsigned i = 0; i < 100; i++) {
    glm::dvec2 uv = builder.generate();
    glm::dvec2 sph = math::spherical::uvToSpherical(uv);
    glm::dvec2 test = math::spherical::sphericalToUv(sph);
    EXPECT_LE(glm::length(test - uv), math::epsilon);
  }
}

TEST(EnvmapComputation, uvCartesianCohesion) {
  RandomVecBuilder<glm::dvec2> builder;
  for (unsigned i = 0; i < 100; i++) {
    glm::dvec2 sph = builder.generate();
    glm::dvec3 cart = math::spherical::sphericalToCartesian(sph);
    glm::dvec2 test = math::spherical::cartesianToSpherical(cart);
    EXPECT_LE(glm::length(test - sph), math::epsilon);
  }
}

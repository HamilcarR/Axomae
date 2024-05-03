#include "TextureProcessing.h"
#include "Test.h"
#include "math_random.h"

constexpr unsigned int NUM_CHANNELS = 3;
constexpr unsigned int SAMPLES = 500;
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
        data.push_back(math::random::nrandf(0, 1));
    else
      for (unsigned i = 0; i < width * height * NUM_CHANNELS; i++)
        data.push_back(value_if_not_random);
  }

  TexStruct getData() {
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
  T generate() { return T(math::random::nrandf(0, 1)); }
};

TEST(Texturing, WrapPixelCoords) {
  TextureBuilder<float> builder(128, 256);
  std::vector<float> texture_data_raw = builder.getData().data;
  TextureOperations<float> process(texture_data_raw, builder.getData().width, builder.getData().height);
  std::vector<std::pair<glm::vec2, glm::vec2>> test_cases = {std::pair(glm::vec2(0, 0), glm::vec2(0, 0)),
                                                             std::pair(glm::vec2(128, 256), glm::vec2(0, 0)),
                                                             std::pair(glm::vec2(129, 0), glm::vec2(1, 0)),
                                                             std::pair(glm::vec2(129, -257), glm::vec2(1, 255))};
  for (auto A : test_cases) {
    glm::vec2 case_test = A.first;
    glm::vec2 expected_result = A.second;
    glm::vec2 wrap_result = process.wrapAroundPixelCoords(case_test.x, case_test.y);
    EXPECT_EQ(wrap_result, expected_result);
  }
}

TEST(Texturing, discreteSample) {
  TextureBuilder<float> builder(16, 16, false, 1.f);
  std::vector<float> texture_data_raw = builder.getData().data;
  TextureOperations process(texture_data_raw, builder.getData().width, builder.getData().height);
  for (unsigned i = 0; i < builder.width; i++)
    for (unsigned j = 0; j < builder.height; j++) {
      glm::vec3 compared = glm::abs(process.discreteSample(i, j) - glm::vec3(1.f));
      glm::vec3 epsilon = glm::vec3(math::epsilon);
      ASSERT_LE(compared.x, epsilon.x);
      ASSERT_LE(compared.y, epsilon.y);
      ASSERT_LE(compared.z, epsilon.z);
    }
}

TEST(Texturing, uvSphericalCohesion) {
  RandomVecBuilder<glm::vec2> builder;
  for (unsigned i = 0; i < SAMPLES; i++) {
    glm::vec2 uv = builder.generate();
    glm::vec2 sph = math::spherical::uvToSpherical(uv);
    glm::vec2 test = math::spherical::sphericalToUv(sph);
    EXPECT_LE(glm::length(test - uv), math::epsilon);
  }
}

TEST(Texturing, uvCartesianCohesionNoApproximation) {
  RandomVecBuilder<glm::dvec2> builder;
  for (unsigned i = 0; i < SAMPLES; i++) {
    glm::dvec2 sph = builder.generate();
    glm::dvec3 cart = math::spherical::sphericalToCartesian(sph);
    glm::dvec2 test = math::spherical::cartesianToSpherical(cart, false);
    EXPECT_LE(glm::length(test - sph), math::epsilon);
  }
}

TEST(Texturing, uvCartesianCohesionApproximated) {
  float epsilon_ = 5e-03;
  RandomVecBuilder<glm::dvec2> builder;
  for (unsigned i = 0; i < SAMPLES; i++) {
    glm::dvec2 sph = builder.generate();
    glm::dvec3 cart = math::spherical::sphericalToCartesian(sph);
    glm::dvec2 test = math::spherical::cartesianToSpherical(cart);
    EXPECT_LE(glm::length(test - sph), epsilon_);
  }
}
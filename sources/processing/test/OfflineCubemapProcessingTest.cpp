#include "OfflineCubemapProcessing.h"
#include "Math.h"
#include "Test.h"

#define f_rand rand() / ((float)RAND_MAX)

constexpr unsigned int NUM_CHANNELS = 3;
template<typename T>
class TextureBuilder {
 public:
  struct TexStruct {
    T *data;
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
    tex.data = data.data();
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
      bool test_eq = process.discreteSample(i, j) == glm::dvec3(1.);
      ASSERT_EQ(process.discreteSample(i, j), glm::dvec3(1.f));
    }
}

TEST(EnvmapComputation, uvSphericalCohesion) {
  RandomVecBuilder<glm::dvec2> builder;
  for (unsigned i = 0; i < 100; i++) {
    glm::dvec2 uv = builder.generate();
    glm::dvec2 sph = spherical_math::uvToSpherical(uv);
    glm::dvec2 test = spherical_math::sphericalToUv(sph);
    EXPECT_LE(glm::length(test - uv), epsilon);
  }
}

TEST(EnvmapComputation, uvCartesianCohesion) {
  RandomVecBuilder<glm::dvec2> builder;
  for (unsigned i = 0; i < 100; i++) {
    glm::dvec2 sph = builder.generate();
    glm::dvec3 cart = spherical_math::sphericalToCartesian(sph);
    glm::dvec2 test = spherical_math::cartesianToSpherical(cart);
    EXPECT_LE(glm::length(test - sph), epsilon);
  }
}

/*TEST(EnvmapComputation , computeDiffuseIrradianceCPU){
  std::string image = "test2.hdr" ;
    int width = 0 ;
    int height = 0 ;
    unsigned _width = 512;
    unsigned _height = 256 ;
    int channels = 0 ;
    float *hdr_data = stbi_loadf( image.c_str() , &width , &height , &channels , 0);
    if(stbi_failure_reason())
    std::cout << stbi_failure_reason() << "\n";
    try{
        PerformanceLogger logger;

        EnvmapProcessing process(hdr_data , (unsigned) width , (unsigned) height);

        logger.startTimer();
        auto tex = process.computeDiffuseIrradiance(_width , _height , 1000);
        logger.endTimer();
        logger.print();
        stbi_write_hdr("response_cpu.hdr" , _width , _height , channels , tex->f_data);
        tex->clean();
    }
    catch(const std::exception &e){
        const char* exception = e.what();
        std::cout << exception << std::endl;
    }
    float *src_texture = new float[width * height * 4];
    float *dest_texture ;
    unsigned j = 0 ;
    for(unsigned i = 0 ; i < width * height * 3 ; i+=3){
       src_texture[j++] = hdr_data[i] ;
       src_texture[j++] = hdr_data[i + 1] ;
       src_texture[j++] = hdr_data[i + 2] ;
       src_texture[j++] = 0 ;
    }
    gpgpu_functions::irradiance_mapping::GPU_compute_irradiance(src_texture , width , height , 4 , &dest_texture , _width , _height , 1000);

    stbi_write_hdr("response_gpu.hdr" , _width , _height , 4 , dest_texture);
    delete[] src_texture ;
}*/
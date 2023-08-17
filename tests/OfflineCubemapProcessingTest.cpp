#include "Test.h"
#include "../includes/OfflineCubemapProcessing.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../vendor/stb/stb_image_write.h"
#include "../vendor/stb/stb_image.h"

#define f_rand rand() / ((float) RAND_MAX) 

constexpr unsigned int NUM_CHANNELS = 3 ; 
template<typename T>
class TextureBuilder{
public:
    struct TexStruct{
        T* data ; 
        unsigned width ; 
        unsigned height ; 
    };
public:
    TextureBuilder(unsigned _width , unsigned _height , bool random = true , T value_if_not_random = 1.f){
        srand(time(nullptr));
        width = _width ; 
        height = _height ; 
        if(random)
            for(unsigned i = 0 ; i < width * height * NUM_CHANNELS ; i++)
                data.push_back(f_rand); 
        else
            for(unsigned i = 0 ; i < width * height * NUM_CHANNELS ; i++)
                data.push_back(value_if_not_random); 
    
    }

    virtual ~TextureBuilder(){

    }

    const TexStruct getData(){
        TexStruct tex ; 
        tex.data = data.data(); 
        tex.width = width ; 
        tex.height = height ;
        return tex ;  
    }
public:
    std::vector<T> data;
    unsigned width ;
    unsigned height ; 

};

template<class T>
class RandomVecBuilder{
public:
    RandomVecBuilder(){
        srand(time(nullptr)); 
    }
    virtual ~RandomVecBuilder(){}
    T generate(){
        return T(f_rand); 
    }
};




/*
TEST(EnvmapComputation , WrapPixelCoords){
    TextureBuilder<float> builder(100 , 100); 
    EnvmapProcessing process(builder.getData().data , builder.getData().width , builder.getData().height); 
    std::vector<std::pair<glm::dvec2 , glm::dvec2>> test_cases ={
        std::pair(glm::dvec2(0 , 0) , glm::dvec2(0 , 0)) , 
        std::pair(glm::dvec2(100 , 100) , glm::dvec2(0 , 0)), 
        std::pair(glm::dvec2(101 , 0) , glm::dvec2(1 , 0)) , 
        std::pair(glm::dvec2(101 , -101) , glm::dvec2(1 , 99))
    };
    for(auto A : test_cases){
        glm::dvec2 case_test = A.first ; 
        glm::dvec2 expected_result = A.second ;
        glm::dvec2 wrap_result = process.wrapAroundPixelCoords(case_test.x , case_test.y); 
        EXPECT_EQ(wrap_result , expected_result); 
    }
}


TEST(EnvmapComputation , discreteSample){
    TextureBuilder<float> builder(10 , 10 , false , 1.f) ; 
    EnvmapProcessing process(builder.getData().data , builder.getData().width, builder.getData().height); 
    for(unsigned i = 0 ; i < builder.width ; i++)
        for(unsigned j = 0 ; j < builder.height ; j++){
            bool test_eq = process.discreteSample(i , j) == glm::dvec3(1.) ; 
            ASSERT_EQ(process.discreteSample(i , j) , glm::dvec3(1.f));
        }
}

TEST(EnvmapComputation , uvSample){
    std::vector<float> array = {
        -0.7f  , 0.3f , 0.9f,  0.7f , 0.2f , 0.1f ,
        -0.3f , 0.1f , 0.5f , 0.1f , 0.2f , 0.3f 
    };
    EnvmapProcessing process(array.data() , 2 , 2);
    EXPECT_EQ(process.uvSample(0.f , 0.f) , glm::dvec3(-0.7f , 0.3f , 0.9f));
    EXPECT_EQ(process.uvSample(1.f , 0.f) , glm::dvec3(-0.3f , 0.1f , 0.5f)); 
    EXPECT_EQ(process.uvSample(0.f , 1.f) , glm::dvec3(0.7f , 0.2f , 0.1f)); 
    EXPECT_EQ(process.uvSample(1.f , 1.f) , glm::dvec3(0.1f , 0.2f , 0.3f));
}

TEST(EnvmapComputation , uvSphericalCohesion){
    RandomVecBuilder<glm::dvec2> builder;
    for(unsigned i = 0 ; i < 10 ; i ++){
        glm::dvec2 uv = builder.generate(); 
        glm::dvec2 sph = gpgpu_math::uvToSpherical(uv); 
        glm::dvec2 test = gpgpu_math::sphericalToUv(sph);
        std::cout << "UV : (" << uv.x << " , " << uv.y << ")   Test:(" <<test.x << " , " << test.y << ")\n"; 
        std::cout << (test == uv ? "Equals" : "Not Equals") << "\n" ; 
        EXPECT_EQ(test , uv);
    } 
}

*/
TEST(EnvmapComputation , computeDiffuseIrradiance){
	std::string image = "test3.hdr" ;
    int width = 0 ; 
    int height = 0 ;
    int channels = 0 ;  
    float *hdr_data = stbi_loadf( image.c_str() , &width , &height , &channels , 0);
    if(stbi_failure_reason())
		std::cout << stbi_failure_reason() << "\n"; 
    EnvmapProcessing process(hdr_data , (unsigned) width , (unsigned) height);
    stbi_write_hdr("template.hdr" , width , height , 3 , hdr_data); 
    auto tex = process.computeDiffuseIrradiance(1.f); 
    stbi_write_hdr("response.hdr" , width , height , 3 , tex->f_data); 
    tex->clean(); 
}
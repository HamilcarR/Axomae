#ifndef CUBEMAPPROCESSING_CUH
#define CUBEMAPPROCESSING_CUH
#include "Includes.cuh"
#include "../includes/PerformanceLogger.h"
#include <cmath>







    /**
     * @brief Convert Pixel coordinates to UV.
     * 
     * @tparam D Type of the pixel coordinates.
     * @param coord 
     * @param dim 
     * @return const double 
     */
    template<class D>
    __host__ __device__
    inline double pixelToUv(const D coord , const unsigned dim)  {
        return static_cast<double>(coord) / static_cast<double>(dim); 
    }

    /**
     * @brief Convert UV component to pixel coordinate.
     * 
     * @tparam D Type of the UV coordinate.
     * @param coord UV coordinate component.
     * @param dim Dimension of the image in pixels .
     * @return const unsigned Pixel coordinate. 
     */
    template<class D>
    __host__ __device__
    inline unsigned uvToPixel(const D coord , unsigned dim) {
        return static_cast<unsigned> (coord * dim) ; 
    }


namespace spherical_math{

__device__
inline const float3 sphericalToCartesian(const float phi , const float theta);

__device__
inline const float2 cartesianToSpherical(const float x , const float y , const float z);

__device__ 
inline const float2 uvToSpherical(const float u , const float v); 

__device__
inline const float2 sphericalToUv(const float u , const float v); 

__device__
inline float3 gpu_pgc3d(unsigned x , unsigned y , unsigned z) ; 
    















template<class T>
__host__ __device__ 
inline const glm::dvec2 uvToSpherical(const T &u , const T &v){
    const T phi = 2 * PI * u; 
    const T theta = PI * v ;
    return glm::dvec2(phi , theta); 
}

__host__ __device__
inline const glm::dvec2 uvToSpherical(const glm::dvec2& uv){
    return uvToSpherical(uv.x , uv.y); 
}

template<class T>
__host__ __device__
inline const glm::dvec2 sphericalToUv(const T &phi , const T &theta){
    const T u = phi / (2 * PI) ; 
    const T v = theta / PI ; 
    return glm::dvec2(u , v); 
}

__host__ __device__
inline const glm::dvec2 sphericalToUv(const glm::dvec2& sph){
    return sphericalToUv(sph.x , sph.y);
}

template<class T>
__host__ __device__
inline const glm::dvec3 sphericalToCartesian(const T &phi , const T &theta){
    const T z = cos(theta);
    const T x = sin(theta) * cos(phi); 
    const T y = sin(theta) * sin(phi); 
    return glm::dvec3(x , y , z); 
}

__host__ __device__
inline const glm::dvec3 sphericalToCartesian(const glm::dvec2& sph){
    return sphericalToCartesian(sph.x , sph.y); 
}

template<class T>
__host__ __device__
inline const glm::dvec2 cartesianToSpherical(const T& x , const T& y , const T& z){
    const T theta = acos(z) ; 
    const T phi = atan2f(y , x);
    return glm::dvec2(phi , theta); 
}

__host__ __device__
inline const glm::dvec2 cartesianToSpherical(const glm::dvec3& xyz){
    return cartesianToSpherical(xyz.x , xyz.y , xyz.z); 
}

__host__ __device__
inline glm::vec3 pgc3d(unsigned x , unsigned y , unsigned z) {
    x = x * 1664525u + 1013904223u; 
    y = y * 1664525u + 1013904223u;
    z = z * 1664525u + 1013904223u;
    x += y*z; y += z*x; z += x*y;
    x ^= x >> 16u;y ^= y >> 16u;z ^= z >> 16u;
    x += y*z; y += z*x; z += x*y;
    return glm::vec3(x , y , z) * (1.f / float(0xFFFFFFFFu));
}

__host__ __device__
inline double radicalInverse(unsigned bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return double(bits) * 2.3283064365386963e-10; 
}

__host__ __device__
inline glm::dvec3 hammersley3D(unsigned  i , unsigned N){
    return glm::dvec3(double(i) / double(N) , radicalInverse(i) , radicalInverse(i ^ 0xAAAAAAAAu)); 
}

}; // End namespace gpgpu_math

/**********************************************************************************************************************************************************************************/



namespace gpgpu_functions{

    inline void gpgpu_float_to_rgbe(unsigned char rgbe[4] , float r , float g , float b){
        float mantissa ; 
        int exponent ; 
        mantissa = std::max(std::max(r , g) , b) ; 
        if(mantissa < 1e-32)
            std::memset(rgbe , 0 , 4) ;
        else{
            mantissa = std::frexp(mantissa , &exponent) * 256/mantissa;
            rgbe[0] = (unsigned char) (r * mantissa); 
            rgbe[1] = (unsigned char) (g * mantissa); 
            rgbe[2] = (unsigned char) (b * mantissa); 
            rgbe[3] = (unsigned char) (mantissa + 128); 
        }
    }

    inline void gpgpu_rgbe_to_float(float rgbe , float rgb[3]){
        float f ;
        if(rgbe){
            uint8_t exp = *((uint32_t*) &rgbe) << 24;
            uint8_t b = *((uint32_t*) &rgbe) << 16 ; 
            uint8_t g = *((uint32_t*) &rgbe) << 8 ; 
            uint8_t r = *((uint32_t*) &rgbe) & 0xFF; 
            f = std::ldexp(1 , exp - (int) (128 + 8)); 
            rgb[0] = r * f; 
            rgb[1] = g * f; 
            rgb[2] = b * f; 
        }
    }

    namespace irradiance_mapping{

        __global__
        void gpgpu_device_compute_diffuse_irradiance(float * D_result_buffer , cudaTextureObject_t texture , unsigned width , unsigned height ,  unsigned _width , unsigned _height , unsigned samples);
        

        __host__
        void gpgpu_kernel_call(void (*device_function)(float* , cudaTextureObject_t , unsigned , unsigned , unsigned , unsigned , unsigned),
                        float *D_result_buffer , 
                        cudaTextureObject_t , 
                        unsigned width , 
                        unsigned height , 
                        unsigned _width , 
                        unsigned _height , 
                        unsigned samples ); 
        
        __global__
        void gpgpu_device_compute_diffuse_irradiance(float * D_result_buffer , float *D_src_buffer , unsigned width , unsigned height ,  unsigned _width , unsigned _height , unsigned samples);
        

        __host__
        void gpgpu_kernel_call(void (*device_function)(float* , float* , unsigned , unsigned , unsigned , unsigned , unsigned),
                        float *D_result_buffer , 
                        float *D_src_buffer , 
                        unsigned width , 
                        unsigned height , 
                        unsigned _width , 
                        unsigned _height , 
                        unsigned samples ); 
                
        static void GPU_compute_channel_irradiance(float* src_texture , unsigned src_texture_width , unsigned src_texture_height , float** dest_texture , unsigned dest_texture_width , unsigned dest_texture_height , unsigned samples){
            float *D_src_texture; 
            size_t pitch_src = src_texture_width * sizeof(float);
            size_t pitch_dest = dest_texture_width * sizeof(float); 
            cudaErrCheck(cudaMalloc((void**)&D_src_texture , src_texture_width * src_texture_height * sizeof(float))); 
            cudaErrCheck(cudaMemcpy(D_src_texture , src_texture , pitch_src * src_texture_height , cudaMemcpyHostToDevice));   
            cudaErrCheck(cudaMallocManaged((void**) dest_texture , dest_texture_height * pitch_dest));
            gpgpu_kernel_call(gpgpu_device_compute_diffuse_irradiance , *dest_texture , D_src_texture , src_texture_width , src_texture_height , dest_texture_width , dest_texture_height , samples);
        }
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        /**
         * @brief Computes an irradiance map using an nvidia gpu , and stores it in "dest_texture". 
         * 
         * @param src_texture 
         * @param src_texture_width 
         * @param src_texture_height 
         * @param channels 
         * @param dest_texture 
         * @param dest_texture_width 
         * @param dest_texture_height 
         *
        static void GPU_compute_irradiance(float* src_texture , unsigned src_texture_width , unsigned src_texture_height , unsigned channels , float** dest_texture ,  unsigned dest_texture_width , unsigned dest_texture_height , unsigned samples){
            cudaResourceDesc resource_descriptor ; 
            std::memset(&resource_descriptor , 0 , sizeof(resource_descriptor)) ;
            cudaTextureDesc texture_descriptor ; 
            std::memset(&texture_descriptor , 0 , sizeof(texture_descriptor)); 
            cudaTextureObject_t texture_object = 0 ;
            // Initialize Cuda array and copy to device 
            cudaArray_t cuda_array ; 
            cudaChannelFormatDesc format_desc = cudaCreateChannelDesc(32 , 32 , 32 , 32 , cudaChannelFormatKindFloat); 
            check_error(__FILE__ , __LINE__);
            size_t pitch  = src_texture_width * channels * sizeof(float) ;
            cudaMallocArray(&cuda_array , &format_desc , src_texture_width , src_texture_height );
            check_error(__FILE__ , __LINE__);
            cudaMemcpy2DToArray(cuda_array , 0 , 0 , src_texture , pitch , src_texture_width * channels * sizeof(float), src_texture_height , cudaMemcpyHostToDevice);  
            check_error(__FILE__ , __LINE__);
            // Initialize resource descriptors 
            resource_descriptor.resType = cudaResourceTypeArray ; 
            resource_descriptor.res.array.array = cuda_array ;
            // Initialize texture descriptors 
            texture_descriptor.addressMode[0] = cudaAddressModeWrap; 
            texture_descriptor.addressMode[1] = cudaAddressModeWrap;
            texture_descriptor.filterMode = cudaFilterModeLinear ; 
            texture_descriptor.readMode = cudaReadModeElementType ; 
            texture_descriptor.normalizedCoords = 1 ; 
            // Initialize texture object 
            cudaCreateTextureObject(&texture_object , &resource_descriptor , &texture_descriptor , nullptr);
            check_error(__FILE__ , __LINE__);
            cudaMallocManaged((void**) dest_texture , dest_texture_height * pitch);
            check_error(__FILE__ , __LINE__);
            gpgpu_kernel_call(gpgpu_device_compute_diffuse_irradiance , *dest_texture , texture_object , channels , src_texture_width , src_texture_height , dest_texture_width , dest_texture_height , samples);
            check_error(__FILE__ , __LINE__);
            cudaDeviceSynchronize(); 
            check_error(__FILE__ , __LINE__);
            cudaDestroyTextureObject(texture_object);   
            cudaFreeArray(cuda_array); 
            check_error(__FILE__ , __LINE__);
        }
        */




       
    







    };//End namespace irradiance_mapping
};//End namespace gpgpu_functions





































#endif
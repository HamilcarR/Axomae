#ifndef CUBEMAPPROCESSING_CUH
#define CUBEMAPPROCESSING_CUH
#include "Includes.cuh"
#include <cmath>
namespace gpgpu_math{


template<class T>
__host__ __device__
inline const glm::dvec2 uvToSpherical(const T u , const T v){
    const T phi = 2 * PI * u; 
    const T theta = PI * v ;
    return glm::dvec2(phi , theta); 
}

__host__ __device__
inline const glm::dvec2 uvToSpherical(const glm::dvec2 uv){
    return uvToSpherical(uv.x , uv.y); 
}

template<class T>
__host__ __device__
inline const glm::dvec2 sphericalToUv(const T phi , const T theta){
    const T u = phi / (2 * PI) ; 
    const T v = theta / PI ; 
    return glm::dvec2(u , v); 
}

__host__ __device__
inline const glm::dvec2 sphericalToUv(const glm::dvec2 sph){
    return sphericalToUv(sph.x , sph.y);
}

template<class T>
__host__ __device__
inline const glm::dvec3 sphericalToCartesian(const T phi , const T theta){
    const T z = cos(theta);
    const T x = sin(theta) * cos(phi); 
    const T y = sin(theta) * sin(phi); 
    return glm::normalize(glm::dvec3(x , y , z)); 
}

__host__ __device__
inline const glm::dvec3 sphericalToCartesian(const glm::dvec2 sph){
    return sphericalToCartesian(sph.x , sph.y); 
}

template<class T>
__host__ __device__
inline const glm::dvec2 cartesianToSpherical(const T x , const T y , const T z){
    const T theta = acos(z) ; 
    const T phi = atan2f(y , x);
    return glm::dvec2(phi , theta); 
}

__host__ __device__
inline const glm::dvec2 cartesianToSpherical(const glm::dvec3 xyz){
    return cartesianToSpherical(xyz.x , xyz.y , xyz.z); 
}

__host__ __device__
inline glm::vec3 pgc3d(unsigned x , unsigned y , unsigned z) {
    x = x * 1664525u + 1013904223u; 
    y = y * 1664525u + 1013904223u;
    z = z * 1664525u + 1013904223u;

    x += y*z; 
    y += z*x; 
    z += x*y;
    
    x ^= x >> 16u;
    y ^= y >> 16u;
    z ^= z >> 16u;
    
    x += y*z; 
    y += z*x; 
    z += x*y;

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











};
#endif
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














};
#endif
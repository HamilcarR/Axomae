#ifndef CUBEMAPPROCESSING_CUH
#define CUBEMAPPROCESSING_CUH
#include "Includes.cuh"

namespace gpgpu_math{



__host__ __device__
inline const glm::dvec2 uvToSpherical(const double u , const double v){
    assert(u <= 1.f); 
    assert(v <= 1.f); 
    const double phi = 2 * PI * u; 
    const double theta = PI * v ;
    return glm::dvec2(phi , theta); 
}

__host__ __device__
inline const glm::dvec2 uvToSpherical(glm::dvec2 uv){
    return uvToSpherical(uv.x , uv.y); 
}

__host__ __device__
inline const glm::dvec2 sphericalToUv(const double phi , const double theta){
    const double u = phi / (2 * PI) ; 
    const double v = theta / PI ; 
    return glm::dvec2(u , v); 
}

__host__ __device__
inline const glm::dvec2 sphericalToUv(glm::dvec2 sph){
    return sphericalToUv(sph.x , sph.y);
}

__host__ __device__
inline const glm::dvec3 sphericalToCartesian(const double phi , const double theta){
    const double z = cos(theta);
    const double x = sin(theta) * cos(phi); 
    const double y = sin(theta) * sin(phi); 
    return glm::dvec3(x , y , z); 
}

__host__ __device__
inline const glm::dvec3 sphericalToCartesian(glm::dvec2 sph){
    return sphericalToCartesian(sph.x , sph.y); 
}

__host__ __device__
inline const glm::dvec2 cartesianToSpherical(const double x , const double y , const double z){
    const double theta = std::acos(z/sqrtf32(x*x + y*y + z*z)); 
    const double phi = atan2f(y , x);
    return glm::dvec2(phi , theta); 
}

__host__ __device__
inline const glm::dvec2 cartesianToSpherical(glm::dvec3 xyz){
    return cartesianToSpherical(xyz.x , xyz.y , xyz.z); 
}














};
#endif
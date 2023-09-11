#include "CubemapProcessing.cuh"


constexpr unsigned MAX_ITER_THREAD = 100 ; 

namespace spherical_math{
__device__
inline const float3 sphericalToCartesian(const float phi , const float theta){
    float z = cos(theta); 
    float x = sin(theta) * cos(phi); 
    float y = sin(theta) * sin(phi);
    float3 xyz; 
    xyz.x = x ; 
    xyz.y = y ; 
    xyz.z = z ;
    return xyz ; 
}

__device__
inline const float2 cartesianToSpherical(const float x , const float y , const float z){
    const float theta = acos(z) ; 
    const float phi = atan2f(y , x);
    float2 sph; 
    sph.x = phi ; 
    sph.y = theta ;
    return sph ;  
}

__device__ 
inline const float2 uvToSpherical(const float u , const float v){
    float phi = 2 * PI * u; 
    float theta  = PI * v ;
    float2 spherical ; 
    spherical.x = phi ; 
    spherical.y = theta; 
    return spherical; 
}

__device__
inline const float2 sphericalToUv(const float phi , const float theta){
    const float u = phi / (2 * PI) ; 
    const float v = theta / PI ; 
    float2 uv; 
    uv.x = u ; 
    uv.y = v ; 
    return uv; 
}

__device__
inline float3 gpu_pgc3d(unsigned x , unsigned y , unsigned z) {
    x = x * 1664525u + 1013904223u; 
    y = y * 1664525u + 1013904223u;
    z = z * 1664525u + 1013904223u;
    x += y*z; y += z*x; z += x*y;
    x ^= x >> 16u;y ^= y >> 16u;z ^= z >> 16u;
    x += y*z; y += z*x; z += x*y;
    
    float3 ret; 
    float cste = 1.f / float(0xFFFFFFFFu) ; 
    ret.x = x * cste ; 
    ret.y = y * cste ; 
    ret.z = z * cste ; 
    
    return ret;
}
};//End namespace spherical_math

__device__
inline float dot(const float2& vec1 , const float2& vec2){
    return vec1.x * vec2.x + vec1.y * vec2.y ; 
}

__device__
inline float dot(const float3& vec1 , const float3& vec2){
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z ; 
}

__device__
inline float3 cross(const float3& A , const float3& B){
    float3 C ; 
    C.x = A.y * B.z - A.z * B.y ; 
    C.y = A.z * B.x - A.x * B.z ; 
    C.z = A.x * B.y - A.y * B.x ; 
    return C; 
}

__device__ 
inline float3 normalize(const float3& A){
    float d = sqrt(A.x * A.x + A.y * A.y + A.z * A.z); 
    float3 A1; 
    A1.x = A.x / d ; 
    A1.y = A.y / d ; 
    A1.z = A.z / d ;
    return A1 ; 
}
__device__
inline float3 operator*(float k , float3 vec){
    float3 res ; 
    res.x = vec.x * k ; 
    res.y = vec.y * k ; 
    res.z = vec.z * k ;
    return res ;  
}
__device__
inline float3 operator+(float3 vec1 , float3 vec2){
    float3 res ; 
    res.x = vec1.x + vec2.x ; 
    res.y = vec1.y + vec2.y ; 
    res.z = vec1.z + vec2.z ;
    return res; 
}

template<class T>
__device__
inline void gpgpu_device_write_buffer(T* D_result_buffer , const T val , const int x , const int y , const unsigned _width ){
    atomicAdd(&D_result_buffer[(y * _width + x)] , val) ;  
   // D_result_buffer[y * _width + x] = val; 
}

__global__
void gpgpu_functions::irradiance_mapping::gpgpu_device_compute_diffuse_irradiance(float* D_result_buffer , float* D_src_buffer , unsigned width , unsigned height ,  unsigned _width , unsigned _height , unsigned total_samples){
    extern __shared__ volatile float shared_array[];

    int i = blockDim.x * blockIdx.x + threadIdx.x ; 
    int j = blockDim.y * blockIdx.y + threadIdx.y ;
    //int k = blockDim.z * blockIdx.z + threadIdx.z ;
    
    int x = threadIdx.x ; 
    int y = threadIdx.y ; 
    if(i < _width && j < _height ){
        float u = (float) i / (float) _width ; 
        float v = (float) j / (float) _height ; 
        float2 sph = spherical_math::uvToSpherical(u , v); 
        float3 cart = spherical_math::sphericalToCartesian(sph.x , sph.y);
        float3 normal ; normal.x = cart.x ; normal.y = cart.y ; normal.z = cart.z ; 
        float3 someVec ; someVec.x = 1.f ; someVec.y = 0.f ; someVec.z = 0.f ; 
        float dd = dot(someVec, normal) ; 
        float3 tangent = {.x = 0.f , .y = 1.f , .z = 0.f } ; 
        if(1.0 - abs(dd) > 1e-6) 
            tangent = normalize(cross(someVec, normal));
        float3 bitangent = cross(normal, tangent);  
        float3 random = spherical_math::gpu_pgc3d( i , j , (j * (gridDim.x * blockDim.x) + i) * blockIdx.z);  
        float phi = 2 * PI * random.x ; 
        float theta = asin(sqrt(random.y));
        //float3 uv_cart = spherical_math::sphericalToCartesian(phi , theta) ;
        float uv_cart_x = sin(theta) * cos(phi); 
        float uv_cart_z = cos(theta);
        float uv_cart_y = sin(theta) * sin(phi);  
        float t_x = tangent.x * uv_cart_x , t_y  = tangent.y * uv_cart_y , t_z = tangent.z * uv_cart_z ; 
        float b_x = bitangent.x * uv_cart_x , b_y  = bitangent.y * uv_cart_y , b_z = bitangent.z * uv_cart_z ; 
        float n_x = normal.x * uv_cart_x , n_y  = normal.y * uv_cart_y , n_z = normal.z * uv_cart_z ;
        float x_c = t_x + b_x + n_x ; 
        float y_c = t_y + b_y + n_y ; 
        float z_c = t_z + b_z + n_z ; 
        float2 spherical = spherical_math::cartesianToSpherical(x_c , y_c , z_c);
        float2 uvt = spherical_math::sphericalToUv(spherical.x , spherical.y);
        //float irradiance = tex2D<float>(texture , uvt.x , uvt.y);
        int src_x = abs(uvt.x) * width; 
        int src_y = abs(uvt.y) * height ;
        //float irradiance = D_src_buffer[src_y * width + src_x] ; 
        float irradiance = shared_array[y * blockDim.x + x] ; 
        gpgpu_device_write_buffer(D_result_buffer , 1.f ,  i , j , _width); 
      /*******************************************************
     *!note : Threads aren't coalesced :  
     *!Divide image by channels. 3 textures = 3 channels. 
     *!Compute each differently using async kernel launch. 
     *!Merge together . 
     * 
     * */ 
    }
    
}


__global__
void test(float* D_result_buffer , float* D_src_buffer , unsigned width , unsigned height ,  unsigned _width , unsigned _height , unsigned total_samples){
    extern __shared__ float shared[]; 
    int x = threadIdx.x ;  
    int y = threadIdx.y ;
    int thd_x = blockIdx.x * blockDim.x + x ; 
    int thd_y = blockIdx.y * blockDim.y + y ; 
    int shared_index = y * blockDim.x + x ; 
    int pixel_index = blockIdx.y * _width + blockIdx.x ;

    if(blockIdx.x < _width && blockIdx.y < _height){
        shared[shared_index] = D_src_buffer[pixel_index] ;
        __syncthreads();
        
        for(unsigned s = blockDim.x * blockDim.y / 2 ; s > 0 ; s >>= 1){
            if(shared_index < s)
                shared[shared_index] += shared[shared_index + s] ; 
            __syncthreads(); 
        }
        if(shared_index == 0)
            D_result_buffer[pixel_index] = shared[0]; 
    
    } 
}







__host__
void gpgpu_functions::irradiance_mapping::gpgpu_kernel_call(void (*device_function)(float* , float* , unsigned , unsigned , unsigned , unsigned , unsigned) ,
                                                    float* D_result_buffer,
                                                    float* D_src_texture, 
                                                    const unsigned width , 
                                                    const unsigned height , 
                                                    const unsigned _width , 
                                                    const unsigned _height ,
                                                    const unsigned samples )
{
    
    
    cudaEvent_t start , stop ; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    dim3 threads_per_blocks(32 , 32);
    dim3 blocks; 
    blocks.x = _width  ; 
    blocks.y = _height  ; 
    cudaEventRecord(start);  
    size_t shared_mem = threads_per_blocks.x * threads_per_blocks.y  * sizeof(float);
    test<<<blocks , threads_per_blocks  , shared_mem >>> (D_result_buffer , D_src_texture , width , height , _width , _height , samples); 
    cudaErrCheck(cudaDeviceSynchronize()); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 
    float time; 
    cudaEventElapsedTime(&time , start , stop); 
    std::cout << "Diffuse irradiance kernel elapsed time : " << time << " ms \n";  
    cudaEventDestroy(start); 
    cudaEventDestroy(stop); 
} 





















/*

__host__
void gpgpu_functions::irradiance_mapping::gpgpu_kernel_call(void (*device_function)(float* , cudaTextureObject_t , unsigned , unsigned , unsigned , unsigned , unsigned) ,
                                                    float* D_result_buffer,
                                                    cudaTextureObject_t texture, 
                                                    const unsigned width , 
                                                    const unsigned height , 
                                                    const unsigned _width , 
                                                    const unsigned _height ,
                                                    const unsigned samples )
{
    
    
    cudaEvent_t start , stop ; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    dim3 threads_per_blocks(16 , 16 , 4);
    dim3 blocks; 
    blocks.x = _width / threads_per_blocks.x ; 
    blocks.y = _height / threads_per_blocks.y ; 
    blocks.z = samples / threads_per_blocks.z ; 
    blocks.x++; 
    blocks.y++; 
    blocks.z++;
    cudaEventRecord(start);  
    size_t shared_mem = threads_per_blocks.x * threads_per_blocks.y  * sizeof(float);
    device_function <<<blocks , threads_per_blocks , shared_mem>>> (D_result_buffer , texture , width , height , _width , _height , samples); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 
    float time; 
    cudaEventElapsedTime(&time , start , stop); 
    std::cout << "Diffuse irradiance kernel elapsed time : " << time << "\n";  
    cudaEventDestroy(start); 
    cudaEventDestroy(stop); 
} 

*/













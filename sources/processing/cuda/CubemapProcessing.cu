#include "CubemapProcessing.cuh"
#include "device/cuda/CudaDevice.h"
#include "device/cuda/CudaParams.h"
#include "device/device_utils.h"
#include "device/cuda/cuda_utils.h"
#include "math/math_spherical.h"
#include "device/kernel_launch_interface.h"
#ifdef AXOMAE_STATS_TIMER
#  include "PerformanceLogger.h"
#endif

namespace spherical_math {

  AX_DEVICE_ONLY inline float3 sphericalToCartesian(const float phi, const float theta);

  AX_DEVICE_ONLY inline float2 cartesianToSpherical(const float x, const float y, const float z);

  AX_DEVICE_ONLY inline float2 uvToSpherical(const float u, const float v);

  AX_DEVICE_ONLY inline float2 sphericalToUv(const float u, const float v);

  AX_DEVICE_ONLY inline float3 gpu_pgc3d(unsigned x, unsigned y, unsigned z);

  AX_DEVICE_ONLY float3 sphericalToCartesian(const float phi, const float theta) {
    float z = cos(theta);
    float x = sin(theta) * cos(phi);
    float y = sin(theta) * sin(phi);
    float3 xyz;
    xyz.x = x;
    xyz.y = y;
    xyz.z = z;
    return xyz;
  }

  AX_DEVICE_ONLY inline float2 cartesianToSpherical(const float x, const float y, const float z) {
    const float theta = acos(z);
    const float phi = atan2f(y, x);
    float2 sph;
    sph.x = phi;
    sph.y = theta;
    return sph;
  }

  AX_DEVICE_ONLY inline float2 uvToSpherical(const float u, const float v) {
    float phi = 2 * PI * u;
    float theta = PI * v;
    float2 spherical;
    spherical.x = phi;
    spherical.y = theta;
    return spherical;
  }

  AX_DEVICE_ONLY inline float2 sphericalToUv(const float phi, const float theta) {
    const float u = phi / (2 * PI);
    const float v = theta / PI;
    float2 uv;
    uv.x = u;
    uv.y = v;
    return uv;
  }

  AX_DEVICE_ONLY inline float3 gpu_pgc3d(unsigned x, unsigned y, unsigned z) {
    x = x * 1664525u + 1013904223u;
    y = y * 1664525u + 1013904223u;
    z = z * 1664525u + 1013904223u;
    x += y * z;
    y += z * x;
    z += x * y;
    x ^= x >> 16u;
    y ^= y >> 16u;
    z ^= z >> 16u;
    x += y * z;
    y += z * x;
    z += x * y;

    float3 ret;
    float cste = 1.f / float(0xFFFFFFFFu);
    ret.x = x * cste;
    ret.y = y * cste;
    ret.z = z * cste;

    return ret;
  }
};  // End namespace spherical_math

AX_DEVICE_ONLY inline float dot(const float2 &vec1, const float2 &vec2) { return vec1.x * vec2.x + vec1.y * vec2.y; }

AX_DEVICE_ONLY inline float dot(const float3 &vec1, const float3 &vec2) { return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z; }

AX_DEVICE_ONLY inline float3 cross(const float3 &A, const float3 &B) {
  float3 C;
  C.x = A.y * B.z - A.z * B.y;
  C.y = A.z * B.x - A.x * B.z;
  C.z = A.x * B.y - A.y * B.x;
  return C;
}

AX_DEVICE_ONLY inline float3 normalize(const float3 &A) {
  float d = sqrt(A.x * A.x + A.y * A.y + A.z * A.z);
  float3 A1;
  A1.x = A.x / d;
  A1.y = A.y / d;
  A1.z = A.z / d;
  return A1;
}
AX_DEVICE_ONLY inline float3 operator*(float k, float3 vec) {
  float3 res;
  res.x = vec.x * k;
  res.y = vec.y * k;
  res.z = vec.z * k;
  return res;
}
AX_DEVICE_ONLY inline float3 operator+(float3 vec1, float3 vec2) {
  float3 res;
  res.x = vec1.x + vec2.x;
  res.y = vec1.y + vec2.y;
  res.z = vec1.z + vec2.z;
  return res;
}

template<class T>
AX_DEVICE_ONLY void gpgpu_device_write_buffer(T *D_result_buffer, const float3 val, const int x, const int y, const unsigned _width) {
  D_result_buffer[(y * _width + x) * 4] = val.x;
  D_result_buffer[(y * _width + x) * 4 + 1] = val.y;
  D_result_buffer[(y * _width + x) * 4 + 2] = val.z;
  D_result_buffer[(y * _width + x) * 4 + 3] = 1.f;
}

AX_KERNEL static void gpgpu_device_compute_diffuse_irradiance(
    float *D_result_buffer, cudaTextureObject_t texture, unsigned width, unsigned height, unsigned _width, unsigned _height, unsigned total_samples) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < _width && j < _height) {
    float u = (float)i / (float)_width;
    float v = (float)j / (float)_height;
    float2 sph = spherical_math::uvToSpherical(u, v);
    float3 cart = spherical_math::sphericalToCartesian(sph.x, sph.y);
    float3 normal;
    normal.x = cart.x;
    normal.y = cart.y;
    normal.z = cart.z;
    float3 someVec;
    someVec.x = 1.f;
    someVec.y = 0.f;
    someVec.z = 0.f;
    float dd = dot(someVec, normal);
    float3 tangent = {.x = 0.f, .y = 1.f, .z = 0.f};
    if (1.0 - abs(dd) > 1e-6)
      tangent = normalize(cross(someVec, normal));
    float3 bitangent = cross(normal, tangent);
    float3 irradiance;
    irradiance.x = 0;
    irradiance.y = 0;
    irradiance.z = 0;
    for (unsigned samples = 0; samples < total_samples; samples++) {
      float3 random = spherical_math::gpu_pgc3d(i, j, samples);
      float phi = 2 * PI * random.x;
      float theta = asin(sqrt(random.y));
      float3 uv_cart = spherical_math::sphericalToCartesian(phi, theta);
      uv_cart = uv_cart.x * tangent + uv_cart.y * bitangent + uv_cart.z * normal;
      float2 spherical = spherical_math::cartesianToSpherical(uv_cart.x, uv_cart.y, uv_cart.z);
      float2 uvt = spherical_math::sphericalToUv(spherical.x, spherical.y);
      float4 sampled_texel = tex2D<float4>(texture, uvt.x, uvt.y);
      irradiance.x += sampled_texel.x;
      irradiance.y += sampled_texel.y;
      irradiance.z += sampled_texel.z;
    }
    irradiance.x /= total_samples;
    irradiance.y /= total_samples;
    irradiance.z /= total_samples;
    gpgpu_device_write_buffer(D_result_buffer, irradiance, i, j, _width);
  }
}

static void gpgpu_kernel_call(
    void (*device_function)(float *, cudaTextureObject_t, unsigned, unsigned, unsigned, unsigned, unsigned),
    float *D_result_buffer,
    cudaTextureObject_t texture,
    const unsigned width,
    const unsigned height,
    const unsigned _width,
    const unsigned _height,
    const unsigned samples) {

#ifdef USE_STATS_TIMER
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  dim3 threads_per_blocks(32, 32);
  dim3 blocks;
  blocks.x = _width / threads_per_blocks.x;
  blocks.y = _height / threads_per_blocks.y;
  blocks.x++;
  blocks.y++;
  blocks.z = 1;
  cudaEventRecord(start);
  size_t shared_mem = threads_per_blocks.x * threads_per_blocks.y * sizeof(float);
  device_function<<<blocks, threads_per_blocks, shared_mem>>>(D_result_buffer, texture, width, height, _width, _height, samples);
  check_error(__FILE__, __LINE__);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time, start, stop);
  LOG("Diffuse irradiance kernel elapsed time : " + std::to_string(time), LogLevel::INFO);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
#else
  dim3 threads_per_blocks(32, 32);
  dim3 blocks;
  blocks.x = _width / threads_per_blocks.x;
  blocks.y = _height / threads_per_blocks.y;
  blocks.x++;
  blocks.y++;
  blocks.z = 1;
  size_t shared_mem = threads_per_blocks.x * threads_per_blocks.y * sizeof(float);
  kernel_argpack_t argpack{blocks, threads_per_blocks, shared_mem};
  exec_kernel(argpack, device_function, D_result_buffer, texture, width, height, _width, _height, samples);
#endif
}

static void init_device_default(ax_cuda::CudaDevice &device, ax_cuda::CudaParams &cuda_device_params) {
  cuda_device_params.setDeviceID(0);
  cuda_device_params.setDeviceFlags(cudaInitDeviceFlagsAreValid);
  cuda_device_params.setFlags(cudaDeviceScheduleAuto);
  AXCUDA_ERROR_CHECK(device.GPUInitDevice(cuda_device_params));
}

static void init_descriptors(cudaArray_t cuda_array, ax_cuda::CudaParams &cuda_device_params) {
  cudaResourceDesc resource_descriptor{};
  cudaTextureDesc texture_descriptor{};
  resource_descriptor.resType = cudaResourceTypeArray;
  resource_descriptor.res.array.array = cuda_array;
  // Initialize texture descriptors
  texture_descriptor.addressMode[0] = cudaAddressModeWrap;
  texture_descriptor.addressMode[1] = cudaAddressModeWrap;
  texture_descriptor.filterMode = cudaFilterModeLinear;
  texture_descriptor.readMode = cudaReadModeElementType;
  texture_descriptor.normalizedCoords = 1;
  cuda_device_params.setResourceDesc(resource_descriptor);
  cuda_device_params.setTextureDesc(texture_descriptor);
}

void gpgpu_functions::irradiance_mapping::GPU_compute_irradiance(float *src_texture,
                                                                 unsigned src_texture_width,
                                                                 unsigned src_texture_height,
                                                                 unsigned channels,
                                                                 float **dest_texture,
                                                                 unsigned dest_texture_width,
                                                                 unsigned dest_texture_height,
                                                                 unsigned samples) {

  /* Init device */
  std::unique_ptr<ax_cuda::CudaDevice> device = std::make_unique<ax_cuda::CudaDevice>(0);
  if (!device)
    return;
  ax_cuda::CudaParams cuda_device_params;
  init_device_default(*device, cuda_device_params);
  // Initialize Cuda array and copy to device
  cudaArray_t cuda_array = nullptr;
  cuda_device_params.setChanDescriptors(32, 32, 32, 32, cudaChannelFormatKindFloat);
  AXCUDA_ERROR_CHECK(device->GPUMallocArray(&cuda_array, cuda_device_params, src_texture_width, src_texture_height));
  size_t pitch = src_texture_width * channels * sizeof(float);
  cuda_device_params.setMemcpyKind(cudaMemcpyHostToDevice);
  AXCUDA_ERROR_CHECK(device->GPUMemcpy2DToArray(
      cuda_array, 0, 0, src_texture, pitch, src_texture_width * channels * sizeof(float), src_texture_height, cuda_device_params));
  init_descriptors(cuda_array, cuda_device_params);
  cudaTextureObject_t texture_object = 0;
  AXCUDA_ERROR_CHECK(device->GPUCreateTextureObject(&texture_object, cuda_device_params));
  cuda_device_params.setFlags(1);
  AXCUDA_ERROR_CHECK(
      device->GPUMallocManaged((void **)dest_texture, dest_texture_height * dest_texture_width * channels * sizeof(float), cuda_device_params));
  gpgpu_kernel_call(gpgpu_device_compute_diffuse_irradiance,
                    *dest_texture,
                    texture_object,
                    src_texture_width,
                    src_texture_height,
                    dest_texture_width,
                    dest_texture_height,
                    samples);
  AXCUDA_ERROR_CHECK(device->GPUDeviceSynchronize());
  AXCUDA_ERROR_CHECK(device->GPUDestroyTextureObject(texture_object));
  AXCUDA_ERROR_CHECK(device->GPUFreeArray(cuda_array));
}

#ifndef CUBEMAPPROCESSING_CUH
#define CUBEMAPPROCESSING_CUH
#include "Includes.cuh"
#include "Math.h"
#include "PerformanceLogger.h"
#include <cmath>

/**********************************************************************************************************************************************************************************/

namespace gpgpu_functions {

  inline void gpgpu_float_to_rgbe(unsigned char rgbe[4], float r, float g, float b) {
    float mantissa;
    int exponent;
    mantissa = std::max(std::max(r, g), b);
    if (mantissa < 1e-32)
      std::memset(rgbe, 0, 4);
    else {
      mantissa = std::frexp(mantissa, &exponent) * 256 / mantissa;
      rgbe[0] = (unsigned char)(r * mantissa);
      rgbe[1] = (unsigned char)(g * mantissa);
      rgbe[2] = (unsigned char)(b * mantissa);
      rgbe[3] = (unsigned char)(mantissa + 128);
    }
  }

  inline void gpgpu_rgbe_to_float(float rgbe, float rgb[3]) {
    float f;
    if (rgbe) {
      uint8_t exp = *((uint32_t *)&rgbe) << 24;
      uint8_t b = *((uint32_t *)&rgbe) << 16;
      uint8_t g = *((uint32_t *)&rgbe) << 8;
      uint8_t r = *((uint32_t *)&rgbe) & 0xFF;
      f = std::ldexp(1, exp - (int)(128 + 8));
      rgb[0] = r * f;
      rgb[1] = g * f;
      rgb[2] = b * f;
    }
  }

  namespace irradiance_mapping {

    __global__ void gpgpu_device_compute_diffuse_irradiance(
        float *D_result_buffer, cudaTextureObject_t texture, unsigned width, unsigned height, unsigned _width, unsigned _height, unsigned samples);

    __host__ void gpgpu_kernel_call(void (*device_function)(float *, cudaTextureObject_t, unsigned, unsigned, unsigned, unsigned, unsigned),
                                    float *D_result_buffer,
                                    cudaTextureObject_t,
                                    unsigned width,
                                    unsigned height,
                                    unsigned _width,
                                    unsigned _height,
                                    unsigned samples);

    __global__ void gpgpu_device_compute_diffuse_irradiance(
        float *D_result_buffer, float *D_src_buffer, unsigned width, unsigned height, unsigned _width, unsigned _height, unsigned samples);

    __host__ void gpgpu_kernel_call(void (*device_function)(float *, float *, unsigned, unsigned, unsigned, unsigned, unsigned),
                                    float *D_result_buffer,
                                    float *D_src_buffer,
                                    unsigned width,
                                    unsigned height,
                                    unsigned _width,
                                    unsigned _height,
                                    unsigned samples);

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
     */
    static void GPU_compute_irradiance(float *src_texture,
                                       unsigned src_texture_width,
                                       unsigned src_texture_height,
                                       unsigned channels,
                                       float **dest_texture,
                                       unsigned dest_texture_width,
                                       unsigned dest_texture_height,
                                       unsigned samples) {
      cudaResourceDesc resource_descriptor;
      std::memset(&resource_descriptor, 0, sizeof(resource_descriptor));
      cudaTextureDesc texture_descriptor;
      std::memset(&texture_descriptor, 0, sizeof(texture_descriptor));
      cudaTextureObject_t texture_object = 0;
      // Initialize Cuda array and copy to device
      cudaArray_t cuda_array;
      cudaChannelFormatDesc format_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
      size_t pitch = src_texture_width * channels * sizeof(float);
      cudaErrCheck(cudaMallocArray(&cuda_array, &format_desc, src_texture_width, src_texture_height));
      cudaErrCheck(cudaMemcpy2DToArray(
          cuda_array, 0, 0, src_texture, pitch, src_texture_width * channels * sizeof(float), src_texture_height, cudaMemcpyHostToDevice));
      // Initialize resource descriptors
      resource_descriptor.resType = cudaResourceTypeArray;
      resource_descriptor.res.array.array = cuda_array;
      // Initialize texture descriptors
      texture_descriptor.addressMode[0] = cudaAddressModeWrap;
      texture_descriptor.addressMode[1] = cudaAddressModeWrap;
      texture_descriptor.filterMode = cudaFilterModeLinear;
      texture_descriptor.readMode = cudaReadModeElementType;
      texture_descriptor.normalizedCoords = 1;
      // Initialize texture object
      cudaErrCheck(cudaCreateTextureObject(&texture_object, &resource_descriptor, &texture_descriptor, nullptr));
      cudaErrCheck(cudaMallocManaged((void **)dest_texture, dest_texture_height * dest_texture_width * channels * sizeof(float)));
      gpgpu_kernel_call(gpgpu_device_compute_diffuse_irradiance,
                        *dest_texture,
                        texture_object,
                        src_texture_width,
                        src_texture_height,
                        dest_texture_width,
                        dest_texture_height,
                        samples);
      cudaErrCheck(cudaDeviceSynchronize());
      cudaErrCheck(cudaDestroyTextureObject(texture_object));
      cudaErrCheck(cudaFreeArray(cuda_array));
    }

    /*
   static void GPU_compute_channel_irradiance(float* src_texture , unsigned src_texture_width , unsigned src_texture_height , float** dest_texture ,
   unsigned dest_texture_width , unsigned dest_texture_height , unsigned samples){ float *D_src_texture; size_t pitch_src = src_texture_width *
   sizeof(float); size_t pitch_dest = dest_texture_width * sizeof(float); cudaErrCheck(cudaMalloc((void**)&D_src_texture , src_texture_width *
   src_texture_height * sizeof(float))); cudaErrCheck(cudaMemcpy(D_src_texture , src_texture , pitch_src * src_texture_height ,
   cudaMemcpyHostToDevice)); cudaErrCheck(cudaMallocManaged((void**) dest_texture , dest_texture_height * pitch_dest));
       gpgpu_kernel_call(gpgpu_device_compute_diffuse_irradiance , *dest_texture , D_src_texture , src_texture_width , src_texture_height ,
   dest_texture_width , dest_texture_height , samples);
   }
   */

  };  // End namespace irradiance_mapping
};    // End namespace gpgpu_functions

#endif
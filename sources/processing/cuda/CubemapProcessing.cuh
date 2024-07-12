#ifndef CUBEMAPPROCESSING_CUH
#define CUBEMAPPROCESSING_CUH
#include "Includes.cuh"

/**********************************************************************************************************************************************************************************/

namespace gpgpu_functions {

  namespace irradiance_mapping {

    KERNEL void gpgpu_device_compute_diffuse_irradiance(
        float *D_result_buffer, cudaTextureObject_t texture, unsigned width, unsigned height, unsigned _width, unsigned _height, unsigned samples);

    void gpgpu_kernel_call(void (*device_function)(float *, cudaTextureObject_t, unsigned, unsigned, unsigned, unsigned, unsigned),
                           float *D_result_buffer,
                           cudaTextureObject_t,
                           unsigned width,
                           unsigned height,
                           unsigned _width,
                           unsigned _height,
                           unsigned samples);

    KERNEL void gpgpu_device_compute_diffuse_irradiance(
        float *D_result_buffer, float *D_src_buffer, unsigned width, unsigned height, unsigned _width, unsigned _height, unsigned samples);

    void gpgpu_kernel_call(void (*device_function)(float *, float *, unsigned, unsigned, unsigned, unsigned, unsigned),
                           float *D_result_buffer,
                           float *D_src_buffer,
                           unsigned width,
                           unsigned height,
                           unsigned _width,
                           unsigned _height,
                           unsigned samples);

    void GPU_compute_irradiance(float *src_texture,
                                unsigned src_texture_width,
                                unsigned src_texture_height,
                                unsigned channels,
                                float **dest_texture,
                                unsigned dest_texture_width,
                                unsigned dest_texture_height,
                                unsigned samples);

  };  // End namespace irradiance_mapping
};    // End namespace gpgpu_functions

#endif
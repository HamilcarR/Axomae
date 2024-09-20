#ifndef CUBEMAPPROCESSING_CUH
#define CUBEMAPPROCESSING_CUH
#include "internal/device/cuda/cuda_utils.h"
#include "internal/device/device_utils.h"

/**********************************************************************************************************************************************************************************/

namespace gpgpu_functions::irradiance_mapping {

  void GPU_compute_irradiance(float *src_texture,
                              unsigned src_texture_width,
                              unsigned src_texture_height,
                              unsigned channels,
                              float **dest_texture,
                              unsigned dest_texture_width,
                              unsigned dest_texture_height,
                              unsigned samples);

}
#endif
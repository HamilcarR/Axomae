#ifndef CU_MACRO_H
#define CU_MACRO_H

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>

#define GPU_CALLABLE __host__ __device__
#define GPU_ONLY __device__
#define KERNEL __global__

#endif  // CU_MACRO_H

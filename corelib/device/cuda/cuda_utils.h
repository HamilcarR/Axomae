#ifndef CU_MACRO_H
#define CU_MACRO_H

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>

#define AX_DEVICE_CALLABLE __host__ __device__
#define AX_DEVICE_SHARED __shared__
#define AX_DEVICE_ONLY __device__
#define AX_HOST_ONLY __host__
#define AX_KERNEL __global__

#endif  // CU_MACRO_H

#ifndef CUDA_MACROS_H
#define CUDA_MACROS_H

constexpr unsigned AX_GPU_WARP_SIZE = 32;
constexpr unsigned AX_GPU_MAX_BLOCKS = 65535;

#define ax_device_thread_idx_x (blockDim.x * blockIdx.x + threadIdx.x)
#define ax_device_thread_idx_y (blockDim.y * blockIdx.y + threadIdx.y)
#define ax_device_thread_idx_z (blockDim.z * blockIdx.z + threadIdx.z)

#define ax_device_grid_dim_x (gridDim.x * blockDim.x)
#define ax_device_grid_dim_y (gridDim.y * blockDim.y)
#define ax_device_grid_dim_z (gridDim.z * blockDim.z)

#define ax_device_linearCM2D_idx (ax_device_thread_idx_x * ax_device_grid_dim_y + ax_device_thread_idx_y)
#define ax_device_linearRM2D_idx (ax_device_thread_idx_y * ax_device_grid_dim_x + ax_device_thread_idx_x)
#define ax_device_linearCM3D_idx \
  (ax_device_thread_idx_x * ax_device_grid_dim_z * ax_device_grid_dim_y + ax_device_thread_idx_y * ax_device_grid_dim_z + ax_device_thread_idx_z)
#define ax_device_linearRM3D_idx \
  (ax_device_thread_idx_z * ax_device_grid_dim_x * ax_device_grid_dim_y + ax_device_thread_idx_y * ax_device_grid_dim_x + ax_device_thread_idx_x)

#define ax_device_lane_id (ax_device_linearCM3D_idx % AX_GPU_WARP_SIZE)
#define ax_device_warp_id (ax_device_linearCM3D_idx / AX_GPU_WARP_SIZE)

/* Atomic operations */
#define AX_GPU_ATOMICADD(address, val) atomicAdd(address, val)
#define AX_GPU_ATOMICSUB(address, val) atomicSub(address, val)
#define AX_GPU_ATOMICMIN(address, val) atomicMin(address, val)
#define AX_GPU_ATOMICMAX(address, val) atomicMax(address, val)
#define AX_GPU_ATOMICAND(address, val) atomicAnd(address, val)
#define AX_GPU_ATOMICOR(address, val) atomicOr(address, val)
#define AX_GPU_ATOMICXOR(address, val) atomicXor(address, val)
#define AX_GPU_ATOMICEXCH(address, val) atomicExch(address, val)
#define AX_GPU_ATOMICINC(address, val) atomicInc(address, val)
#define AX_GPU_ATOMICDEC(address, val) atomicDec(address, val)

/* Bounds checking */
#define AX_GPU_IN_BOUNDS_3D(x, y, z, width, height, depth) ((x) < (width) && (y) < (height) && (z) < (depth))
#define AX_GPU_IN_BOUNDS_2D(x, y, width, height) ((x) < (width) && (y) < (height))
#define AX_GPU_IN_BOUNDS(x, size) ((x) < (size))
#endif  // CUDA_MACROS_H

#ifndef CUDA_MACROS_H
#define CUDA_MACROS_H

constexpr unsigned AX_GPU_WARP_SIZE = 32;
constexpr unsigned AX_GPU_MAX_BLOCKS = 65535;

#define ax_device_thread_idx_x (blockDim.x * blockIdx.x + threadIdx.x)
#define ax_device_thread_idx_y (blockDim.y * blockIdx.y + threadIdx.y)
#define ax_device_thread_idx_z (blockDim.z * blockIdx.z + threadIdx.z)

#define ax_grid_dim_x (gridDim.x * blockDim.x)
#define ax_grid_dim_y (gridDim.y * blockDim.y)
#define ax_grid_dim_z (gridDim.z * blockDim.z)

#define ax_linearCM2D_idx (ax_device_thread_idx_x * ax_grid_dim_y + ax_device_thread_idx_y)
#define ax_linearRM2D_idx (ax_device_thread_idx_y * ax_grid_dim_x + ax_device_thread_idx_x)
#define ax_linearCM3D_idx (ax_device_thread_idx_x * ax_grid_dim_z * ax_grid_dim_y + ax_device_thread_idx_y * ax_grid_dim_z + ax_device_thread_idx_z)
#define ax_linearRM3D_idx (ax_device_thread_idx_z * ax_grid_dim_x * ax_grid_dim_y + ax_device_thread_idx_y * ax_grid_dim_x + ax_device_thread_idx_x)

#define ax_gpu_lane_id (ax_linearCM3D_idx % AX_GPU_WARP_SIZE)
#define ax_gpu_warp_id (ax_linearCM3D_idx / AX_GPU_WARP_SIZE)

/* Synchronization */
#define ax_gpu_syncthread __syncthreads()
#define ax_gpu_syncwarp __syncwarp()
#define ax_gpu_memfence __threadfence()
#define ax_gpu_memfence_block __threadfence_block()
#define ax_gpu_memfence_system __threadfence_system()

/* Math operations */
#define AX_GPU_FASTCOS(value) __cosf(value)
#define AX_GPU_FASTSIN(value) __sinf(value)
#define AX_GPU_FASTEXP(value) __expf(value)
#define AX_GPU_FASTLOG(value) __logf(value)
#define AX_GPU_FASTSQRT(value) __fsqrt_rn(value)
#define AX_GPU_FASTPOW(x, y) __powf((x), (y))
#define AX_GPU_FASTTANH(value) tanhf(value)

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

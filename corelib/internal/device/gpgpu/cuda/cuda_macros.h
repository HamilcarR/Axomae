#ifndef CUDA_MACROS_H
#define CUDA_MACROS_H

#define ax_device_thread_idx_x (blockDim.x * blockIdx.x + threadIdx.x)
#define ax_device_thread_idx_y (blockDim.y * blockIdx.y + threadIdx.y)
#define ax_device_thread_idx_z (blockDim.z * blockIdx.z + threadIdx.z)

#define ax_grid_dim_x (gridDim.x * blockDim.x)
#define ax_grid_dim_y (gridDim.y * blockDim.y)
#define ax_grid_dim_z (gridDim.z * blockDim.z)

#define AX_GPU_WARP_SIZE 32;
#define ax_gpu_lane_id (threadIdx.x % AX_GPU_WARP_SIZE)
#define ax_gpu_warp_id (threadIdx.x / AX_GPU_WARP_SIZE)

#if defined(__CUDA_ARCH__) && defined(AXOMAE_USE_CUDA)
/* Synchronization */
#  define ax_gpu_syncthread __syncthreads()
#  define ax_gpu_syncwarp __syncwarp()
#  define ax_gpu_memfence __threadfence()
#  define ax_gpu_memfence_block __threadfence_block()
#  define ax_gpu_memfence_system __threadfence_system()

/* Math operations */
#  define AX_GPU_FASTCOS(value) __cosf(value)
#  define AX_GPU_FASTSIN(value) __sinf(value)
#  define AX_GPU_FASTEXP(value) __expf(value)
#  define AX_GPU_FASTLOG(value) __logf(value)
#  define AX_GPU_FASTSQRT(value) __fsqrt_rn(value)
#  define AX_GPU_FASTPOW(x, y) __powf((x), (y))
#  define AX_GPU_FASTTANH(value) tanhf(value)

/* Atomic operations */
#  define AX_GPU_ATOMICADD(address, val) atomicAdd(address, val)
#  define AX_GPU_ATOMICSUB(address, val) atomicSub(address, val)
#  define AX_GPU_ATOMICMIN(address, val) atomicMin(address, val)
#  define AX_GPU_ATOMICMAX(address, val) atomicMax(address, val)
#  define AX_GPU_ATOMICAND(address, val) atomicAnd(address, val)
#  define AX_GPU_ATOMICOR(address, val) atomicOr(address, val)
#  define AX_GPU_ATOMICXOR(address, val) atomicXor(address, val)
#  define AX_GPU_ATOMICEXCH(address, val) atomicExch(address, val)
#  define AX_GPU_ATOMICINC(address, val) atomicInc(address, val)
#  define AX_GPU_ATOMICDEC(address, val) atomicDec(address, val)

/* Bounds checking */
#  define AX_GPU_IN_BOUNDS_3D(x, y, z, width, height, depth) ((x) < (width) && (y) < (height) && (z) < (depth))
#  define AX_GPU_IN_BOUNDS_2D(x, y, width, height) ((x) < (width) && (y) < (height))
#endif
#endif  // CUDA_MACROS_H

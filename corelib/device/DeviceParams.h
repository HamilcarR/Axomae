#ifndef DEVICEPARAMS_H
#define DEVICEPARAMS_H
#include "cuda/cuda_utils.h"
struct kernel_argpack_t {
  dim3 num_blocks;
  dim3 block_size;
  std::size_t shared_mem_bytes{};
#if defined(__NVCC__)
  cudaStream_t stream{};
#elif defined(__HIP_DEVICE_COMPILE_)
  int stream{};
#endif
};

class DeviceParams {
 public:
  virtual ~DeviceParams() = default;

  [[nodiscard]] virtual int getDeviceID() const = 0;
  virtual void setDeviceID(int device_id) = 0;
  [[nodiscard]] virtual unsigned getDeviceFlags() const = 0;
  virtual void setDeviceFlags(unsigned device_flags) = 0;
  [[nodiscard]] virtual unsigned getFlags() const = 0;
  virtual void setFlags(unsigned flags) = 0;
  virtual void setMemcpyKind(unsigned copy_kind) = 0;
  [[nodiscard]] virtual int getMemcpyKind() const = 0;
};

#endif  // DEVICEPARAMS_H

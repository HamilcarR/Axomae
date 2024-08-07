#ifndef GPUBACKENDINTERFACE_H
#define GPUBACKENDINTERFACE_H
#include "DeviceError.h"
#include "DeviceParams.h"
#include <memory>

class GPUBackendInterface {
 public:
  virtual ~GPUBackendInterface() = default;
  virtual DeviceError init(const DeviceParams &params) = 0;
  virtual DeviceError set(const DeviceParams &params) = 0;
  virtual DeviceError allocateMemory(void **ptr, std::size_t size) = 0;
  virtual DeviceError deallocateMemory(void *ptr) = 0;
  virtual DeviceError allocateMemoryManaged(void **ptr, std::size_t byte_count, const DeviceParams &params) = 0;
  virtual DeviceError synchronize() = 0;
  virtual DeviceError copyMemory(const void *ptr_source, void *ptr_dest, std::size_t byte_count, const DeviceParams &params) = 0;
};

#endif  // BACKENDINTERFACE_H

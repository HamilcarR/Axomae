

#include "CudaDevice.h"

#include <GenericException.h>

namespace exception {
  class CudaDeviceException : public GenericException {
   public:
    explicit CudaDeviceException(const std::string &err) : GenericException() { saveErrorString(err); }
  };
}  // namespace exception

namespace ax_cuda {

  CudaDevice::CudaDevice(int device) : device_id(device) {}

  DeviceError CudaDevice::init(const DeviceParams &params) {
    return DeviceError(cudaInitDevice(device_id, params.getDeviceFlags(), params.getFlags()));
  }

  DeviceError CudaDevice::set(const DeviceParams &params) { return DeviceError(cudaSetDevice(params.getDeviceID())); }

  DeviceError CudaDevice::allocateMemory(void **ptr, std::size_t size) { return DeviceError(cudaMalloc(ptr, size)); }

  DeviceError CudaDevice::deallocateMemory(void *ptr) { return DeviceError(cudaFree(ptr)); }

  DeviceError CudaDevice::allocateMemoryManaged(void **ptr, std::size_t size, const DeviceParams &params) {
    return DeviceError(cudaMallocManaged(ptr, size, params.getFlags()));
  }

  DeviceError CudaDevice::synchronize() { return DeviceError(cudaDeviceSynchronize()); }

  DeviceError CudaDevice::copyMemory(const void *ptr_source, void *ptr_dest, std::size_t byte_count, const DeviceParams &params) {
    cudaMemcpyKind copy_type = static_cast<cudaMemcpyKind>(params.getMemcpyKind());
    return DeviceError(cudaMemcpy(ptr_dest, ptr_source, byte_count, copy_type));
  }

}  // namespace ax_cuda
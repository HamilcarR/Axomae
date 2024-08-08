

#include "CudaDevice.h"

#include <GenericException.h>

namespace ax_cuda {

  CudaDevice::CudaDevice(int device) : device_id(device) {}

  DeviceError CudaDevice::init(const CudaParams &params) {
    return DeviceError(cudaInitDevice(device_id, params.getDeviceFlags(), params.getFlags()));
  }

  DeviceError CudaDevice::set(const CudaParams &params) { return DeviceError(cudaSetDevice(params.getDeviceID())); }

  DeviceError CudaDevice::allocateMemory(void **ptr, std::size_t size) { return DeviceError(cudaMalloc(ptr, size)); }

  DeviceError CudaDevice::deallocateMemory(void *ptr) { return DeviceError(cudaFree(ptr)); }

  DeviceError CudaDevice::allocateMemoryManaged(void **ptr, std::size_t size, const CudaParams &params) {
    return DeviceError(cudaMallocManaged(ptr, size, params.getFlags()));
  }

  DeviceError CudaDevice::synchronize() { return DeviceError(cudaDeviceSynchronize()); }

  DeviceError CudaDevice::copyMemory(const void *ptr_source, void *ptr_dest, std::size_t byte_count, const CudaParams &params) {
    cudaMemcpyKind copy_type = static_cast<cudaMemcpyKind>(params.getMemcpyKind());
    return DeviceError(cudaMemcpy(ptr_dest, ptr_source, byte_count, copy_type));
  }

  DeviceError CudaDevice::allocateMemoryArray(cudaArray_t *array, const CudaParams &params, unsigned width, unsigned height, unsigned flags) {
    cudaChannelFormatDesc chan_desc = params.getChanDescriptors();
    return DeviceError(cudaMallocArray(array, &chan_desc, width, height, flags));
  }
  DeviceError CudaDevice::deallocateMemoryArray(cudaArray_t array) { return DeviceError(cudaFreeArray(array)); }

  DeviceError CudaDevice::copy2DToArray(
      cudaArray_t array, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, const CudaParams &params) {
    return DeviceError(cudaMemcpy2DToArray(array, wOffset, hOffset, src, spitch, width, height, static_cast<cudaMemcpyKind>(params.getMemcpyKind())));
  }
  DeviceError CudaDevice::createTextureObject(cudaTextureObject_t *tex, const CudaParams &params, bool use_resc_view) {
    if (use_resc_view)
      return DeviceError(cudaCreateTextureObject(tex, &params.getResourceDesc(), &params.getTextureDesc(), &params.getResourceViewDesc()));
    return DeviceError(cudaCreateTextureObject(tex, &params.getResourceDesc(), &params.getTextureDesc(), nullptr));
  }
  DeviceError CudaDevice::destroyTextureObject(cudaTextureObject_t texture_object) { return DeviceError(cudaDestroyTextureObject(texture_object)); }

}  // namespace ax_cuda
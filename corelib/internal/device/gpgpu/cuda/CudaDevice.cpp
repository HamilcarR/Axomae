#include "CudaDevice.h"

namespace ax_cuda {

  CudaDevice::CudaDevice(int device) : device_id(device) {}

  DeviceError CudaDevice::GPUInitDevice(const CudaParams &params) {
    return DeviceError(cudaInitDevice(device_id, params.getDeviceFlags(), params.getFlags()));
  }

  DeviceError CudaDevice::GPUSetDevice(const CudaParams &params) { return DeviceError(cudaSetDevice(params.getDeviceID())); }

  DeviceError CudaDevice::GPUMalloc(void **ptr, std::size_t size) { return DeviceError(cudaMalloc(ptr, size)); }

  DeviceError CudaDevice::GPUFree(void *ptr) { return DeviceError(cudaFree(ptr)); }

  DeviceError CudaDevice::GPUMallocManaged(void **ptr, std::size_t size, const CudaParams &params) {
    return DeviceError(cudaMallocManaged(ptr, size, params.getFlags()));
  }

  DeviceError CudaDevice::GPUDeviceSynchronize() { return DeviceError(cudaDeviceSynchronize()); }

  DeviceError CudaDevice::GPUMemcpy(const void *ptr_source, void *ptr_dest, std::size_t byte_count, const CudaParams &params) {
    cudaMemcpyKind copy_type = params.getMemcpyKind();
    return DeviceError(cudaMemcpy(ptr_dest, ptr_source, byte_count, copy_type));
  }

  DeviceError CudaDevice::GPUMallocArray(cudaArray_t *array, const CudaParams &params, unsigned width, unsigned height, unsigned flags) {
    cudaChannelFormatDesc chan_desc = params.getChanDescriptors();
    return DeviceError(cudaMallocArray(array, &chan_desc, width, height, flags));
  }
  DeviceError CudaDevice::GPUFreeArray(cudaArray_t array) { return DeviceError(cudaFreeArray(array)); }

  DeviceError CudaDevice::GPUMemcpy2DToArray(
      cudaArray_t array, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, const CudaParams &params) {
    return DeviceError(cudaMemcpy2DToArray(array, wOffset, hOffset, src, spitch, width, height, static_cast<cudaMemcpyKind>(params.getMemcpyKind())));
  }

  DeviceError CudaDevice::GPUCreateTextureObject(cudaTextureObject_t *tex, const CudaParams &params, bool use_resc_view) {
    if (use_resc_view)
      return DeviceError(cudaCreateTextureObject(tex, &params.getResourceDesc(), &params.getTextureDesc(), &params.getResourceViewDesc()));
    return DeviceError(cudaCreateTextureObject(tex, &params.getResourceDesc(), &params.getTextureDesc(), nullptr));
  }
  DeviceError CudaDevice::GPUDestroyTextureObject(cudaTextureObject_t texture_object) {
    return DeviceError(cudaDestroyTextureObject(texture_object));
  }

  DeviceError CudaDevice::GPUHostRegister(void *ptr, std::size_t size_bytes, unsigned flags) {
    return DeviceError(cudaHostRegister(ptr, size_bytes, flags));
  }
  DeviceError CudaDevice::GPUHostGetDevicePointer(void **ptr_device, void *ptr_host, unsigned flags) {
    return DeviceError(cudaHostGetDevicePointer(ptr_device, ptr_host, flags));
  }
  DeviceError CudaDevice::GPUHostUnregister(void *ptr_host) { return DeviceError(cudaHostUnregister(ptr_host)); }

}  // namespace ax_cuda
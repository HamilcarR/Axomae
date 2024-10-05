#include "CudaDevice.h"
#include "../DeviceError.h"
namespace ax_cuda {

  CudaDevice::CudaDevice(int device) : device_id(device) {}

  void CudaDevice::GPUInitDevice(const CudaParams &params) {
    DEVICE_ERROR_CHECK(cudaInitDevice(device_id, params.getDeviceFlags(), params.getFlags()));
  }

  void CudaDevice::GPUSetDevice(const CudaParams &params) { DEVICE_ERROR_CHECK(cudaSetDevice(params.getDeviceID())); }

  void CudaDevice::GPUMalloc(void **ptr, std::size_t size) { DEVICE_ERROR_CHECK(cudaMalloc(ptr, size)); }

  void CudaDevice::GPUFree(void *ptr) { DEVICE_ERROR_CHECK(cudaFree(ptr)); }

  void CudaDevice::GPUMallocManaged(void **ptr, std::size_t size, const CudaParams &params) { void(cudaMallocManaged(ptr, size, params.getFlags())); }

  void CudaDevice::GPUDeviceSynchronize() { DEVICE_ERROR_CHECK(cudaDeviceSynchronize()); }

  void CudaDevice::GPUMemcpy(const void *ptr_source, void *ptr_dest, std::size_t byte_count, const CudaParams &params) {
    cudaMemcpyKind copy_type = params.getMemcpyKind();
    void(cudaMemcpy(ptr_dest, ptr_source, byte_count, copy_type));
  }

  void CudaDevice::GPUMallocArray(cudaArray_t *array, const CudaParams &params, unsigned width, unsigned height, unsigned flags) {
    cudaChannelFormatDesc chan_desc = params.getChanDescriptors();
    void(cudaMallocArray(array, &chan_desc, width, height, flags));
  }
  void CudaDevice::GPUFreeArray(cudaArray_t array) { DEVICE_ERROR_CHECK(cudaFreeArray(array)); }

  void CudaDevice::GPUMemcpy2DToArray(
      cudaArray_t array, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, const CudaParams &params) {
    void(cudaMemcpy2DToArray(array, wOffset, hOffset, src, spitch, width, height, static_cast<cudaMemcpyKind>(params.getMemcpyKind())));
  }

  void CudaDevice::GPUCreateTextureObject(cudaTextureObject_t *tex, const CudaParams &params, bool use_resc_view) {
    if (use_resc_view)
      void(cudaCreateTextureObject(tex, &params.getResourceDesc(), &params.getTextureDesc(), &params.getResourceViewDesc()));
    void(cudaCreateTextureObject(tex, &params.getResourceDesc(), &params.getTextureDesc(), nullptr));
  }
  void CudaDevice::GPUDestroyTextureObject(cudaTextureObject_t texture_object) { void(cudaDestroyTextureObject(texture_object)); }

  void CudaDevice::GPUHostRegister(void *ptr, std::size_t size_bytes, unsigned flags) { void(cudaHostRegister(ptr, size_bytes, flags)); }
  void CudaDevice::GPUHostGetDevicePointer(void **ptr_device, void *ptr_host, unsigned flags) {
    void(cudaHostGetDevicePointer(ptr_device, ptr_host, flags));
  }
  void CudaDevice::GPUHostUnregister(void *ptr_host) { DEVICE_ERROR_CHECK(cudaHostUnregister(ptr_host)); }

}  // namespace ax_cuda
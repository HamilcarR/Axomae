#ifndef CUDADEVICE_H
#define CUDADEVICE_H
#include "../DeviceError.h"
#include "../device_utils.h"
#include "CudaParams.h"
#include "cuda_utils.h"
#include "internal/macro/project_macros.h"

namespace ax_cuda {

  class CudaDevice {

   private:
    int device_id{};

   public:
    CLASS_CM(CudaDevice)

    /* Init */
    ax_host_only explicit CudaDevice(int device);
    ax_host_only DeviceError GPUInitDevice(const CudaParams &params);
    ax_host_only DeviceError GPUSetDevice(const CudaParams &params);

    /* Memory */
    ax_device_callable DeviceError GPUMalloc(void **ptr, std::size_t size);
    ax_device_callable DeviceError GPUFree(void *ptr);
    ax_host_only DeviceError GPUMallocManaged(void **ptr, std::size_t, const CudaParams &params);
    ax_host_only DeviceError GPUMemcpy(const void *ptr_source, void *ptr_dest, std::size_t byte_count, const CudaParams &params);
    ax_host_only DeviceError GPUMallocArray(cudaArray_t *array, const CudaParams &params, unsigned width, unsigned height, unsigned flags = 0);
    ax_host_only DeviceError GPUFreeArray(cudaArray_t array);
    ax_host_only DeviceError GPUMemcpy2DToArray(
        cudaArray_t array, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, const CudaParams &params);
    ax_host_only DeviceError GPUHostRegister(void *ptr, std::size_t size_bytes, unsigned flags);
    ax_host_only DeviceError GPUHostGetDevicePointer(void **ptr_device, void *ptr_host, unsigned flags);
    ax_host_only DeviceError GPUHostUnregister(void *ptr_host);
    /* Textures */
    ax_host_only DeviceError GPUCreateTextureObject(cudaTextureObject_t *tex, const CudaParams &params, bool use_resc_view = false);
    ax_host_only DeviceError GPUDestroyTextureObject(cudaTextureObject_t texture_object);

    /* Synchronizations*/
    ax_host_only DeviceError GPUDeviceSynchronize();
  };

}  // namespace ax_cuda

#endif  // CUDADEVICE_H

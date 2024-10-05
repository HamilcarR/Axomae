#ifndef CUDADEVICE_H
#define CUDADEVICE_H
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
    AX_HOST_ONLY explicit CudaDevice(int device);
    AX_HOST_ONLY void GPUInitDevice(const CudaParams &params);
    AX_HOST_ONLY void GPUSetDevice(const CudaParams &params);

    /* Memory */
    AX_DEVICE_CALLABLE void GPUMalloc(void **ptr, std::size_t size);
    AX_DEVICE_CALLABLE void GPUFree(void *ptr);
    AX_HOST_ONLY void GPUMallocManaged(void **ptr, std::size_t, const CudaParams &params);
    AX_HOST_ONLY void GPUMemcpy(const void *ptr_source, void *ptr_dest, std::size_t byte_count, const CudaParams &params);
    AX_HOST_ONLY void GPUMallocArray(cudaArray_t *array, const CudaParams &params, unsigned width, unsigned height, unsigned flags = 0);
    AX_HOST_ONLY void GPUFreeArray(cudaArray_t array);
    AX_HOST_ONLY void GPUMemcpy2DToArray(
        cudaArray_t array, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, const CudaParams &params);
    AX_HOST_ONLY void GPUHostRegister(void *ptr, std::size_t size_bytes, unsigned flags);
    AX_HOST_ONLY void GPUHostGetDevicePointer(void **ptr_device, void *ptr_host, unsigned flags);
    AX_HOST_ONLY void GPUHostUnregister(void *ptr_host);
    /* Textures */
    AX_HOST_ONLY void GPUCreateTextureObject(cudaTextureObject_t *tex, const CudaParams &params, bool use_resc_view = false);
    AX_HOST_ONLY void GPUDestroyTextureObject(cudaTextureObject_t texture_object);

    /* Synchronizations*/
    AX_HOST_ONLY void GPUDeviceSynchronize();
  };

}  // namespace ax_cuda

#endif  // CUDADEVICE_H

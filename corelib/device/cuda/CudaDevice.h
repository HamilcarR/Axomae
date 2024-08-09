#ifndef CUDADEVICE_H
#define CUDADEVICE_H
#include "../DeviceError.h"
#include "CudaParams.h"
#include "cuda_utils.h"
#include "project_macros.h"

namespace ax_cuda {

  class CudaDevice {

   private:
    int device_id{};

   public:
    CLASS_CM(CudaDevice)

    /* Init */
    AX_HOST_ONLY explicit CudaDevice(int device);
    AX_HOST_ONLY DeviceError init(const CudaParams &params);
    AX_HOST_ONLY DeviceError set(const CudaParams &params);

    /* Memory */
    AX_HOST_ONLY DeviceError allocateMemory(void **ptr, std::size_t size);
    AX_HOST_ONLY DeviceError deallocateMemory(void *ptr);
    AX_HOST_ONLY DeviceError allocateMemoryManaged(void **ptr, std::size_t, const CudaParams &params);
    AX_HOST_ONLY DeviceError copyMemory(const void *ptr_source, void *ptr_dest, std::size_t byte_count, const CudaParams &params);
    AX_HOST_ONLY DeviceError allocateMemoryArray(cudaArray_t *array, const CudaParams &params, unsigned width, unsigned height, unsigned flags = 0);
    AX_HOST_ONLY DeviceError deallocateMemoryArray(cudaArray_t array);
    AX_HOST_ONLY DeviceError copy2DToArray(
        cudaArray_t array, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, const CudaParams &params);
    AX_HOST_ONLY DeviceError registerMemoryHost(void *ptr, std::size_t size_bytes, unsigned flags);
    AX_HOST_ONLY DeviceError getHostDevicePtr(void **ptr_device, void *ptr_host, unsigned flags);
    AX_HOST_ONLY DeviceError unregisterMemoryHost(void *ptr_host);
    /* Textures */
    AX_HOST_ONLY DeviceError createTextureObject(cudaTextureObject_t *tex, const CudaParams &params, bool use_resc_view = false);
    AX_HOST_ONLY DeviceError destroyTextureObject(cudaTextureObject_t texture_object);

    /* Synchronizations*/
    AX_HOST_ONLY DeviceError synchronize();
  };

}  // namespace ax_cuda

#endif  // CUDADEVICE_H

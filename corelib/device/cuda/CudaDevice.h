#ifndef CUDADEVICE_H
#define CUDADEVICE_H
#include "../GPUBackendInterface.h"
#include "cuda_utils.h"
#include "kernel/kernel_launcher.cuh"
#include "project_macros.h"
namespace ax_cuda {

  class CudaDevice : public GPUBackendInterface {

   private:
    int device_id{};

   public:
    CLASS_OCM(CudaDevice)

    AX_HOST_ONLY explicit CudaDevice(int device);
    AX_HOST_ONLY DeviceError init(const DeviceParams &params) override;
    AX_HOST_ONLY DeviceError set(const DeviceParams &params) override;
    AX_HOST_ONLY DeviceError allocateMemory(void **ptr, std::size_t size) override;
    AX_HOST_ONLY DeviceError deallocateMemory(void *ptr) override;
    AX_HOST_ONLY DeviceError allocateMemoryManaged(void **ptr, std::size_t, const DeviceParams &params) override;
    AX_HOST_ONLY DeviceError synchronize() override;
    AX_HOST_ONLY DeviceError copyMemory(const void *ptr_source, void *ptr_dest, std::size_t byte_count, const DeviceParams &params) override;
  };

}  // namespace ax_cuda

#endif  // CUDADEVICE_H

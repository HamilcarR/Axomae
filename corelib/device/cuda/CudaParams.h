//
// Created by hamilcar on 8/6/24.
//

#ifndef CUDAPARAMS_H
#define CUDAPARAMS_H
#include "../DeviceParams.h"
#include "CudaDevice.h"
#include "project_macros.h"

namespace ax_cuda {

  class CudaParams : public DeviceParams {

   private:
    int device_id{};
    unsigned deviceFlags{};
    unsigned flags{};
    cudaMemcpyKind memcpy_kind;

   public:
    CLASS_OCM(CudaParams)

    [[nodiscard]] int getDeviceID() const override { return device_id; }
    void setDeviceID(int device_id) override { this->device_id = device_id; }
    [[nodiscard]] unsigned getDeviceFlags() const override { return deviceFlags; }
    void setDeviceFlags(unsigned device_flags) override { deviceFlags = device_flags; }
    [[nodiscard]] unsigned getFlags() const override { return flags; }
    void setFlags(unsigned flags) override { this->flags = flags; }
    void setMemcpyKind(unsigned copy_kind) override;
    [[nodiscard]] int getMemcpyKind() const override { return memcpy_kind; }
  };
}  // namespace ax_cuda
#endif  // CUDAPARAMS_H

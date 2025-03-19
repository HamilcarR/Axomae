#ifndef DEVICEMEMORYRESOURCETRACKERINTERFACE_H
#define DEVICEMEMORYRESOURCETRACKERINTERFACE_H

namespace device::gpgpu {
  class GPUStream;
}

class DeviceMemoryResourceTrackerInterface {
 public:
  virtual ~DeviceMemoryResourceTrackerInterface() = default;
  virtual void mapResource() = 0;
  virtual void unmapResource() = 0;
  virtual void mapResource(device::gpgpu::GPUStream &stream) = 0;
  virtual void unmapResource(device::gpgpu::GPUStream &stream) = 0;
  virtual void mapBuffer() = 0;
  virtual bool isValid() const = 0;
};

#endif  // DEVICEMEMORYRESOURCETRACKERINTERFACE_H

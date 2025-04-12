#ifndef DEVICE_ACCELERATION_INTERFACE_H
#define DEVICE_ACCELERATION_INTERFACE_H
#include "aggregate_datastructures.h"
namespace nova::aggregate {

  /**
   * Used as a wrapper around a gpu traversable AS handler.
   * Always runs on device.
   */
  class DeviceIntersectorInterface {
   public:
    ax_device_only virtual ~DeviceIntersectorInterface() = default;
    ax_device_only virtual bool hit(const Ray &ray, bvh_hit_data &hit_data) = 0;
  };

  using AcceleratorHandle = unsigned long long;
  class DeviceIntersector : public DeviceIntersectorInterface {
    AcceleratorHandle accelerator;

   public:
    ax_device_only DeviceIntersector(AcceleratorHandle handle_id);
    ax_device_only bool hit(const Ray &ray, bvh_hit_data &hit_data) override;
  };

  /**
   * Abstraction of a generic gpu AS builder.
   * Always runs on host.
   */
  class DeviceAcceleratorInterface {
   public:
    virtual ~DeviceAcceleratorInterface() = default;
    virtual AcceleratorHandle build(primitive_aggregate_data_s primitive_data_list) = 0;
    virtual void cleanup() = 0;
    /**
     * Uses currently built backend to return an opaque AS builder.
     */
    static std::unique_ptr<DeviceAcceleratorInterface> make();  // Independently implemented in api/devaccel_factory.cpp
  };

}  // namespace nova::aggregate

#endif  // DEVICE_ACCELERATION_INTERFACE_H

#ifndef DEVICE_ACCELERATION_INTERFACE_H
#define DEVICE_ACCELERATION_INTERFACE_H
#include "aggregate_datastructures.h"
namespace nova::aggregate {

  /**
   * Used as a wrapper around a gpu traversable GAS handler.
   */
  class DeviceIntersectorInterface {
   public:
    virtual ~DeviceIntersectorInterface() = default;
    virtual bool hit(const Ray &ray, bvh_hit_data &hit_data) = 0;
  };

  using AcceleratorHandle = unsigned long long;

  /**
   * Abstraction of a generic gpu GAS builder.
   * Always runs on host.
   */
  class DeviceAcceleratorInterface {
   public:
    virtual ~DeviceAcceleratorInterface() = default;
    virtual AcceleratorHandle build(primitive_aggregate_data_s primitive_data_list) = 0;
    virtual void cleanup() = 0;
    virtual unsigned getMaxRecursiveDepth() const = 0;

    /**
     * Uses currently built backend to return an opaque GAS builder.
     */
    static std::unique_ptr<DeviceAcceleratorInterface> make();  // Independently implemented in api/devaccel_factory.cpp
  };

  class DeviceIntersector : public DeviceIntersectorInterface {
    AcceleratorHandle accelerator;

   public:
    DeviceIntersector(AcceleratorHandle handle_id);
    bool hit(const Ray &ray, bvh_hit_data &hit_data) override;
  };

}  // namespace nova::aggregate

#endif  // DEVICE_ACCELERATION_INTERFACE_H

#ifndef DEVICE_ACCELERATION_INTERFACE_H
#define DEVICE_ACCELERATION_INTERFACE_H
#include "aggregate_datastructures.h"

namespace nova {
  struct device_traversal_param_s;
}

namespace nova::aggregate {

  class DeviceIntersectorInterface {
   public:
    virtual ~DeviceIntersectorInterface() = default;
    virtual void traverse(const device_traversal_param_s &params) const = 0;
  };

  /**
   * Abstraction of a generic gpu GAS builder.
   * Always runs on host.
   */
  class DeviceAcceleratorInterface {
   public:
    virtual ~DeviceAcceleratorInterface() = default;
    virtual void build(primitive_aggregate_data_s primitive_data_list) = 0;
    virtual void cleanup() = 0;
    virtual unsigned getMaxRecursiveDepth() const = 0;
    virtual void copyParamsToDevice(const device_traversal_param_s &params) const = 0;
    /**
     * Retrieves an intersector instance containing the api data necessary for a gpu job launch.
     * Ex : Pipeline , sbt , pipeline parameters etc.
     */
    virtual std::unique_ptr<DeviceIntersectorInterface> getIntersectorObject() const = 0;

    /**
     * Uses currently built backend to return an opaque GAS builder.
     */
    static std::unique_ptr<DeviceAcceleratorInterface> make();  // Independently implemented in api/devaccel_factory.cpp
  };

}  // namespace nova::aggregate

#endif  // DEVICE_ACCELERATION_INTERFACE_H

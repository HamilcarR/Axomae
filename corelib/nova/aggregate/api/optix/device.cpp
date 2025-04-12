#include "aggregate/device_acceleration_interface.h"
#include <optix.h>

namespace nova::aggregate {

  ax_device_only DeviceIntersector::DeviceIntersector(OptixTraversableHandle handle_id) : accelerator(handle_id) {}

  ax_device_only bool DeviceIntersector::hit(const Ray &ray, bvh_hit_data &hit_data) { return false; }

}  // namespace nova::aggregate

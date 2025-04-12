#include "../device_acceleration_interface.h"
#include <optix_host.h>

namespace nova::aggregate {

  ax_device_only OptixIntersector::OptixIntersector(OptixTraversableHandle handle_id) : accelerator(handle_id) {}

  ax_device_only bool OptixIntersector::hit(const Ray &ray, bvh_hit_data &hit_data) { return false; }

}  // namespace nova::aggregate
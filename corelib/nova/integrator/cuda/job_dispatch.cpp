#include "integrator/gpu_launcher.h"
#include "manager/ManagerInternalStructs.h"
#include "manager/NovaResourceManager.h"

namespace nova {

  void device_start_integrator(const device_traversal_param_s &parameters, nova_eng_internals &nova_internals) {
    AX_ASSERT_NOTNULL(nova_internals.resource_manager);
    const aggregate::DeviceAcceleratorInterface *accel = nova_internals.resource_manager->getGpuManagedAccelerator();
    std::unique_ptr<aggregate::DeviceIntersectorInterface> device_intersector = accel->getIntersectorObject();
    device_intersector->traverse(parameters);
  }

}  // namespace nova

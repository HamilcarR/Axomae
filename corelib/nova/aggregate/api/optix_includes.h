#ifndef OPTIX_H
#define OPTIX_H
#include "../device_acceleration_interface.h"

/**
 * Internal header file. Don't include it outside of api/
 */
namespace nova::aggregate {

  class BackendOptix : public DeviceAcceleratorInterface {

   public:
    BackendOptix();

    AcceleratorHandle build(primitive_aggregate_data_s primitive_data_list) override;
    void cleanup() override;
  };

}  // namespace nova::aggregate

#endif  // OPTIX_H

#ifndef ACCELERATIONINTERNALSINTERFACE_H
#define ACCELERATIONINTERNALSINTERFACE_H
#include "aggregate_datastructures.h"

namespace nova::aggregate {

  template<class SUBTYPE>
  class AccelerationInternalsInterface {
   public:
    void build(primitive_aggregate_data_s scene) { static_cast<SUBTYPE *>(this)->build(scene); }
    bool intersect(const Ray &ray, bvh_hit_data &hit_data) const { return static_cast<SUBTYPE *>(this)->intersect(ray, hit_data); }
    void cleanup() { static_cast<SUBTYPE *>(this)->cleanup(); };
  };

}  // namespace nova::aggregate

#endif

#ifndef ACCELERATION_INTERFACE_H
#define ACCELERATION_INTERFACE_H
#include "aggregate_datastructures.h"
#include <internal/common/utils.h>

/* Abstracts CPU acceleration structures calls.  */

namespace nova::aggregate {

  struct EmbreeBuild {};
  struct NativeBuild {};

  template<class AccelerationBackend = std::conditional_t<core::build::is_embree_build, EmbreeBuild, NativeBuild>>
  class GenericHostAccelerator {
    class Impl;
    std::unique_ptr<Impl> pimpl;

   public:
    GenericHostAccelerator();
    ~GenericHostAccelerator();
    GenericHostAccelerator(const GenericHostAccelerator &) = delete;
    GenericHostAccelerator &operator=(const GenericHostAccelerator &) = delete;
    GenericHostAccelerator(GenericHostAccelerator &&) noexcept;
    GenericHostAccelerator &operator=(GenericHostAccelerator &&) noexcept;

    /* Takes global transformations for each mesh. */
    void build(primitive_aggregate_data_s primitives_data_list);
    /* Tests for single ray.*/
    bool hit(const Ray &ray, bvh_hit_data &hit_data) const;
    void cleanup();
  };
  using DefaultAccelerator = GenericHostAccelerator<>;

}  // namespace nova::aggregate

#endif

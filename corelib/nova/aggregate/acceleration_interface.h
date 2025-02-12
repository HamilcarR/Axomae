#ifndef ACCELERATION_INTERFACE_H
#define ACCELERATION_INTERFACE_H
#include "primitive/PrimitiveInterface.h"
#include <internal/common/utils.h>
#include <type_traits>
/* Abstracts Embree and Optix calls */

struct bvh_hit_data {
  bool is_hit{false};
  const nova::primitive::NovaPrimitiveInterface *last_primit{nullptr};
  nova::hit_data hit_d;
  float prim_min_t{};
  float prim_max_t{};
  const bool *is_rendering{nullptr};
};

namespace nova::aggregate {

  struct primitive_aggregate_data_s {
    primitives_view_tn primitive_list_view;
    shape::mesh_shared_views_t mesh_geometry;
  };

  struct EmbreeBuild {};
  struct NativeBuild {};

  template<class AccelerationBackend = std::conditional_t<core::build::is_embree_build, EmbreeBuild, NativeBuild>>
  class GenericAccelerator {
    class Impl;
    std::unique_ptr<Impl> pimpl;

   public:
    GenericAccelerator();
    ~GenericAccelerator();
    GenericAccelerator(const GenericAccelerator &) = delete;
    GenericAccelerator &operator=(const GenericAccelerator &) = delete;
    GenericAccelerator(GenericAccelerator &&) noexcept;
    GenericAccelerator &operator=(GenericAccelerator &&) noexcept;

    /* Takes global transformations for each mesh. */
    void build(primitive_aggregate_data_s primitives_data_list);
    /* Tests for single ray.*/
    bool hit(const Ray &ray, bvh_hit_data &hit_data) const;
    void cleanup();
  };

  using DefaultAccelerator = GenericAccelerator<>;

}  // namespace nova::aggregate

#endif

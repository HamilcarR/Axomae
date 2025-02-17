#ifndef AGGREGATE_DATASTRUCTURES_H
#define AGGREGATE_DATASTRUCTURES_H

#include "primitive/PrimitiveInterface.h"

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

}  // namespace nova::aggregate
#endif

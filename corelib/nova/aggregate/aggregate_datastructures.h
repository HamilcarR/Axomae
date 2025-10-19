#ifndef AGGREGATE_DATASTRUCTURES_H
#define AGGREGATE_DATASTRUCTURES_H

#include "primitive/PrimitiveInterface.h"

struct bvh_hit_data {
  bool is_hit{false};
  const nova::primitive::NovaPrimitiveInterface *last_primit{nullptr};
  nova::intersection_record_s hit_d;
  float prim_min_t{};
  float prim_max_t{};
  const bool *is_rendering{nullptr};
};

namespace nova::aggregate {

  struct primitive_aggregate_data_s {
    primitive::primitives_view_tn primitive_list_view;
    shape::MeshBundleViews mesh_geometry;
  };

}  // namespace nova::aggregate
#endif

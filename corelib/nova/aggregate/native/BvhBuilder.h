#ifndef BVHBUILDER_H
#define BVHBUILDER_H
#include "../aggregate_datastructures.h"
#include "primitive/PrimitiveInterface.h"
#include "shape/MeshContext.h"
#include <internal/common/axstd/span.h>
#include <vector>
namespace nova::aggregate {
  /* bvh node.*/
  struct Bvhnl {
    float min[3];
    float max[3];
    int32_t left;
    int32_t primitive_count;
  };

  struct Bvht_data {
    std::vector<Bvhnl> l_tree;
    std::vector<int32_t> prim_idx;
  };

  class BvhtlBuilder {
   public:
    enum SEGMENTATION : unsigned { SAH, HLBVH };
    enum BUILD_TYPE : unsigned {
      QUALITY,
      MEDIUM,
      PERFORMANCE

    };

   public:
    static Bvht_data buildTriangleBasedScene(const primitive_aggregate_data_s &scene,
                                             BUILD_TYPE build_option = PERFORMANCE,
                                             SEGMENTATION segmentation = SAH);
  };
}  // namespace nova::aggregate
#endif  // BVHBUILDER_H

#ifndef BVHBUILDER_H
#define BVHBUILDER_H

#include "primitive/nova_primitive.h"
#include <memory>
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

   public:
    static Bvht_data build(const std::vector<std::unique_ptr<primitive::NovaPrimitiveInterface>> &primitives, SEGMENTATION segmentation = SAH);

   private:
    static void update_aabb(const std::vector<std::unique_ptr<primitive::NovaPrimitiveInterface>> &primitives, int32_t node_id, Bvht_data &bvh_data);
    static void subdivide(const std::vector<std::unique_ptr<primitive::NovaPrimitiveInterface>> &primitives,
                          int32_t node_id,
                          int32_t &nodes_used,
                          SEGMENTATION segmentation,
                          Bvht_data &bvh_data);
  };
}  // namespace nova::aggregate
#endif  // BVHBUILDER_H

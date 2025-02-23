#ifndef BVHBUILDER_H
#define BVHBUILDER_H
#include "aggregate/aggregate_datastructures.h"
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
    struct cached_primitive_data_s {
      std::vector<geometry::BoundingBox> aabbs;
      std::vector<glm::vec3> centroids;
    };

    struct geometric_data_s {
      const cached_primitive_data_s *precomputed_cache;
      const shape::MeshCtx *geometry;
      const primitives_view_tn *primitives;
    };

   public:
    enum SEGMENTATION : unsigned { SAH, HLBVH };
    enum BUILD_TYPE : unsigned {
      QUALITY,
      MEDIUM,
      PERFORMANCE

    };

   public:
    static Bvht_data build(const primitive_aggregate_data_s &scene, BUILD_TYPE build_option = PERFORMANCE, SEGMENTATION segmentation = SAH);
    static Bvht_data buildTriangleBasedScene(const primitive_aggregate_data_s &scene,
                                             BUILD_TYPE build_option = PERFORMANCE,
                                             SEGMENTATION segmentation = SAH);

   private:
    static int32_t create_nodes(const geometric_data_s &scene, std::vector<int32_t> &prim_idx, Bvhnl &node, const float split_axis[3], int axis);
    static int axis_subdiv_sah(const geometric_data_s &scene,
                               const Bvht_data &bvh_tree_data,
                               int32_t node_id,
                               float &best_coast_r,
                               float subdivided_axis[3],
                               BvhtlBuilder::BUILD_TYPE build_type);
    static float eval_sah(const geometric_data_s &scene, const Bvht_data &tree, const Bvhnl &node, int axis, float candidate_pos);
    static cached_primitive_data_s precompute_cached_prim_data(const shape::MeshCtx &geometry, const primitive_aggregate_data_s &scene);
    static void update_aabb(const geometric_data_s &scene_data, int32_t node_id, Bvht_data &bvh_data);
    static void subdivide(const geometric_data_s &scene_data,
                          int32_t node_id,
                          int32_t &nodes_used,
                          BUILD_TYPE build_type,
                          SEGMENTATION segmentation,
                          Bvht_data &bvh_data);
  };
}  // namespace nova::aggregate
#endif  // BVHBUILDER_H

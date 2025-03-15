#include "BvhBuilder.h"
#include "aggregate_datastructures.h"
#include "primitive/PrimitiveInterface.h"
#include "shape/MeshContext.h"
#include <internal/common/axstd/span.h>
#include <internal/common/exception/GenericException.h>
#include <internal/geometry/BoundingBox.h>

namespace exception {
  class BadSizeException : public GenericException {
   public:
    explicit BadSizeException(const char *error_msg) : GenericException() { saveErrorString(error_msg); }
  };
}  // namespace exception

namespace nova::aggregate {

  struct cached_primitive_data_s {
    std::vector<geometry::BoundingBox> aabbs;
    std::vector<glm::vec3> centroids;
  };

  struct geometric_data_s {
    const cached_primitive_data_s *precomputed_cache;
    const shape::MeshCtx *geometry;
    const primitives_view_tn *primitives;
  };

  static void update_aabb(const geometric_data_s &scene, int32_t node_id, Bvht_data &bvh_data);
  static void subdivide(const geometric_data_s &scene,
                        int32_t node_id,
                        int32_t &nodes_used,
                        BvhtlBuilder::BUILD_TYPE build_type,
                        BvhtlBuilder::SEGMENTATION seg,
                        Bvht_data &bvh_data);

  cached_primitive_data_s precompute_cached_prim_data(const shape::MeshCtx &geometry, const primitive_aggregate_data_s &scene) {
    cached_primitive_data_s cache;
    cache.aabbs.reserve(scene.primitive_list_view.size());
    cache.centroids.reserve(scene.primitive_list_view.size());
    for (const auto &prim : scene.primitive_list_view) {
      cache.aabbs.push_back(prim.computeAABB(geometry));
      cache.centroids.push_back(prim.centroid(geometry));
    }
    return cache;
  }

  Bvht_data BvhtlBuilder::buildTriangleBasedScene(const primitive_aggregate_data_s &scene, BUILD_TYPE type, SEGMENTATION segmentation) {
    const primitives_view_tn &primitives = scene.primitive_list_view;
    Bvht_data bvh_data;
    const std::size_t prim_size = primitives.size();
    if (prim_size == 0)
      throw ::exception::BadSizeException("Primitives list is empty.");
    /* max nodes for a btree is 2N-1*/
    bvh_data.l_tree.resize(2 * prim_size - 1);
    bvh_data.prim_idx.reserve(primitives.size());
    for (int i = 0; i < primitives.size(); i++)
      bvh_data.prim_idx.push_back(i);
    Bvhnl &root = bvh_data.l_tree[0];
    root.left = 0;
    AX_ASSERT_LT(prim_size, INT32_MAX);
    root.primitive_count = (int32_t)prim_size;
    shape::MeshCtx geometry = shape::MeshCtx(scene.mesh_geometry);
    auto cache = precompute_cached_prim_data(geometry, scene);
    geometric_data_s scene_data_struct{};
    scene_data_struct.geometry = &geometry;
    scene_data_struct.primitives = &primitives;
    scene_data_struct.precomputed_cache = &cache;

    update_aabb(scene_data_struct, 0, bvh_data);
    int32_t node_count = 1;
    subdivide(scene_data_struct, 0, node_count, type, segmentation, bvh_data);
    return bvh_data;
  }

  /* Makes a union between two AABBs , merging them , creating a new bounding box with the highests coordinates and lowests.*/
  void update_aabb(const geometric_data_s &scene, int32_t node_id, Bvht_data &bvh_data) {
    AX_ASSERT_FALSE(scene.primitives->empty());
    AX_ASSERT_LT(node_id, bvh_data.l_tree.size());

    Bvhnl &node = bvh_data.l_tree[node_id];
    geometry::BoundingBox node_box(glm::vec3(INT_MAX), glm::vec3(INT_MIN));
    size_t offset = node.left;
    for (size_t i = 0; i < node.primitive_count; i++) {
      int32_t idx = bvh_data.prim_idx[i + offset];
      AX_ASSERT_LT(idx, scene.primitives->size());
      AX_ASSERT_NOTNULL((*scene.primitives)[idx].get());
      geometry::BoundingBox current = scene.precomputed_cache->aabbs[idx];
      node_box = node_box + current;

      node.min[0] = node_box.getMinCoords().x;
      node.min[1] = node_box.getMinCoords().y;
      node.min[2] = node_box.getMinCoords().z;

      node.max[0] = node_box.getMaxCoords().x;
      node.max[1] = node_box.getMaxCoords().y;
      node.max[2] = node_box.getMaxCoords().z;
    }
  }

  static int axis_subdiv(const std::vector<Bvhnl> &l_tree, int32_t node_id, float subdivided_axis[3]) {
    AX_ASSERT_LT(node_id, l_tree.size());
    const Bvhnl &node = l_tree[node_id];
    float extent[3] = {node.max[0] - node.min[0], node.max[1] - node.min[1], node.max[2] - node.min[2]};
    int axis = 0;
    if (extent[0] <= extent[1])
      axis = 1;
    if (extent[axis] <= extent[2])
      axis = 2;
    subdivided_axis[axis] = extent[axis] * 0.5f + node.min[axis];
    return axis;
  }

  float eval_sah(const geometric_data_s &scene, const Bvht_data &tree, const Bvhnl &node, int axis, float candidate_pos) {
    AX_ASSERT_LT(axis, 3);
    AX_ASSERT_GE(axis, 0);
    geometry::BoundingBox aabb_left, aabb_right;
    int prim_left_count = 0, prim_right_count = 0;
    for (int32_t i = 0; i < node.primitive_count; i++) {
      const int32_t offset = i + node.left;
      AX_ASSERT_LT(offset, tree.prim_idx.size());
      const int32_t p_idx = tree.prim_idx[offset];
      AX_ASSERT_LT(p_idx, scene.primitives->size());
      const glm::vec3 centroid_vec = scene.precomputed_cache->centroids[p_idx];
      const float *centroid_ptr = glm::value_ptr(centroid_vec);
      const geometry::BoundingBox prim_aabb = scene.precomputed_cache->aabbs[p_idx];
      if (centroid_ptr[axis] < candidate_pos) {
        prim_left_count++;
        aabb_left = aabb_left + prim_aabb;
      } else {
        prim_right_count++;
        aabb_right = aabb_right + prim_aabb;
      }
    }
    const float cost = aabb_left.halfArea() * (float)prim_left_count + aabb_right.halfArea() * (float)prim_right_count;
    return cost > 0.f ? cost : INT_MAX;
  }

  static unsigned get_centroid_seg_number(BvhtlBuilder::BUILD_TYPE build_type) {
    switch (build_type) {
      case BvhtlBuilder::QUALITY:
        return 256;
      case BvhtlBuilder::MEDIUM:
        return 32;
      case BvhtlBuilder::PERFORMANCE:
        return 4;
    }
    return 4;
  }

  int axis_subdiv_sah(const geometric_data_s &scene,
                      const Bvht_data &bvh_tree_data,
                      int32_t node_id,
                      float &best_coast_r,
                      float subdivided_axis[3],
                      BvhtlBuilder::BUILD_TYPE build_type) {

    const unsigned SEGMENT_COUNT = get_centroid_seg_number(build_type);
    int best_axis = 0;
    float best_position = 0.f;
    float best_cost = INT_MAX;
    const Bvhnl &node = bvh_tree_data.l_tree[node_id];

    for (int axis = 0; axis < 3; axis++) {
      const float bound_min = node.min[axis];
      const float bound_max = node.max[axis];
      float aabb_axis_dist = bound_max - bound_min;
      if (aabb_axis_dist == 0)
        continue;
      float segment_size = aabb_axis_dist / (float)SEGMENT_COUNT;
      for (int i = 0; i < SEGMENT_COUNT; i++) {
        float candidate = segment_size * i + bound_min;
        const float cost = eval_sah(scene, bvh_tree_data, node, axis, candidate);
        if (cost < best_cost) {
          best_cost = cost;
          best_axis = axis;
          best_position = candidate;
        }
      }
    }
    subdivided_axis[best_axis] = best_position;
    best_coast_r = best_cost;
    return best_axis;
  }

  int32_t create_nodes(const geometric_data_s &scene, std::vector<int32_t> &prim_idx, Bvhnl &node, const float split_axis[3], int axis) {

    int32_t i = node.left;
    int32_t j = i + node.primitive_count - 1;
    /* i is the index of prim_idx under which every primitive is at the left of the split plane*/
    while (i <= j) {
      AX_ASSERT_GE(j, 0);
      int32_t p_idx = prim_idx[i];
      AX_ASSERT_LT(p_idx, scene.primitives->size());
      const glm::vec3 primitive_centroid = scene.precomputed_cache->centroids[p_idx];
      float centroid = axis == 0 ? primitive_centroid.x : axis == 1 ? primitive_centroid.y : primitive_centroid.z;
      if (centroid < split_axis[axis])
        i++;
      else
        std::swap(prim_idx[i], prim_idx[j--]);
    }
    return i;
  }

  void subdivide(const geometric_data_s &scene,
                 int32_t node_id,
                 int32_t &nodes_used,
                 BvhtlBuilder::BUILD_TYPE build_type,
                 BvhtlBuilder::SEGMENTATION seg,
                 Bvht_data &bvh_data) {
    Bvhnl &node = bvh_data.l_tree[node_id];
    float split_axis[3] = {0};
    float best_cost = 0;

    int axis = -1;
    /* Retrieves longest axis */
    if (seg != BvhtlBuilder::SAH)
      axis = axis_subdiv(bvh_data.l_tree, node_id, split_axis);
    else
      /* SAH */
      axis = axis_subdiv_sah(scene, bvh_data, node_id, best_cost, split_axis, build_type);

    geometry::BoundingBox node_bbox(node.min, node.max);
    const float node_bbox_area = node_bbox.halfArea();
    const float cost = node_bbox_area * (float)node.primitive_count;
    if (best_cost >= cost)
      return;
    /* Sort the index array for each primitive at the left of the split_axis*/
    int i = create_nodes(scene, bvh_data.prim_idx, node, split_axis, axis);

    int left_count = i - node.left;
    if (left_count == 0 || left_count == node.primitive_count)
      return;
    int32_t left_idx = nodes_used++;
    int32_t right_idx = nodes_used++;
    Bvhnl &left = bvh_data.l_tree[left_idx];
    Bvhnl &right = bvh_data.l_tree[right_idx];

    left.left = node.left;
    left.primitive_count = left_count;

    right.left = i;
    right.primitive_count = node.primitive_count - left_count;

    AX_ASSERT_GE(right.primitive_count, 0);
    AX_ASSERT_GE(left.primitive_count, 0);
    node.primitive_count = 0;
    node.left = left_idx;
    update_aabb(scene, left_idx, bvh_data);
    update_aabb(scene, right_idx, bvh_data);
    subdivide(scene, left_idx, nodes_used, build_type, seg, bvh_data);
    subdivide(scene, right_idx, nodes_used, build_type, seg, bvh_data);
  }
}  // namespace nova::aggregate

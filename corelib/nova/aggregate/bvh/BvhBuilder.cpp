#include "BvhBuilder.h"

#include "primitive/nova_primitive.h"

#include <GenericException.h>

namespace exception {
  class BadSizeException : public GenericException {
   public:
    explicit BadSizeException(const char *error_msg) : GenericException() { saveErrorString(error_msg); }
  };
}  // namespace exception

namespace nova::aggregate {

  Bvht_data BvhtlBuilder::build(const std::vector<std::unique_ptr<primitive::NovaPrimitiveInterface>> &primitives,
                                BUILD_TYPE type,
                                SEGMENTATION segmentation) {

    Bvht_data bvh_data;
    const int32_t prim_size = primitives.size();
    if (prim_size == 0)
      throw ::exception::BadSizeException("Primitives list is empty.");
    /* max nodes for a btree is 2N-1*/
    bvh_data.l_tree.resize(2 * prim_size - 1);
    bvh_data.prim_idx.reserve(primitives.size());
    for (int i = 0; i < primitives.size(); i++)
      bvh_data.prim_idx.push_back(i);
    Bvhnl &root = bvh_data.l_tree[0];
    root.left = 0;
    root.primitive_count = prim_size;
    update_aabb(primitives, 0, bvh_data);
    int32_t node_count = 1;
    subdivide(primitives, 0, node_count, type, segmentation, bvh_data);
    return bvh_data;
  }

  void BvhtlBuilder::update_aabb(const std::vector<std::unique_ptr<primitive::NovaPrimitiveInterface>> &primitives,
                                 int32_t node_id,
                                 Bvht_data &bvh_data) {
    AX_ASSERT_FALSE(primitives.empty());
    AX_ASSERT_LT(node_id, bvh_data.l_tree.size());

    Bvhnl &node = bvh_data.l_tree[node_id];
    geometry::BoundingBox node_box(glm::vec3(INT_MAX), glm::vec3(INT_MIN));
    size_t offset = node.left;
    for (size_t i = 0; i < node.primitive_count; i++) {
      int32_t idx = bvh_data.prim_idx[i + offset];
      AX_ASSERT_LT(idx, primitives.size());
      AX_ASSERT_NOTNULL(primitives[idx]);
      geometry::BoundingBox current = primitives[idx]->computeAABB();
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

  static float eval_sah(const Bvht_data &tree,
                        const std::vector<std::unique_ptr<nova::primitive::NovaPrimitiveInterface>> &primitives,
                        const Bvhnl &node,
                        int axis,
                        float candidate_pos) {
    AX_ASSERT_LT(axis, 3);
    AX_ASSERT_GE(axis, 0);
    geometry::BoundingBox aabb_left, aabb_right;
    int prim_left_count = 0, prim_right_count = 0;
    for (int32_t i = 0; i < node.primitive_count; i++) {
      const int32_t offset = i + node.left;
      AX_ASSERT_LT(offset, tree.prim_idx.size());
      const int32_t p_idx = tree.prim_idx[offset];
      AX_ASSERT_LT(p_idx, primitives.size());
      const nova::primitive::NovaPrimitiveInterface *primitive = primitives[p_idx].get();
      const glm::vec3 centroid_vec = primitive->centroid();
      const float *centroid_ptr = glm::value_ptr(centroid_vec);
      const geometry::BoundingBox prim_aabb = primitive->computeAABB();
      if (centroid_ptr[axis] < candidate_pos) {
        prim_left_count++;
        aabb_left = aabb_left + prim_aabb;
      } else {
        prim_right_count++;
        aabb_right = aabb_right + prim_aabb;
      }
    }
    const float cost = aabb_left.area() * (float)prim_left_count + aabb_right.area() * (float)prim_right_count;
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

  static int axis_subdiv_sah(const Bvht_data &bvh_tree_data,
                             const std::vector<std::unique_ptr<nova::primitive::NovaPrimitiveInterface>> &primitives,
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
        const float cost = eval_sah(bvh_tree_data, primitives, node, axis, candidate);
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

  static int32_t create_nodes(const std::vector<std::unique_ptr<nova::primitive::NovaPrimitiveInterface>> &primitives,
                              std::vector<int32_t> &prim_idx,
                              Bvhnl &node,
                              const float split_axis[3],
                              int axis) {

    int32_t i = node.left;
    int32_t j = i + node.primitive_count - 1;
    /* i is the index of prim_idx under which every primitive is at the left of the split plane*/
    while (i <= j) {
      AX_ASSERT_GE(j, 0);
      int32_t p_idx = prim_idx[i];
      AX_ASSERT_LT(p_idx, primitives.size());
      const nova::primitive::NovaPrimitiveInterface *primitive = primitives[p_idx].get();
      float centroid = axis == 0 ? primitive->centroid().x : axis == 1 ? primitive->centroid().y : primitive->centroid().z;
      if (centroid < split_axis[axis])
        i++;
      else
        std::swap(prim_idx[i], prim_idx[j--]);
    }
    return i;
  }

  void BvhtlBuilder::subdivide(const std::vector<std::unique_ptr<primitive::NovaPrimitiveInterface>> &primitives,
                               int32_t node_id,
                               int32_t &nodes_used,
                               BUILD_TYPE build_type,
                               SEGMENTATION seg,
                               Bvht_data &bvh_data) {
    Bvhnl &node = bvh_data.l_tree[node_id];
    float split_axis[3] = {0};
    float best_cost = 0;

    int axis = -1;
    /* Retrieves longest axis */
    if (seg != SAH)
      axis = axis_subdiv(bvh_data.l_tree, node_id, split_axis);
    else
      /* SAH */
      axis = axis_subdiv_sah(bvh_data, primitives, node_id, best_cost, split_axis, build_type);

    geometry::BoundingBox node_bbox(node.min, node.max);
    const float node_bbox_area = node_bbox.area();
    const float cost = node_bbox_area * (float)node.primitive_count;
    if (best_cost >= cost)
      return;
    /* Sort the index array for each primitive at the left of the split_axis*/
    int i = create_nodes(primitives, bvh_data.prim_idx, node, split_axis, axis);

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
    update_aabb(primitives, left_idx, bvh_data);
    update_aabb(primitives, right_idx, bvh_data);
    subdivide(primitives, left_idx, nodes_used, build_type, seg, bvh_data);
    subdivide(primitives, right_idx, nodes_used, build_type, seg, bvh_data);
  }
}  // namespace nova::aggregate
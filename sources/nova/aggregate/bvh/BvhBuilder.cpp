#include "BvhBuilder.h"

#include "nova_primitive.h"

using namespace nova::aggregate;
using namespace nova::primitive;

Bvht_data BvhtlBuilder::build(const std::vector<std::unique_ptr<primitive::NovaPrimitiveInterface>> &primitives, SEGMENTATION segmentation) {
  Bvht_data bvh_data;
  const int32_t prim_size = primitives.size();
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
  subdivide(primitives, 0, node_count, segmentation, bvh_data);
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

static float eval_sah(const Bvht_data &tree,
                      const std::vector<std::unique_ptr<nova::primitive::NovaPrimitiveInterface>> &primitives,
                      const Bvhnl &node,
                      int axis) {}

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
                             SEGMENTATION seg,
                             Bvht_data &bvh_data) {
  Bvhnl &node = bvh_data.l_tree[node_id];
  if (node.primitive_count <= 2)
    return;
  float split_axis[3] = {0};

  /* Retrieves longest axis */
  int axis = axis_subdiv(bvh_data.l_tree, node_id, split_axis);

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
  subdivide(primitives, left_idx, nodes_used, seg, bvh_data);
  subdivide(primitives, right_idx, nodes_used, seg, bvh_data);
}

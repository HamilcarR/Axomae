#include "Bvh.h"
#include "internal/geometry/BoundingBox.h"
#include "primitive/PrimitiveInterface.h"
#include "ray/Ray.h"
#include "shape/Box.h"
#include <deque>

namespace nova::aggregate {
  namespace prim = nova::primitive;
  Bvhtl::Bvhtl(const axstd::span<primitive::NovaPrimitiveInterface> &primitives_, BvhtlBuilder::BUILD_TYPE type, BvhtlBuilder::SEGMENTATION seg)
      : primitives(primitives_) {
    AX_ASSERT_FALSE(primitives.empty());
    bvh = BvhtlBuilder::build(primitives, type, seg);
  }

  void Bvhtl::build(const axstd::span<primitive::NovaPrimitiveInterface> &primitives_,
                    BvhtlBuilder::BUILD_TYPE type,
                    BvhtlBuilder::SEGMENTATION segmentation) {
    AX_ASSERT_FALSE(primitives_.empty());
    primitives = primitives_;
    bvh = BvhtlBuilder::build(primitives, type, segmentation);
  }

  using primitive_ptr = prim::NovaPrimitiveInterface;

  bool Bvhtl::rec_traverse(const Ray &r,
                           float tmin,
                           float tmax,
                           hit_data &data,
                           base_options *user_options,
                           const axstd::span<primitive_ptr> &primitives,
                           const Bvht_data &bvh,
                           int32_t node_id) const {

    AX_ASSERT_NOTNULL(user_options);
    /* not the best solution , but dynamic cast here is worse */
    auto *hit_option = (base_options_bvh *)user_options;
    const Bvhnl &node = bvh.l_tree[node_id];
    const geometry::BoundingBox node_bbox(node.min, node.max);
    /*Big BVHs requires a stop condition or they will go on forever if we quickly want to switch scene*/
    if (!node_bbox.intersect(r.direction, r.origin))
      return false;
    if (node.primitive_count != 0) {  // node is a leaf
      bool hit = false;
      for (int32_t i = 0; i < node.primitive_count; i++) {
        if (!*hit_option->data.is_rendering)
          return false;
        int32_t offset = i + node.left;
        AX_ASSERT_LT(offset, bvh.prim_idx.size());
        int32_t p_idx = bvh.prim_idx[offset];
        AX_ASSERT_LT(p_idx, primitives.size());
        const primitive::NovaPrimitiveInterface *prim = &(primitives)[p_idx];

        if (prim->hit(r, tmin, hit_option->data.tmin, data, nullptr)) {
          hit = true;
          if (data.t <= hit_option->data.tmin) {
            hit_option->data.last_prim = prim;
            hit_option->data.tmin = data.t;
          }
        }
      }
      return hit;
    }
    bool left = rec_traverse(r, tmin, tmax, data, user_options, primitives, bvh, node.left);
    bool right = rec_traverse(r, tmin, tmax, data, user_options, primitives, bvh, node.left + 1);
    return right || left;
  }

  static constexpr int MAX_STACK_SIZE = 2048;

  inline void add2stack(const Bvhnl *node_stack[MAX_STACK_SIZE], const Bvhnl *element, int &iterator_idx) {
    if (iterator_idx + 1 < MAX_STACK_SIZE)
      node_stack[++iterator_idx] = element;
  }

  bool Bvhtl::iter_traverse(const Ray &r,
                            float tmin,
                            float /*tmax*/,
                            hit_data &data,
                            base_options *user_options,
                            const axstd::span<primitive_ptr> &primitives,
                            const Bvht_data &bvh) const {
    AX_ASSERT_NOTNULL(user_options);
    auto *options = (base_options_bvh *)user_options;
    /* Starts at root */
    const Bvhnl *iterator_node = &bvh.l_tree[0];
    AX_ASSERT_NOTNULL(iterator_node);
    bool hit = false;
    const Bvhnl *node_stack[MAX_STACK_SIZE];
    node_stack[0] = iterator_node;
    int iterator_idx = 0;
    while (iterator_idx != -1 && *options->data.is_rendering) {
      iterator_node = node_stack[iterator_idx];
      iterator_idx--;
      /* Is not a leaf */
      if (iterator_node->primitive_count == 0) {

        AX_ASSERT_LT(iterator_node->left, bvh.l_tree.size());
        AX_ASSERT_LT(iterator_node->left + 1, bvh.l_tree.size());

        const Bvhnl *left = &bvh.l_tree[iterator_node->left];
        const Bvhnl *right = &bvh.l_tree[iterator_node->left + 1];
        const geometry::BoundingBox left_aabb(left->min, left->max);
        const geometry::BoundingBox right_aabb(right->min, right->max);
        float left_intersect = left_aabb.intersect(r.direction, r.origin, options->data.tmin);
        float right_intersect = right_aabb.intersect(r.direction, r.origin, options->data.tmin);
        if (left_intersect > right_intersect) {
          std::swap(left_intersect, right_intersect);
          std::swap(left, right);
        }

        /* No intersection.*/
        if (left_intersect == MAXFLOAT)
          continue;
        /* Only intersection with left child aabb */
        if (right_intersect == MAXFLOAT) {
          if (iterator_idx + 1 < MAX_STACK_SIZE)
            add2stack(node_stack, left, iterator_idx);
          continue;
        }
        add2stack(node_stack, right, iterator_idx);
        add2stack(node_stack, left, iterator_idx);
      }

      /* Is a leaf */
      if (iterator_node->primitive_count != 0) {
        for (int32_t i = 0; i < iterator_node->primitive_count && *options->data.is_rendering; i++) {
          int32_t primitive_offset = iterator_node->left + i;
          AX_ASSERT_LT(primitive_offset, bvh.prim_idx.size());
          int32_t p_idx = bvh.prim_idx[primitive_offset];
          AX_ASSERT_LT(p_idx, primitives.size());
          const primitive::NovaPrimitiveInterface *prim = &(primitives)[p_idx];
          if (!prim->hit(r, tmin, options->data.tmin, data, nullptr))
            continue;
          hit = true;
          if (data.t <= options->data.tmin) {
            options->data.last_prim = prim;
            options->data.tmin = data.t;
          }
        }
      }
    }
    return hit;
  }

  bool Bvhtl::hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options *user_options) const {
    if (primitives.empty())
      return false;
    return iter_traverse(r, tmin, tmax, data, user_options, primitives, bvh);
  }
}  // namespace nova::aggregate
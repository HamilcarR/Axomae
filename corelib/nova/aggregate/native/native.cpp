#include "AccelerationInternalsInterface.h"
#include "BvhBuilder.h"
#include "acceleration_interface.h"
#include "primitive/PrimitiveInterface.h"
#include "ray/Hitable.h"
#include <cmath>
namespace nova::aggregate {

#define acceleration_internal_interface \
 protected \
  AccelerationInternalsInterface<GenericAccelerator<NativeBuild>::Impl>

  namespace prim = nova::primitive;
  using primitive_ptr = prim::NovaPrimitiveInterface;

  template<>
  class GenericAccelerator<NativeBuild>::Impl : acceleration_internal_interface {
    Bvht_data bvh;
    shape::MeshCtx geometry_context;
    primitive_aggregate_data_s scene;
    BvhtlBuilder::BUILD_TYPE build_type;
    BvhtlBuilder::SEGMENTATION segmentation;

   public:
    Impl() {
      build_type = BvhtlBuilder::PERFORMANCE;
      segmentation = BvhtlBuilder::SAH;
    }
    ~Impl() = default;
    Impl(const Impl &other) = default;
    Impl &operator=(const Impl &other) = default;
    Impl(Impl &&other) noexcept = default;
    Impl &operator=(Impl &&other) noexcept = default;

    void build(primitive_aggregate_data_s scene_) {
      AX_ASSERT_FALSE(scene_.primitive_list_view.empty());
      scene = scene_;
      bvh = BvhtlBuilder::buildTriangleBasedScene(scene, build_type, segmentation);
      geometry_context = shape::MeshCtx(scene.mesh_geometry);
    }

    bool intersect(const Ray &r, bvh_hit_data &data) const { return iter_traverse(r, data); }

    void cleanup() {
      bvh.l_tree.clear();
      bvh.prim_idx.clear();
    }

   private:
    static constexpr int MAX_STACK_SIZE = 2048;

    void add2stack(const Bvhnl *node_stack[MAX_STACK_SIZE], const Bvhnl *element, int &iterator_idx) const {
      if (iterator_idx + 1 < MAX_STACK_SIZE)
        node_stack[++iterator_idx] = element;
    }

    bool iter_traverse(const Ray &r, bvh_hit_data &hit_result) const {
      float tmax = MAXFLOAT;
      const Bvhnl *iterator_node = &bvh.l_tree[0];
      AX_ASSERT_NOTNULL(iterator_node);
      hit_result.is_hit = false;
      const Bvhnl *node_stack[MAX_STACK_SIZE];
      node_stack[0] = iterator_node;
      int iterator_idx = 0;
      while (iterator_idx != -1 && *hit_result.is_rendering) {
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
          float left_intersect = left_aabb.intersect(r.direction, r.origin, tmax);
          float right_intersect = right_aabb.intersect(r.direction, r.origin, tmax);
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
        if (iterator_node->primitive_count == 0)
          continue;

        for (int32_t i = 0; i < iterator_node->primitive_count && *hit_result.is_rendering; i++) {
          int32_t primitive_offset = iterator_node->left + i;
          AX_ASSERT_LT(primitive_offset, bvh.prim_idx.size());
          int32_t p_idx = bvh.prim_idx[primitive_offset];
          AX_ASSERT_LT(p_idx, scene.primitive_list_view.size());
          const primitive::NovaPrimitiveInterface *prim = &(scene.primitive_list_view)[p_idx];
          if (!prim->hit(r, r.tnear, hit_result.hit_d.t, hit_result.hit_d, geometry_context))
            continue;
          hit_result.is_hit = true;
          if (hit_result.hit_d.t <= tmax) {
            hit_result.last_primit = prim;
            hit_result.prim_min_t = hit_result.hit_d.t;
            tmax = hit_result.hit_d.t;
          }
        }
      }
      return hit_result.is_hit;
    }
  };

  /****************************************************************************************************************************************************************************/

  template<>
  GenericAccelerator<NativeBuild>::GenericAccelerator() : pimpl(std::make_unique<Impl>()) {}
  template<>
  GenericAccelerator<NativeBuild>::~GenericAccelerator() {}
  template<>
  GenericAccelerator<NativeBuild>::GenericAccelerator(GenericAccelerator &&) noexcept = default;
  template<>
  GenericAccelerator<NativeBuild> &GenericAccelerator<NativeBuild>::operator=(GenericAccelerator &&) noexcept = default;
  template<>
  void GenericAccelerator<NativeBuild>::build(primitive_aggregate_data_s meshes) {
    pimpl->build(meshes);
  }
  template<>
  bool GenericAccelerator<NativeBuild>::hit(const Ray &ray, bvh_hit_data &hit_data) const {
    return pimpl->intersect(ray, hit_data);
  }
  template<>
  void GenericAccelerator<NativeBuild>::cleanup() {
    pimpl->cleanup();
  }

}  // namespace nova::aggregate

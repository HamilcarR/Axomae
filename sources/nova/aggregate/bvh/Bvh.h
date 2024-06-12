#ifndef BVH_H
#define BVH_H
#include "BoundingBox.h"
#include "primitive/nova_primitive.h"

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

  /* Creates a linear bvh with good cache coherence*/
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

  /****************************************************************************************************************************/

  struct bvh_helper_struct {
    float tmin;
    const nova::primitive::NovaPrimitiveInterface *last_prim;
    const bool *stop_traversal;
  };
  struct base_options_bvh : public hit_options<bvh_helper_struct> {};

  class Bvhtl final : public Hitable {
   private:
    Bvht_data bvh;
    const std::vector<std::unique_ptr<primitive::NovaPrimitiveInterface>> *primitives{};

   public:
    Bvhtl() = default;
    explicit Bvhtl(const std::vector<std::unique_ptr<primitive::NovaPrimitiveInterface>> *primitives,
                   BvhtlBuilder::SEGMENTATION segmentation = BvhtlBuilder::SAH);
    ~Bvhtl() override = default;
    Bvhtl(const Bvhtl &other) = default;
    Bvhtl(Bvhtl &&other) noexcept = default;
    Bvhtl &operator=(const Bvhtl &other) = default;
    Bvhtl &operator=(Bvhtl &&other) noexcept = default;
    void build(const std::vector<std::unique_ptr<primitive::NovaPrimitiveInterface>> *primitives,
               BvhtlBuilder::SEGMENTATION segmentation = BvhtlBuilder::SAH);
    bool hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options *user_options) const override;

   private:
    bool traverse(const Ray &r,
                  float tmin,
                  float tmax,
                  hit_data &data,
                  base_options *user_options,
                  const std::vector<std::unique_ptr<primitive::NovaPrimitiveInterface>> *primitives,
                  const Bvht_data &bvh,
                  int32_t node_id) const;

    /* Doesn't use recursion */
    bool linear_traverse(const Ray &r,
                         float tmin,
                         float tmax,
                         hit_data &data,
                         base_options *user_options,
                         const std::vector<std::unique_ptr<primitive::NovaPrimitiveInterface>> *primitives,
                         const Bvht_data &bvh) const;
  };

}  // namespace nova::aggregate
#endif  // BVH_H

#ifndef BVH_H
#define BVH_H
#include "BoundingBox.h"
#include "BvhBuilder.h"
#include "primitive/nova_primitive.h"
namespace nova::aggregate {

  /****************************************************************************************************************************/

  struct bvh_helper_struct {
    float tmin;
    const primitive::NovaPrimitiveInterface *last_prim;
    const bool *is_rendering;  // We need this in case we stop the rendering mid traversal
  };
  struct base_options_bvh : public hit_options<bvh_helper_struct> {};

  class Bvhtl final : public Hitable {
   private:
    Bvht_data bvh;
    const std::vector<primitive::NovaPrimitiveInterface> *primitives{};

   public:
    Bvhtl() = default;
    explicit Bvhtl(const std::vector<primitive::NovaPrimitiveInterface> *primitives,
                   BvhtlBuilder::BUILD_TYPE build_type = BvhtlBuilder::PERFORMANCE,
                   BvhtlBuilder::SEGMENTATION segmentation = BvhtlBuilder::SAH);
    ~Bvhtl() override = default;
    Bvhtl(const Bvhtl &other) = default;
    Bvhtl(Bvhtl &&other) noexcept = default;
    Bvhtl &operator=(const Bvhtl &other) = default;
    Bvhtl &operator=(Bvhtl &&other) noexcept = default;
    void build(const std::vector<primitive::NovaPrimitiveInterface> *primitives,
               BvhtlBuilder::BUILD_TYPE build_type = BvhtlBuilder::PERFORMANCE,
               BvhtlBuilder::SEGMENTATION segmentation = BvhtlBuilder::SAH);
    bool hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options *user_options) const override;

   private:
    bool rec_traverse(const Ray &r,
                      float tmin,
                      float tmax,
                      hit_data &data,
                      base_options *user_options,
                      const std::vector<primitive::NovaPrimitiveInterface> *primitives,
                      const Bvht_data &bvh,
                      int32_t node_id) const;

    bool iter_traverse(const Ray &r,
                       float tmin,
                       float tmax,
                       hit_data &data,
                       base_options *user_options,
                       const std::vector<primitive::NovaPrimitiveInterface> *primitives,
                       const Bvht_data &bvh) const;
  };

}  // namespace nova::aggregate
#endif  // BVH_H

#include "Bvh.h"
#include "acceleration_interface.h"
#include "primitive/PrimitiveInterface.h"
#include "ray/Hitable.h"
namespace nova::aggregate {

  template<>
  class GenericAccelerator<NativeBuild>::Impl {
    Bvhtl scene_bvh;

   public:
    Impl() {}
    ~Impl() {}
    void build(const primitives_view_tn &primitives) { scene_bvh.build(primitives); }
    bool intersect(const Ray &ray, hit_data &hit_data_returned) { return scene_bvh.hit(ray, ray.tnear, ray.tfar, hit_data_returned, nullptr); }
    void cleanup() { scene_bvh = {}; }
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
    pimpl->build(meshes.primitive_list_view);
  }
  template<>
  bool GenericAccelerator<NativeBuild>::hit(const Ray &ray, bvh_hit_data &hit_data) const {
    return pimpl->intersect(ray, hit_data.hit_d);
  }
  template<>
  void GenericAccelerator<NativeBuild>::cleanup() {
    pimpl->cleanup();
  }

}  // namespace nova::aggregate

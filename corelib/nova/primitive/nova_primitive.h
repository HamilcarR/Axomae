#ifndef NOVA_PRIMITIVE_H
#define NOVA_PRIMITIVE_H
#include "PrimitiveInterface.h"
#include "primitive_datastructures.h"
#include <internal/common/axstd/managed_buffer.h>
namespace nova::primitive {

  class PrimitiveStorage {
    axstd::managed_vector<NovaPrimitiveInterface> primitives{};
    axstd::managed_vector<NovaGeoPrimitive> geo_primitives{};

   public:
    CLASS_CM(PrimitiveStorage)

    NovaPrimitiveInterface add(const NovaGeoPrimitive &primitive) { return append(primitive, geo_primitives); }
    void allocGeometric(std::size_t size) { geo_primitives.reserve(size); }
    CstPrimitivesView view() const { return primitives; }
    primitives_view_tn view() { return primitives; }
    void clear() {
      primitives.clear();
      geo_primitives.clear();
    }

   private:
    template<class T>
    NovaPrimitiveInterface append(const T &primitive, axstd::managed_vector<T> &primitive_vector) {
      AX_ASSERT_GE(primitive_vector.capacity(), primitive_vector.size() + 1);
      primitive_vector.push_back(primitive);
      NovaPrimitiveInterface primitive_ptr = &primitive_vector.back();
      primitives.push_back(primitive_ptr);
      return primitive_ptr;
    }
  };

  class PrimitivesResourcesHolder {
    PrimitiveStorage storage;

   public:
    template<class T, class... Args>
    NovaPrimitiveInterface addPrimitive(Args &&...args) {
      static_assert(core::has<T, TYPELIST>::has_type, "Provided type is not a Primitive type.");
      T prim = T(std::forward<Args>(args)...);
      return storage.add(prim);
    }

    void init(const primitive_init_record_s &primitives);
    CstPrimitivesView getPrimitiveView() const;
    primitives_view_tn getPrimitiveView();
    void clear();
  };

}  // namespace nova::primitive
#endif  // NOVA_PRIMITIVE_H

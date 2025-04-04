#include "nova_primitive.h"

namespace nova::primitive {

  void PrimitivesResourcesHolder::init(const primitive_init_record_s &primitives) { storage.allocGeometric(primitives.geometric_primitive_count); }
  CstPrimitivesView PrimitivesResourcesHolder::getPrimitiveView() const { return storage.view(); }
  primitives_view_tn PrimitivesResourcesHolder::getPrimitiveView() { return storage.view(); }
  void PrimitivesResourcesHolder::clear() { storage.clear(); }

}  // namespace nova::primitive
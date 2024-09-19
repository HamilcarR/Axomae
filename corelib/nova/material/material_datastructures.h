#ifndef MATERIAL_DATASTRUCTURES_H
#define MATERIAL_DATASTRUCTURES_H
#include <cstdlib>
namespace nova::material {

  struct material_init_record_s {
    std::size_t dielectrics_size;
    std::size_t conductors_size;
    std::size_t diffuse_size;
  };

}  // namespace nova::material

#endif  // MATERIAL_DATASTRUCTURES_H

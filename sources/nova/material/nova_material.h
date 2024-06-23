#ifndef NOVA_MATERIAL_H
#define NOVA_MATERIAL_H
#include "Axomae_macros.h"
#include "NovaMaterials.h"
#include "utils/macros.h"
#include <memory>
namespace nova::material {

  struct MaterialResourcesHolder {
    std::vector<std::unique_ptr<NovaMaterialInterface>> materials;

    RESOURCES_DEFINE_ADD(material, NovaMaterialInterface, materials)
  };

  RESOURCES_DEFINE_CREATE(NovaMaterialInterface)

}  // namespace nova::material

#endif

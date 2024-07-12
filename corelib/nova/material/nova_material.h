#ifndef NOVA_MATERIAL_H
#define NOVA_MATERIAL_H
#include "NovaMaterials.h"
#include "project_macros.h"
#include "utils/macros.h"
#include <memory>
namespace nova::material {

  struct MaterialResourcesHolder {
    std::vector<std::unique_ptr<NovaMaterialInterface>> materials;

    REGISTER_RESOURCE(material, NovaMaterialInterface, materials)
  };

  RESOURCES_DEFINE_CREATE(NovaMaterialInterface)

}  // namespace nova::material

#endif

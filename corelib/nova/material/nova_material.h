#ifndef NOVA_MATERIAL_H
#define NOVA_MATERIAL_H
#include "NovaMaterials.h"
#include "material_datastructures.h"
#include <internal/common/axstd/managed_buffer.h>
#include <internal/common/axstd/span.h>
#include <internal/memory/MemoryArena.h>
#include <memory>

namespace nova::material {

  class MaterialStorage {
    axstd::managed_vector<NovaMaterialInterface> materials;
    axstd::managed_vector<NovaDielectricMaterial> dielectric_materials;
    axstd::managed_vector<NovaDiffuseMaterial> diffuse_materials;
    axstd::managed_vector<NovaConductorMaterial> conductor_materials;
    axstd::managed_vector<PrincipledMaterial> principled_materials;

   public:
    NovaMaterialInterface add(const NovaDielectricMaterial &dielec) { return append(dielec, dielectric_materials); }
    NovaMaterialInterface add(const NovaConductorMaterial &conductor) { return append(conductor, conductor_materials); }
    NovaMaterialInterface add(const NovaDiffuseMaterial &diffuse) { return append(diffuse, diffuse_materials); }
    NovaMaterialInterface add(const PrincipledMaterial &principled) { return append(principled, principled_materials); }

    void allocDielectrics(std::size_t dielectric_material_number) { dielectric_materials.reserve(dielectric_material_number); }
    void allocDiffuse(std::size_t diffuse_material_number) { diffuse_materials.reserve(diffuse_material_number); }
    void allocConductors(std::size_t conductor_material_number) { conductor_materials.reserve(conductor_material_number); }
    void allocPrincipled(std::size_t principled_material_number) { principled_materials.reserve(principled_material_number); }
    void clear() {
      materials.clear();
      dielectric_materials.clear();
      diffuse_materials.clear();
      conductor_materials.clear();
      principled_materials.clear();
    }

    CstNovaMatIntfView getMaterialView() const { return materials; }

   private:
    template<class T>
    NovaMaterialInterface append(const T &mat, axstd::managed_vector<T> &material_vector) {
      AX_ASSERT_GE(material_vector.capacity(), material_vector.size() + 1);
      material_vector.push_back(mat);
      NovaMaterialInterface material_ptr = &material_vector.back();
      materials.push_back(material_ptr);
      return material_ptr;
    }
  };

  class MaterialResourcesHolder {
    MaterialStorage material_storage;

   public:
    CLASS_CM(MaterialResourcesHolder)

    CstNovaMatIntfView getMaterialView() const { return material_storage.getMaterialView(); }

    void init(const material_init_record_s &init_data) {
      material_storage.allocDielectrics(init_data.dielectrics_size);
      material_storage.allocDiffuse(init_data.diffuse_size);
      material_storage.allocConductors(init_data.conductors_size);
      material_storage.allocPrincipled(init_data.principled_size);
    }

    template<class T, class... Args>
    NovaMaterialInterface addMaterial(Args &&...args) {
      static_assert(core::has<T, TYPELIST>::has_type, "Provided type is not a Material type.");
      const T tmat = T(std::forward<Args>(args)...);
      return material_storage.add(tmat);
    }

    void clear() { material_storage.clear(); }
  };

}  // namespace nova::material

#endif

#ifndef NOVAAPI_H
#define NOVAAPI_H
#include "api_common.h"
#include "api_geometry.h"
#include "api_material.h"
#include "api_scene.h"
#include "api_texture.h"
#include "integrator/integrator_includes.h"
#include "manager/manager_includes.h"
#include "material/material_includes.h"
#include "primitive/primitive_includes.h"
#include "shape/shape_includes.h"
#include "texturing/texturing_includes.h"
#include "utils/utils_includes.h"
#include <internal/geometry/Object3D.h>

namespace nova {
  class EngineInstance {
   public:
    virtual ~EngineInstance() = default;
    virtual ERROR_STATE registerScene(const NvAbstractScene &scene);
  };
  inline std::unique_ptr<EngineInstance> create_engine();
}  // namespace nova

void nv_get_err_str(int error, char log_buffer_r[128]);

#endif

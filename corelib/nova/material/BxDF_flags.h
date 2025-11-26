#ifndef BXDF_FLAGS_H
#define BXDF_FLAGS_H

#include "spectrum/Spectrum.h"
#include <internal/device/gpgpu/device_utils.h>

namespace nova::material {
  enum BXDFFLAGS {
    NONE = 0,
    REFLECTION = 1 << 0,
    TRANSMISSION = 1 << 1,
    DIFFUSE = 1 << 2,
    GLOSSY = 1 << 3,
    SPECULAR = 1 << 4,
    DIFFUSE_REFLECTION = DIFFUSE | REFLECTION,
    DIFFUSE_TRANSMISSION = DIFFUSE | TRANSMISSION,
    GLOSSY_REFLECTION = GLOSSY | REFLECTION,
    GLOSSY_TRANSMISSION = GLOSSY | TRANSMISSION,
    SPECULAR_REFLECTION = SPECULAR | REFLECTION,
    SPECULAR_TRANSMISSION = SPECULAR | TRANSMISSION,
    ALL = REFLECTION | TRANSMISSION | DIFFUSE | GLOSSY | SPECULAR,
  };

  enum TRANSPORT { RADIANCE, IMPORTANCE };

  struct BSDFSample {
    Spectrum f;
    Spectrum eta = 1.f;

    glm::vec3 wi;
    BXDFFLAGS flags;

    float pdf{};
    float costheta{};                  // Always returns abs(costheta(wi)).
    bool pdf_cosine_weighted = false;  // Indicates if the returned f function already is multiplied with costheta_i.
  };

  enum class REFLTRANSFLAG { NONE = 0, TRANSMISSION = 1, REFLECTION = 1 << 1, ALL = TRANSMISSION | REFLECTION };
  ax_device_callable_inlined unsigned operator&(const REFLTRANSFLAG &a, const REFLTRANSFLAG &b) { return (unsigned)a & (unsigned)b; }
  ax_device_callable_inlined unsigned operator|(const REFLTRANSFLAG &a, const REFLTRANSFLAG &b) { return (unsigned)a | (unsigned)b; }

}  // namespace nova::material
#endif

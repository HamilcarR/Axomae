#ifndef BXDF_H
#define BXDF_H

namespace nova::material {

  enum BXDFFLAGS {
    NONE = 0,
    REFLECTION = 1 << 0,
    TRANSMISSION = 1 << 1,
    DIFFUSE = 1 << 2,
    GLOSSY = 1 << 3,
    SPECULAR = 1 << 4,
  };

  const unsigned DIFFUSE_REFLECTION = DIFFUSE | REFLECTION;
  const unsigned DIFFUSE_TRANSMISSION = DIFFUSE | TRANSMISSION;
  const unsigned GLOSSY_REFLECTION = GLOSSY | REFLECTION;
  const unsigned GLOSSY_TRANSMISSION = GLOSSY | TRANSMISSION;
  const unsigned SPECULAR_REFLECTION = SPECULAR | REFLECTION;
  const unsigned SPECULAR_TRANSMISSION = SPECULAR | TRANSMISSION;

}  // namespace nova::material
#endif

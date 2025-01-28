#ifndef DEVICE_HOST_COMMON_H
#define DEVICE_HOST_COMMON_H
#include <glm/glm.hpp>
#include <internal/device/gpgpu/device_transfer_interface.h>
namespace nova::shape::transform {
  struct transform4x4_t {
    glm::mat4 m;
    glm::mat4 inv;                                                               // inverse
    glm::mat4 t;                                                                 // transpose
    glm::mat3 n;                                                                 // normal ( mat3(transpose(invert(m)) )
    bool operator==(const transform4x4_t &other) const { return m == other.m; }  // no need to compare the others , waste of cycles.
    static constexpr std::size_t padding() { return 57; }                        // how many elements in the record
  };

  struct transform3x3_t {
    glm::mat3 m;
    glm::mat3 inv;
    glm::mat3 t;
    bool operator==(const transform3x3_t &other) const { return m == other.m; }
    static constexpr std::size_t padding() { return 27; }
  };

}  // namespace nova::shape::transform
#endif

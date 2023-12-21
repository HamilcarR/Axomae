#ifndef GLSL_H
#define GLSL_H

#include <cstdlib>
#include <string>

namespace shader_utils {
  namespace glsl_utils {
    std::string bbox_frag();
    std::string bbox_vert();
    std::string brdf_lut_frag();
    std::string cubemap_frag();
    std::string cubemap_vert();
    std::string envmap_bake_frag();
    std::string envmap_bake_vert();
    std::string envmap_prefilter_frag();
    std::string irradiance_baker_frag();
    std::string pbr_frag();
    std::string pbr_vert();
    std::string phong_frag();
    std::string phong_vert();
    std::string screen_fbo_frag();
    std::string screen_fbo_vert();
    std::string simple_frag();
    std::string simple_vert();

  };  // namespace glsl_utils

};  // namespace shader_utils
#endif
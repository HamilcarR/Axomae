#include "glsl.h"
#include "fragment.h"
#include "vertex.h"

static std::string construct_string(const unsigned char array[], unsigned size) { return std::string(array, array + size); }
namespace shader_utils {
  namespace glsl_utils {
    std::string bbox_frag() { return construct_string(glsl_bbox_frag, glsl_bbox_frag_len); }
    std::string bbox_vert() { return construct_string(glsl_bbox_vert, glsl_bbox_vert_len); }
    std::string brdf_lut_frag() { return construct_string(glsl_brdf_lookup_table_baker_frag, glsl_brdf_lookup_table_baker_frag_len); }
    std::string cubemap_frag() { return construct_string(glsl_cubemap_frag, glsl_cubemap_frag_len); }
    std::string cubemap_vert() { return construct_string(glsl_cubemap_vert, glsl_cubemap_vert_len); }
    std::string envmap_bake_frag() { return construct_string(glsl_envmap_bake_frag, glsl_envmap_bake_frag_len); }
    std::string envmap_bake_vert() { return construct_string(glsl_envmap_bake_vert, glsl_envmap_bake_vert_len); }
    std::string envmap_prefilter_frag() { return construct_string(glsl_envmap_prefilter_frag, glsl_envmap_prefilter_frag_len); }
    std::string irradiance_baker_frag() { return construct_string(glsl_irradiance_baker_frag, glsl_irradiance_baker_frag_len); }
    std::string pbr_frag() { return construct_string(glsl_pbr_frag, glsl_pbr_frag_len); }
    std::string pbr_vert() { return construct_string(glsl_pbr_vert, glsl_pbr_vert_len); }
    std::string phong_frag() { return construct_string(glsl_phong_frag, glsl_phong_frag_len); }
    std::string phong_vert() { return construct_string(glsl_phong_vert, glsl_phong_vert_len); }
    std::string screen_fbo_frag() { return construct_string(glsl_screen_fbo_frag, glsl_screen_fbo_frag_len); }
    std::string screen_fbo_vert() { return construct_string(glsl_screen_fbo_vert, glsl_screen_fbo_vert_len); }
    std::string simple_frag() { return construct_string(glsl_simple_frag, glsl_simple_frag_len); }
    std::string simple_vert() { return construct_string(glsl_simple_vert, glsl_simple_vert_len); }

  };  // namespace glsl_utils
};    // namespace shader_utils
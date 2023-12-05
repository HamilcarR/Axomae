#ifndef UNIFORMNAMES_H
#define UNIFORMNAMES_H

/*Shader matrix uniform names*/
constexpr const char uniform_name_matrix_model[] = "MAT_MODEL";
constexpr const char uniform_name_matrix_view[] = "MAT_VIEW";
constexpr const char uniform_name_matrix_modelview[] = "MAT_MODELVIEW";
constexpr const char uniform_name_matrix_projection[] = "MAT_PROJECTION";
constexpr const char uniform_name_matrix_view_projection[] = "MAT_VP";
constexpr const char uniform_name_matrix_model_view_projection[] = "MAT_MVP";
constexpr const char uniform_name_matrix_normal[] = "MAT_NORMAL";
constexpr const char uniform_name_matrix_inverse_model[] = "MAT_INV_MODEL";
constexpr const char uniform_name_matrix_inverse_modelview[] = "MAT_INV_MODELVIEW";
constexpr const char uniform_name_cubemap_matrix_normal[] = "MAT_CUBEMAP_NORMAL";

/**************Shader variables uniforms*********************/

/*Camera*/
constexpr const char uniform_name_vec3_camera_position[] = "camera_position";

/*Materials*/
constexpr const char uniform_name_str_material_struct_name[] = "material";
constexpr const char uniform_name_vec2_material_refractive_index[] = "refractive_index";
constexpr const char uniform_name_float_material_dielectric_factor[] = "dielectric_factor";
constexpr const char uniform_name_float_material_roughness_factor[] = "roughness_factor";
constexpr const char uniform_name_float_material_transmission_factor[] = "transmission_factor";
constexpr const char uniform_name_float_material_emissive_factor[] = "emissive_factor";
constexpr const char uniform_name_float_material_shininess_factor[] = "shininess";
constexpr const char uniform_name_float_material_transparency_factor[] = "alpha_factor";

/*Lighting variables*/
constexpr const char uniform_name_vec3_lighting_position[] = "position";
constexpr const char uniform_name_vec3_lighting_spot_direction[] = "direction";
constexpr const char uniform_name_vec3_lighting_specular_color[] = "specularColor";
constexpr const char uniform_name_vec3_lighting_ambient_color[] = "ambientColor";
constexpr const char uniform_name_vec3_lighting_diffuse_color[] = "diffuseColor";
constexpr const char uniform_name_float_lighting_attenuation_constant[] = "constantAttenuation";
constexpr const char uniform_name_float_lighting_attenuation_linear[] = "linearAttenuation";
constexpr const char uniform_name_float_lighting_attenuation_quadratic[] = "quadraticAttenuation";
constexpr const char uniform_name_float_lighting_intensity[] = "intensity";
constexpr const char uniform_name_float_lighting_spot_theta[] = "theta";
constexpr const char uniform_name_float_lighting_spot_falloff[] = "falloff";

/*Light structures*/
constexpr const char uniform_name_str_lighting_directional_struct_name[] = "directional_light_struct";
constexpr const char uniform_name_str_lighting_point_struct_name[] = "point_light_struct";
constexpr const char uniform_name_str_lighting_spot_struct_name[] = "spot_light_struct";

/*Light arrays*/
constexpr const char uniform_name_uint_lighting_directional_number_name[] = "directional_light_number";
constexpr const char uniform_name_uint_lighting_point_number_name[] = "point_light_number";
constexpr const char uniform_name_uint_lighting_spot_number_name[] = "spot_light_number";

/*HDR*/
constexpr const char uniform_name_float_gamma_name[] = "gamma";
constexpr const char uniform_name_float_exposure_name[] = "exposure";

/*Post-Processing*/
constexpr const char uniform_name_bool_sharpen[] = "uniform_sharpen";
constexpr const char uniform_name_bool_edge[] = "uniform_edge";
constexpr const char uniform_name_bool_blurr[] = "uniform_blurr";

/*PBR*/
constexpr const char uniform_name_float_cubemap_prefilter_roughness[] = "roughness";
constexpr const char uniform_name_uint_prefilter_shader_envmap_resolution[] = "envmap_resolution";
constexpr const char uniform_name_uint_prefilter_shader_samples_count[] = "samples_count";

#endif
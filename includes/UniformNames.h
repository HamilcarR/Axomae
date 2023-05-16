#ifndef UNIFORMNAMES_H
#define UNIFORMNAMES_H

/*Shader matrix uniform names*/
constexpr const char uniform_name_matrix_model[] = "MAT_MODEL"; 
constexpr const char uniform_name_matrix_view[] = "MAT_VIEW" ;
constexpr const char uniform_name_matrix_modelview[] = "MAT_MODELVIEW";  
constexpr const char uniform_name_matrix_projection[] = "MAT_PROJECTION"; 
constexpr const char uniform_name_matrix_view_projection[] = "MAT_VP"; 
constexpr const char uniform_name_matrix_model_view_projection[] = "MAT_MVP" ; 
constexpr const char uniform_name_matrix_normal[] = "MAT_NORMAL" ;
constexpr const char uniform_name_matrix_inverse_model[] = "MAT_INV_MODEL" ;
constexpr const char uniform_name_matrix_inverse_modelview[] = "MAT_INV_MODELVIEW" ;  

/*Shader variables uniforms*/
constexpr const char uniform_name_vec3_camera_position[] = "camera_position" ; 
constexpr const char uniform_name_vec2_material_refractive_index[] = "refractive_index";
constexpr const char uniform_name_float_material_dielectric_factor[] = "dielectric_factor" ; 
constexpr const char uniform_name_float_material_roughness_factor[] = "roughness_factor" ; 
constexpr const char uniform_name_float_material_transmission_factor[] = "transmission_factor" ; 
constexpr const char uniform_name_float_material_emissive_factor[] = "emissive_factor" ; 

#endif
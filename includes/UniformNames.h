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
constexpr const char uniform_name_vector_camera_position[] = "camera_position" ; 


#endif
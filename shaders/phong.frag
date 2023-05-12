#version 460 core

/* Interpolated data from vertex shader */
in vec4 vertex_fragment_colors ; 
in vec2 vertex_fragment_uv ;
in vec3 vertex_fragment_normals; 
in vec3 vertex_fragment_positions ; 
in vec3 vertex_fragment_fragment_position ; 
in vec3 vertex_fragment_light_position; 
in vec3 vertex_fragment_camera_position; 
/*****************************************/

/* Flat data from vertex shader */

/*****************************************/


/* Uniforms */
uniform mat4 MAT_MODEL;
uniform mat4 MAT_MODELVIEW ; 
uniform mat4 MAT_INV_MODEL ;
uniform mat4 MAT_INV_MODELVIEW ;  
uniform mat3 MAT_NORMAL ; 
/*****************************************/

/* Samplers and textures */
layout(binding=0) uniform sampler2D diffuse ; 
layout(binding=1) uniform sampler2D normal ; 
layout(binding=2) uniform sampler2D metallic ; 
layout(binding=3) uniform sampler2D roughness ; 
layout(binding=4) uniform sampler2D ambiantocclusion ;
layout(binding=5) uniform sampler2D specular;
layout(binding=6) uniform sampler2D emissive; 
layout(binding=7) uniform samplerCube cubemap; 
layout(binding=8) uniform sampler2D generic ;
/******************************************/

/* Shader Output*/
out vec4 fragment ;
/******************************************/

/*Constants*/
const float specular_intensity = 1.8f; 
const float shininess = 10; 
const float n1 = 1.f ; 
const float n2 = 2.42f ;  
const vec3 camera_position = vec3(0.f);
/******************************************/

vec3 getViewDirection(){
    return normalize(camera_position - vertex_fragment_fragment_position); 
}

vec3 getLightDirection(){
    return normalize(vertex_fragment_light_position - vertex_fragment_fragment_position); 
}

float computeDiffuseLight(){
    vec3 n = normalize(vertex_fragment_normals);
    float diffuse_light = max(dot(getLightDirection() , n) , 0.f); 
    return diffuse_light ; 
}

float computeSpecularLight(){
    vec3 light_direction = getLightDirection(); 
    vec3 view_direction = getViewDirection(); 
    vec3 normal_vector = vertex_fragment_normals; 
    vec3 specular_reflection = reflect(-light_direction , normal_vector); 
    float angle_reflection_view_direction = dot(view_direction , specular_reflection);     
    float specular = pow(max(angle_reflection_view_direction , 0.f) , shininess) ; 
    float computed_total_specular = specular * specular_intensity ;   
    return computed_total_specular ; 
}

vec4 computeReflectionCubeMap(){
    vec3 view_direction = normalize(vertex_fragment_fragment_position - camera_position); 
    vec3 normal_vector = normalize(vertex_fragment_normals); 
    vec3 cubemap_sample_vector = vec3(inverse(MAT_NORMAL) * reflect(view_direction , normal_vector)); 
    vec4 sampled_value = texture(cubemap , cubemap_sample_vector , 1.f); 
    return sampled_value; 
}

vec4 computeRefractionCubeMap(){
    float refractive_index_ratio = n1 / n2 ; 
    vec3 view_direction = normalize(vertex_fragment_fragment_position - camera_position) ; 
    vec3 normal_vector = normalize(vertex_fragment_normals); 
    vec3 cubemap_sample_vector = vec3(inverse(MAT_NORMAL) * refract(view_direction , normal_vector , refractive_index_ratio)); 
    vec4 sampled_value = texture(cubemap , cubemap_sample_vector); 
    return sampled_value; 

}
vec2 computeFresnelCoefficients(){
    float refractive_index_ratio = n1 / n2 ; 
    vec3 incident_vector = normalize(vertex_fragment_fragment_position - camera_position) ; 
    vec3 normal_vector = normalize(vertex_fragment_normals); 
    float cos_theta1 =  dot(-incident_vector , normal_vector);
    float sin_theta1 = sqrt(1 - cos_theta1 * cos_theta1);  
    float sin_theta2 = sin_theta1 * refractive_index_ratio ;  
    float cos_theta2 = sqrt(1 - sin_theta2 * sin_theta2) ; 
    float Fr1_ratio = (n1 * cos_theta1 - n2 * cos_theta2) / (n1 * cos_theta1 + n2 * cos_theta2) ;
    float Fr2_ratio = (n1 * cos_theta2 - n2 * cos_theta1) / (n1 * cos_theta2 + n2 * cos_theta1) ; 
    float FR1 = Fr1_ratio * Fr1_ratio ; 
    float FR2 = Fr2_ratio * Fr2_ratio ;
    float FR = 0.5 * (FR1 + FR2); 
    return vec2(FR , 1 - FR);  
}

void main(){	
    vec2 fresnel = computeFresnelCoefficients() ; 
    fragment = (fresnel.y * computeRefractionCubeMap() + fresnel.x * computeReflectionCubeMap()) ; //* texture(diffuse  , vertex_fragment_uv) ; //* (computeDiffuseLight() + computeSpecularLight()) ; 
}
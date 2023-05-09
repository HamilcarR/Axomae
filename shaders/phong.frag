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

/* Uniforms */

uniform mat4 MAT_MODEL; 
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
    vec3 cubemap_sample_vector = reflect(view_direction , normal_vector); 
    vec4 sampled_value = texture(cubemap , cubemap_sample_vector); 
    return sampled_value; 
}

vec4 computeRefractionCubeMap(){
    float refractive_index_ratio = 1.f / 1.52f ; 
    vec3 view_direction = normalize(vertex_fragment_fragment_position - camera_position) ; 
    vec3 normal_vector = normalize(vertex_fragment_normals); 
    vec3 cubemap_sample_vector = refract(view_direction , normal_vector , refractive_index_ratio); 
    vec4 sampled_value = texture(cubemap , cubemap_sample_vector); 
    return sampled_value; 

}

void main(){	
    fragment = computeReflectionCubeMap() * texture(diffuse  , vertex_fragment_uv) * (computeDiffuseLight() + computeSpecularLight()) ; 
}
#version 460 core
#define MAX_LIGHTS 20

/* Interpolated data from vertex shader */
in vec4 vertex_fragment_colors ; 
in vec2 vertex_fragment_uv ;
in vec3 vertex_fragment_normals; 
in vec3 vertex_fragment_positions ; 
in vec3 vertex_fragment_fragment_position ; 
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
uniform vec2 refractive_index ;
uniform unsigned int directional_light_number;  

/*Uniforms structs*/

struct DIRECTIONAL_LIGHT_STRUCT {
    vec3 position ; 
    vec3 specularColor ; 
    vec3 ambientColor ; 
    vec3 diffuseColor ; 
    float intensity ; 
};
uniform DIRECTIONAL_LIGHT_STRUCT directional_light_struct[MAX_LIGHTS] ; 
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
const float shininess = 200; 
const vec3 camera_position = vec3(0.f);
const float ambient_factor = 0.2f ;
/******************************************/

vec3 getViewDirection(){
    return normalize(camera_position - vertex_fragment_fragment_position); 
}

vec3 getLightDirection(unsigned int i){
    return normalize(directional_light_struct[i].position - vertex_fragment_fragment_position); 
}

vec3 computeDiffuseLight(){
    vec3 n = normalize(vertex_fragment_normals);
    int i = 0 ; 
    vec3 total_diffuse = vec3(0.f); 
    for(i = 0 ; i < directional_light_number ; i++){
        vec3 dir = getLightDirection(i); 
        float diffuse_light = max(dot(dir , n) , 0.f) * directional_light_struct[i].intensity ; 
        total_diffuse += diffuse_light * directional_light_struct[i].diffuseColor ;
    }
    return total_diffuse;  
}

vec3 computeSpecularLight(){
    vec3 view_direction = getViewDirection(); 
    vec3 normal_vector = normalize(vertex_fragment_normals); 
    vec3 computed_total_specular = vec3(0.f); 
    int i = 0 ; 
    for(i = 0 ; i < directional_light_number ; i++){
        vec3 light_direction = getLightDirection(i); 
        vec3 specular_reflection = reflect(-light_direction , normal_vector); 
        float angle_reflection_view_direction = dot(view_direction , specular_reflection);     
        float specular = pow(max(angle_reflection_view_direction , 0.f) , shininess) * specular_intensity;  
        computed_total_specular += specular * directional_light_struct[i].specularColor * directional_light_struct[i].intensity;   
    }
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
    float refractive_index_ratio = refractive_index.x / refractive_index.y ; 
    vec3 view_direction = normalize(vertex_fragment_fragment_position - camera_position) ; 
    vec3 normal_vector = normalize(vertex_fragment_normals); 
    vec3 cubemap_sample_vector = vec3(inverse(MAT_NORMAL) * refract(view_direction , normal_vector , refractive_index_ratio)); 
    vec4 sampled_value = texture(cubemap , cubemap_sample_vector); 
    return sampled_value; 
}

vec2 computeFresnelCoefficients(){
    float n1 = refractive_index.x ; 
    float n2 = refractive_index.y ; 
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
    vec4 R = computeReflectionCubeMap() * fresnel.x ; 
    vec4 Rf = computeRefractionCubeMap() * fresnel.y ; 
    vec4 C = texture(diffuse , vertex_fragment_uv); 
    fragment = vec4(computeDiffuseLight() + computeSpecularLight() , 1.f) * C * (R + Rf); 
}
#version 460 core
#define MAX_LIGHTS 10

/* Interpolated data from vertex shader */
in vec4 vertex_fragment_colors ; 
in vec2 vertex_fragment_uv ;
in vec3 vertex_fragment_normals; 
in vec3 vertex_fragment_positions ; 
in vec3 vertex_fragment_fragment_position ; 
in vec3 vertex_fragment_camera_position; 
/*****************************************/

/* Flat data from vertex shader */
in mat3 MAT_TBN;  
/*****************************************/


/* Uniforms */
uniform mat4 MAT_MODEL;
uniform mat4 MAT_MODELVIEW ; 
uniform mat4 MAT_INV_MODEL ;
uniform mat4 MAT_INV_MODELVIEW ;  
uniform mat3 MAT_NORMAL ;
uniform vec2 refractive_index ;
uniform uint directional_light_number;  
uniform uint point_light_number;
uniform uint spot_light_number;


struct MATERIAL {
    vec2 refractive_index;
    float dielectric_factor; 
    float roughness_factor; 
    float transmission_factor; 
    float emissive_factor;  
    float shininess; 
    float alpha_factor;  
};
uniform MATERIAL material ; 

/* Point lights */
struct POINT_LIGHT_STRUCT{
    vec3 position ; 
    vec3 specularColor; 
    vec3 ambientColor; 
    vec3 diffuseColor; 
    float intensity; 
    float constantAttenuation;
    float linearAttenuation; 
    float quadraticAttenuation;
};
uniform POINT_LIGHT_STRUCT point_light_struct[MAX_LIGHTS];  

/*Directional lights*/
struct DIRECTIONAL_LIGHT_STRUCT {
    vec3 position ; 
    vec3 specularColor ; 
    vec3 ambientColor ; 
    vec3 diffuseColor ; 
    float intensity ; 
};
uniform DIRECTIONAL_LIGHT_STRUCT directional_light_struct[MAX_LIGHTS];

/*Spot lights*/
struct SPOT_LIGHT_STRUCT{
    vec3 position ; 
    vec3 direction ; 
    vec3 specularColor ; 
    vec3 ambientColor ; 
    vec3 diffuseColor ; 
    float intensity ;
    float theta ;   
    float falloff ; 
};
uniform SPOT_LIGHT_STRUCT spot_light_struct[MAX_LIGHTS]; 

/*****************************************/
/* Samplers and textures */
layout(binding=2) uniform sampler2D diffuse_map ; // Albedo 
layout(binding=3) uniform sampler2D normal_map ; 
layout(binding=4) uniform sampler2D metallic_map ; 
layout(binding=5) uniform sampler2D roughness_map ; 
layout(binding=6) uniform sampler2D ambiantocclusion_map ;
layout(binding=7) uniform sampler2D specular_map;
layout(binding=8) uniform sampler2D emissive_map;
layout(binding=9) uniform sampler2D opacity_map ;  
layout(binding=10) uniform samplerCube cubemap; 
/******************************************/


/*Constants*/
const vec3 camera_position = vec3(0.f) ;
const float ambient_factor = 0.2f ;
/******************************************/
/*Structures*/
struct LIGHT_COMPONENTS{
    vec3 ambient ; 
    vec3 diffuse ; 
    vec3 specular ; 
}; 

/******************************************/
/* Shader Output*/
out vec4 fragment ;
/******************************************/

vec3 getViewDirection(){
    return normalize(camera_position - vertex_fragment_fragment_position); 
}

/* Texture sampling functions*/
/**************************************************************************************************************/
vec3 getSurfaceNormal(){
    vec3 N = normalize(MAT_TBN * (texture(normal_map , vertex_fragment_uv).rgb * 2.f - 1.f));     
  //  N = normalize(vertex_fragment_normals); 
    return N ; 
}

vec4 computeEmissiveValue(){
    return texture(emissive_map , vertex_fragment_uv); 
}

vec4 computeMetallicValue(){
    return texture(metallic_map , vertex_fragment_uv); 
}

vec4 computeDiffuseValue(){
    return texture(diffuse_map , vertex_fragment_uv);
}

/**************************************************************************************************************/
float computePointLightAttenuation(uint point_light_index){
    float dist = length(vertex_fragment_fragment_position - point_light_struct[point_light_index].position);  
    float constant_atten = point_light_struct[point_light_index].constantAttenuation ; 
    float linear_atten = point_light_struct[point_light_index].linearAttenuation ; 
    float quadratic_atten = point_light_struct[point_light_index].quadraticAttenuation ;
    return 1.0f/(constant_atten + linear_atten * dist + quadratic_atten * dist * dist) ;  
}

/**************************************************************************************************************/
vec3 computeDiffusePointLight(vec3 surface_normal , vec3 light_dir , uint i){
    float diffuse_light = max(dot(light_dir , surface_normal) , 0.f) * point_light_struct[i].intensity ; 
    return diffuse_light * point_light_struct[i].diffuseColor ;    
}

vec3 computeDiffuseDirectionalLight(vec3 surface_normal ,vec3 light_dir , uint i){
    float diffuse_light = max(dot(light_dir , surface_normal) , 0.f) * directional_light_struct[i].intensity ; 
    return diffuse_light * directional_light_struct[i].diffuseColor ;    
}

/*fragment_light_drection is the direction from the fragment to the light*/
vec3 computeDiffuseSpotLight(vec3 surface_normal , vec3 fragment_light_direction , uint i){
    vec3 spot_light_direction = normalize(spot_light_struct[i].direction) ; //general direction of the spotlight
    float angle =  dot(-fragment_light_direction , spot_light_direction);
    angle = acos(angle); 
    if(angle < spot_light_struct[i].theta){
        float diffuse_light = max(dot(fragment_light_direction , surface_normal) , 0.f) * spot_light_struct[i].intensity ; 
        return diffuse_light * spot_light_struct[i].diffuseColor ;    
    }
    return vec3(0.f); 
}

/**************************************************************************************************************/
vec3 computeAmbientSpotLight(uint i){
    return spot_light_struct[i].ambientColor ; 
}

vec3 computeAmbientDirectionalLight(uint i){
    return directional_light_struct[i].ambientColor ; 
}

vec3 computeAmbientPointLight(uint i){
    return point_light_struct[i].ambientColor ;
}

/**************************************************************************************************************/
vec3 computeSpecularPointLight(vec3 surface_normal , vec3 light_direction , vec3 view_direction , uint i){
    vec3 specular_reflection = reflect(-light_direction , surface_normal); 
    float angle_reflection_view_direction = dot(specular_reflection , view_direction);     
    float specular = pow(max(angle_reflection_view_direction , 0.f) , material.shininess) ; 
    return specular * point_light_struct[i].specularColor * point_light_struct[i].intensity;   
}

vec3 computeSpecularDirectionalLight(vec3 surface_normal ,vec3 light_direction , vec3 view_direction , uint i){
    vec3 specular_reflection = reflect(-light_direction , surface_normal); 
    float angle_reflection_view_direction = dot(specular_reflection , view_direction); 
    float specular = pow(max(angle_reflection_view_direction , 0.f) , material.shininess) ; 
    return specular * directional_light_struct[i].specularColor * directional_light_struct[i].intensity;   
}

vec3 computeSpecularSpotLight(vec3 surface_normal , vec3 light_direction , vec3 view_direction , uint i){ 
    vec3 spot_light_direction = normalize(spot_light_struct[i].direction) ;
    float angle =  dot(-light_direction , spot_light_direction);
    angle = acos(angle); 
    if(angle < spot_light_struct[i].theta){
        vec3 specular_reflection = reflect(-light_direction , surface_normal); 
        float angle_reflection_view_direction = dot(specular_reflection , view_direction); 
        float specular = pow(max(angle_reflection_view_direction , 0.f) , material.shininess); 
        return specular * spot_light_struct[i].specularColor * spot_light_struct[i].intensity;
    }
    return vec3(0.f); 
}

/**************************************************************************************************************/
/* Returns (ambiant , diffuse , specular) as result*/
LIGHT_COMPONENTS computeDirectionalLightsContrib(){
    LIGHT_COMPONENTS light; 
    uint i = 0 ; 
    vec3 n = getSurfaceNormal();
    vec3 view_direction = getViewDirection(); 
    for(i = 0 ; i < directional_light_number ; i++){
        vec3 light_direction = normalize(directional_light_struct[i].position); 
        light.diffuse += computeDiffuseDirectionalLight(n , light_direction , i); 
        light.ambient += computeAmbientDirectionalLight(i); 
        light.specular += computeSpecularDirectionalLight(n , light_direction , view_direction, i); 
    }
    return light ; 
}

/**************************************************************************************************************/
LIGHT_COMPONENTS computePointLightsContrib(){
    LIGHT_COMPONENTS light;  
    vec3 n = getSurfaceNormal();
    vec3 view_direction = getViewDirection(); 
    uint i = 0 ; 
    for(i = 0 ; i < point_light_number ; i++){
        float attenuation = computePointLightAttenuation(i);
        vec3 light_direction = normalize(point_light_struct[i].position - vertex_fragment_fragment_position); 
        light.diffuse += computeDiffusePointLight(n , light_direction , i) * attenuation ; 
        light.ambient += computeAmbientPointLight(i) * attenuation ; 
        light.specular += computeSpecularPointLight(n , light_direction , view_direction , i) * attenuation; 
    }
    return light ; 
}

/**************************************************************************************************************/
LIGHT_COMPONENTS computeSpotLightsContrib(){
    LIGHT_COMPONENTS light ; 
    vec3 n = getSurfaceNormal();
    vec3 view_direction = getViewDirection(); 
    int i = 0 ; 
    for(i = 0 ; i < spot_light_number ; i++){
        vec3 light_direction = normalize(spot_light_struct[i].position - vertex_fragment_fragment_position); 
        light.diffuse += computeDiffuseSpotLight(n , light_direction , i); 
        light.ambient += computeAmbientSpotLight(i); 
        light.specular += computeSpecularSpotLight(n , light_direction , view_direction , i);     
    }
    return light; 
}

/**************************************************************************************************************/
vec4 computeReflectionCubeMap(float fresnel){
    vec3 view_direction = normalize(vertex_fragment_fragment_position - camera_position); 
    vec3 normal_vector = getSurfaceNormal(); 
    vec3 cubemap_sample_vector = vec3(inverse(MAT_NORMAL) * reflect(view_direction , normal_vector)); 
    vec4 sampled_value = texture(cubemap , cubemap_sample_vector); 
    return vec4(sampled_value.rgb * fresnel, 1.f); 
}

/**************************************************************************************************************/
vec4 computeRefractionCubeMap(float fresnel){
    float refractive_index_ratio = material.refractive_index.x / material.refractive_index.y ; 
    vec3 view_direction = normalize(vertex_fragment_fragment_position - camera_position) ; 
    vec3 normal_vector = getSurfaceNormal(); 
    vec3 cubemap_sample_vector = vec3(inverse(MAT_NORMAL) * refract(view_direction , normal_vector , refractive_index_ratio)); 
    vec4 sampled_value = texture(cubemap , cubemap_sample_vector); 
    return vec4(sampled_value.rgb * fresnel, 1.f); 
}

/**************************************************************************************************************/
vec2 computeFresnelCoefficients(){
    float n1 = material.refractive_index.x ; 
    float n2 = material.refractive_index.y ; 
    float refractive_index_ratio = n1 / n2  ; 
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

/**************************************************************************************************************/
void main(){	
    vec4 final_computed_fragment ; 
    vec2 fresnel = computeFresnelCoefficients() ; 
    vec4 R = computeReflectionCubeMap(fresnel.x) ; 
    vec4 Rf = computeRefractionCubeMap(fresnel.y) ; 
    vec4 metallic = computeMetallicValue(); 
    vec4 E = computeEmissiveValue() ; 
    vec4 C = computeDiffuseValue() ;
    LIGHT_COMPONENTS directional = computeDirectionalLightsContrib() ; 
    LIGHT_COMPONENTS point = computePointLightsContrib();
    LIGHT_COMPONENTS spot = computeSpotLightsContrib();  
    vec3 ambient = directional.ambient + point.ambient + spot.ambient ; 
    vec3 diffuse = directional.diffuse + point.diffuse + spot.diffuse ; 
    vec3 specular = directional.specular + point.specular + spot.specular ; 
    int refract_active = 0 ; 
    if(material.alpha_factor < 1.f)
        refract_active = 1 ;
    if(E.rgb != vec3(0))
        final_computed_fragment = C + (Rf * 0.3f * refract_active + R * refract_active) + E * material.emissive_factor ; 
    else
        final_computed_fragment = vec4(specular + diffuse + ambient * 0.4f, 1.f) * C + (Rf * 0.3f * refract_active + R * refract_active);
    float alpha = C.a < material.alpha_factor ? C.a : material.alpha_factor; 
    final_computed_fragment.a = alpha; 
    fragment = final_computed_fragment; 
}
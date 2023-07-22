#version 460 core
#define MAX_LIGHTS 10
#define PI 3.1415926538
#define EPSILON 0.0001



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
/******************************************/
/*Structures*/
struct LIGHT_COMPONENTS{
    vec3 radiance;  
}; 

/******************************************/
/* Shader Output*/
out vec4 fragment ;
/******************************************/

vec3 getViewDirection(){ // -V
    return normalize(vertex_fragment_fragment_position - camera_position); 
}

vec3 computeHalfVect(vec3 light_pos){
    vec3 view_dir = normalize(-getViewDirection()); 
    vec3 light_dir = normalize(light_pos - vertex_fragment_fragment_position); 
    return normalize(light_dir + view_dir); 
}

/* Texture sampling functions*/
/**************************************************************************************************************/
vec3 getSurfaceNormal(){
    vec3 N = normalize(MAT_TBN * (texture(normal_map , vertex_fragment_uv).rgb * 2.f - 1.f));     
    return N ; 
}
vec4 computeAmbiantOcclusionValue(){
    return texture(ambiantocclusion_map , vertex_fragment_uv); 
}

vec4 computeRoughnessValue(){
    return texture(roughness_map , vertex_fragment_uv); 
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

/* Add spot light attenuation */
float computeSpotLightAttenuation(uint spot_light_index){
    float dist = length(vertex_fragment_fragment_position - spot_light_struct[spot_light_index].position);  
    return 1.0f/(dist * dist) ; 
}
/**************************************************************************************************************/
vec3 computeRadiancePointLight(vec3 surface_normal , uint i){
    float attenuation = computePointLightAttenuation(i); 
    vec3 light_dir = normalize(point_light_struct[i].position - vertex_fragment_fragment_position); 
    float radiance = max(dot(light_dir , surface_normal) , 0.f) * point_light_struct[i].intensity ; 
    return radiance * point_light_struct[i].diffuseColor * attenuation ;    
}

vec3 computeRadianceDirectionalLight(vec3 surface_normal ,vec3 light_dir , uint i){
    float radiance = max(dot(light_dir , surface_normal) , 0.f) * directional_light_struct[i].intensity * 200.f; 
    return radiance * directional_light_struct[i].diffuseColor ;    
}

/*fragment_light_drection is the direction from the fragment to the light*/
vec3 computeRadianceSpotLight(vec3 surface_normal , vec3 fragment_light_direction , uint i){
    vec3 spot_light_direction = normalize(spot_light_struct[i].direction) ; //general direction of the spotlight
    float angle =  dot(-fragment_light_direction , spot_light_direction);
    angle = acos(angle); 
    if(angle < spot_light_struct[i].theta){
        float radiance = max(dot(fragment_light_direction , surface_normal) , 0.f) * spot_light_struct[i].intensity ; 
        return radiance * spot_light_struct[i].diffuseColor  * computeSpotLightAttenuation(i);    
    }
    return vec3(0.f); 
}

/**************************************************************************************************************/
/* DFG FUNCTIONS*/


/**************************** FRESNEL EQUATIONS ***************************************************/
/* Calculates F0 */
vec3 fresnelConstant(){
    float n1 = 1.f ; // We consider incident rays as coming from vacuum 
    float n2 = material.refractive_index.y ; 
    float cste = (n1 - n2) / (n1 + n2); 
    cste = cste * cste ;
    return vec3(cste);  
}

vec3 fresnelSchlickGGX(float cos_theta , float metallic , vec3 albedo){
    vec3 F0 = vec3(0.04) ; //fresnelConstant();
    F0 = mix(F0 , albedo , metallic); 
    return F0 + (1 - F0) * pow(clamp(1.f - cos_theta , 0.f , 1.f) , 5.f) ; 
}

vec3 computeFresnel(vec3 h , vec3 v , float metallic , vec3 albedo){
    return fresnelSchlickGGX(max(dot(h , v) , 0.f) , metallic , albedo); 
}

/**************************** NORMAL DISTRIBUTION FUNCTION ***************************************************/

float distributionGGX(vec3 h , vec3 n , float roughness){
    float aa = roughness * roughness ; 
    float dot_n_h  = max(dot(n , h) , 0.f); 
    float dot_n_h_2 = dot_n_h * dot_n_h ; 
    float divid = dot_n_h_2 * (aa - 1) + 1 ; 
    divid *= divid ; 
    divid *= PI ; 
    return aa/divid ; 
}

/**************************** GEOMETRY FUNCTION ***************************************************/

float geometrySchlickGGX(vec3 d1 , vec3 d2 , float k){
    float d1_dot_d2 = max(dot(d1 , d2) , 0.f) ; 
    float nom = d1_dot_d2 ; 
    float denom = nom * (1 - k) + k ;
    return nom/denom ;  
}

float geometrySmith(vec3 light_direction , vec3 surface_normal , vec3 eye_pos, float roughness , bool IBL){
    float k = IBL ? (roughness * roughness) / 2 : ((roughness + 1) * (roughness + 1) / 8) ; //Either direct lighting or IBL 
    return geometrySchlickGGX(surface_normal , eye_pos , k) * geometrySchlickGGX(surface_normal , light_direction , k); 
}



/**************************************************************************************************************/
vec3 computeBRDF(vec3 light_pos , vec3 n , vec3 v , float roughness , float metallic , vec3 albedo , bool IBL){
    vec3 h = computeHalfVect(light_pos);  
    vec3 l = normalize(light_pos - vertex_fragment_fragment_position); 
    float D = distributionGGX(h , n , roughness * roughness);
    float G = geometrySmith(l , n , v , roughness * roughness , IBL);
    vec3 F = computeFresnel(h , v , metallic , albedo); 
    vec3 nom = D*F*G ; 
    float dot_v_n = max(dot(v , n) , 0.f); 
    float dot_l_n = max(dot(l , n), 0.f);
    float denom = 4*dot_v_n*dot_l_n + EPSILON; 
    vec3 specular = nom / denom ;
    vec3 Ks = F ; 
    vec3 Kd = vec3(1.f) - Ks ; 
    Kd *= (1.f - metallic) ;
    return (Kd * albedo / PI + specular) * dot_l_n;  
}

/**************************************************************************************************************/
LIGHT_COMPONENTS computePointLightsContribBRDF(float roughness , float metallic , vec3 albedo , bool IBL){
    vec3 v = -getViewDirection();
    vec3 n = getSurfaceNormal(); 
    LIGHT_COMPONENTS light;  
    uint i = 0 ; 
    vec3 L0 = vec3(0.f) ; 
    for(i = 0 ; i < point_light_number ; i++){
        vec3 light_pos = point_light_struct[i].position ; 
        vec3 brdf = computeBRDF(light_pos , n , v , roughness , metallic , albedo , IBL); 
        vec3 radiance = computeRadiancePointLight(n ,i); 
        L0 += brdf * radiance ; 
    }
    light.radiance = L0 ; 
    return light ; 
}

/**************************************************************************************************************/
LIGHT_COMPONENTS computeSpotLightsContribBRDF(float roughness , float metallic , vec3 albedo , bool IBL){
    vec3 v = -getViewDirection();
    vec3 n = getSurfaceNormal(); 
    LIGHT_COMPONENTS light;  
    uint i = 0 ; 
    vec3 L0 = vec3(0.f) ; 
    for(i = 0 ; i < spot_light_number ; i++){
        vec3 light_pos = spot_light_struct[i].position ; 
        vec3 brdf = computeBRDF(light_pos , n , v , roughness , metallic , albedo , IBL);
        vec3 l = normalize(light_pos - vertex_fragment_fragment_position);  
        vec3 radiance = computeRadianceSpotLight(n , l, i); 
        L0 += brdf * radiance ; 
    }
    light.radiance = L0 ; 
    return light ; 
}

/**************************************************************************************************************/
LIGHT_COMPONENTS computeDirectionalLightsContribBRDF(float roughness , float metallic , vec3 albedo , bool IBL){
    vec3 v = -getViewDirection();
    vec3 n = getSurfaceNormal(); 
    LIGHT_COMPONENTS light;  
    uint i = 0 ; 
    vec3 L0 = vec3(0.f) ; 
    for(i = 0 ; i < directional_light_number ; i++){
        vec3 l= normalize(directional_light_struct[i].position ); // This is wi 
        vec3 brdf = computeBRDF(directional_light_struct[i].position , n , v , roughness , metallic , albedo , IBL);
        light.radiance += computeRadianceDirectionalLight(n , l , i); 
        L0 += brdf * light.radiance ; 
    }
    light.radiance = L0 ; 
    return light ; 
}


/**************************************************************************************************************/
void main(){	
    vec4 final_computed_fragment ; 
    vec4 mrao = computeMetallicValue(); 
    float metallic = mrao.b ; 
    float roughness = mrao.g; 
    float ambient_occlusion = mrao.r ; 
    vec4 E = computeEmissiveValue() ;
    vec4 C = computeDiffuseValue() ;
    vec3 albedo = pow(C.rgb , vec3(2.2)); 
    LIGHT_COMPONENTS point = computePointLightsContribBRDF(roughness , metallic,  albedo , false );
    LIGHT_COMPONENTS spot = computeSpotLightsContribBRDF(roughness , metallic , albedo , false); 
    LIGHT_COMPONENTS direct = computeDirectionalLightsContribBRDF(roughness , metallic , albedo , false); 
    final_computed_fragment = vec4(point.radiance + spot.radiance + direct.radiance + vec3(0.1) * albedo * ambient_occlusion + E.rgb * material.emissive_factor, 0.f) ;  
    float alpha = C.a < material.alpha_factor ? C.a : material.alpha_factor; 
    final_computed_fragment.a = alpha;
    uint i = 0 ;
  /*  vec4 test = vec4(0.f);  
    for(i ; i < point_light_number ; i++){
        if(length(point_light_struct[i].position - vertex_fragment_fragment_position) < 10){
            test = vec4(1.f); 
        }
    } 
    final_computed_fragment = test ; */
    fragment = final_computed_fragment; 
}
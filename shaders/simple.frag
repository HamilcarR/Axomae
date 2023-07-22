#version 460 core

in vec4 COL ; 
in vec2 UV ;
in vec3 NORM ; 
in vec3 POS ; 

layout(binding=2) uniform sampler2D diffuse_map ; // Albedo 
layout(binding=3) uniform sampler2D normal_map ; 
layout(binding=4) uniform sampler2D metallic_map ; 
layout(binding=5) uniform sampler2D roughness_map ; 
layout(binding=6) uniform sampler2D ambiantocclusion_map ;
layout(binding=7) uniform sampler2D specular_map;
layout(binding=8) uniform sampler2D emissive_map;
layout(binding=9) uniform sampler2D opacity_map ;  
layout(binding=10) uniform samplerCube cubemap; 


out vec4 fragment ;

void main(){
	vec3 I = normalize(POS - vec3(0 , 0 , -100));
	vec3 R = reflect(I , normalize(NORM));
	vec3 S = texture(cubemap , R).rgb ;

	fragment = texture(diffuse  , UV) ; 
}


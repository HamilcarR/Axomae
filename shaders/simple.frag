#version 460 core

in vec4 COL ; 
in vec2 UV ;
in vec3 NORM ; 
in vec3 POS ; 

layout(binding=0) uniform sampler2D diffuse ; 
layout(binding=1) uniform sampler2D normal ; 
layout(binding=2) uniform sampler2D metallic ; 
layout(binding=3) uniform sampler2D roughness ; 
layout(binding=4) uniform sampler2D ambiantocclusion ;
layout(binding=5) uniform sampler2D specular;
layout(binding=6) uniform sampler2D emissive; 
layout(binding=7) uniform samplerCube cubemap; 
layout(binding=8) uniform sampler2D generic ;

out vec4 fragment ;

void main(){
	vec3 I = normalize(POS - vec3(0 , 0 , -100));
	vec3 R = reflect(I , normalize(NORM));
	vec3 S = texture(cubemap , R).rgb ;

	fragment = texture(diffuse  , UV) * COL; 
}


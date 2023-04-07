#version 430 core

in vec4 color ; 
in vec2 texcoord ;

layout(binding=0) uniform sampler2D diffuse ; 
layout(binding=1) uniform sampler2D normal ; 
layout(binding=2) uniform sampler2D metallic ; 
layout(binding=3) uniform sampler2D roughness ; 
layout(binding=4) uniform sampler2D ambiantocclusion ;
layout(binding=5) uniform sampler2D specular;
layout(binding=6) uniform sampler2D emissive; 
layout(binding=7) uniform sampler2D generic ;

out vec4 fragment ;

void main(){
	//fragment = vec4(1.); 
	fragment = texture(diffuse , texcoord) ; 

}


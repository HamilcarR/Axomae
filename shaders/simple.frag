#version 430 core

in vec4 color ; 
in vec2 texcoord ;

uniform sampler2D diffuse ; 

out vec4 fragment ;

void main(){

	fragment = texture(diffuse , texcoord) ; 
}


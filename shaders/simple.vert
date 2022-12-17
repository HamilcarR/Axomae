#version 430 core

layout(location=0) in vec3 position ;
layout(location=1) in vec2 uv ; 

out vec4 color ; 
out vec2 texcoord ;

void main(){
	gl_Position = vec4(position , 1.f) ; 
	color = vec4(1.) ; 
	texcoord = uv ; 
}

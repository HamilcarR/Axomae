#version 460 core

layout(location=0) in vec3 positions ;
layout(location=1) in vec3 colors ;
layout(location=2) in vec3 normals ; 
layout(location=3) in vec2 uv ; 
layout(location=4) in vec3 tangents ; 


uniform mat4 MAT_VP ; 
uniform mat4 MAT_MVP ; 
uniform mat4 MAT_VIEW;
uniform mat3 MAT_NORMAL ; 
out vec4 COL ; 
out vec2 UV ;
out vec3 NORM; 
out vec3 POS ; 

void main(){
	COL = vec4(colors , 1.f) ; 
	UV = uv ; 
	NORM = normals; 
	POS = positions; 

	gl_Position = MAT_MVP * vec4(positions , 1.f) ; 

}

#version 460 core

in vec3 cubemap_vector_sample; 

layout(binding=10) uniform samplerCube cubemap; 

out vec4 fragment ;

void main(){
	fragment = texture(cubemap , cubemap_vector_sample) ; 
}


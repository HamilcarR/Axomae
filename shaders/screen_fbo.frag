#version 460 core


layout(binding=9) uniform sampler2D framebuffer_map; 


in vec2 vertex_fragment_uv ;
out vec4 fragment ; 

const float offset = 1.f / 300.f ; 

vec4 samplePixel(float x , float y , float factor){
    return factor * texture(framebuffer_map , vertex_fragment_uv + vec2(x , y)); 
}
vec4 sharpen(){
    return samplePixel(0 , offset , -1) + samplePixel(-offset , 0 , -1) + samplePixel(0 , 0 , 4) + samplePixel(offset , 0 , -1) + samplePixel(0 , -offset , -1);  
}
void main(){
    
    fragment = sharpen() ; 
    //fragment = vec4(1.f);  
}
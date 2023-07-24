#version 460 core


layout(binding=10) uniform samplerCube cubemap ; 


out vec4 fragment;

in vec3 vertex_fragment_fragment_position ;

const float PI = 3.14159265358;
const float zenith_max = PI / 2 ; 
const float azimuth_max = 2 * PI ;
const float delta = 0.075 ; // set as uniform ?  
int number_samples = 0 ;
const vec3 UP = vec3(0.f , 1.f , 0.f); 




vec3 computeIrradiance(vec3 normal){
    vec3 right = normalize(cross(UP , normal));
    vec3 up = normalize(cross(normal , right)); 
    vec3 irradiance = vec3(0.f); 
    float theta = 0.f ; 
    float phi = 0.f ;
    /* Riemann sum */
    for(phi = 0.f ; phi <= azimuth_max ; phi += delta){
        for(theta = 0.f ; theta <= zenith_max ; theta += delta ){
            float x = cos(phi) * sin(theta);
            float y = sin(theta) * sin(phi);
            float z = cos(theta);
            vec3 sample_vec = x * right + y * up + z * normal; 
            irradiance += texture(cubemap , sample_vec).rgb * cos(theta) * sin(theta);
            number_samples++; 
        }
    }    
    irradiance = PI * irradiance * (1.f / float(number_samples));  
    return irradiance;  
}


void main(){
    vec3 normal = normalize(vertex_fragment_fragment_position); 
    fragment = vec4(computeIrradiance(normal) , 1.f) ; 
}
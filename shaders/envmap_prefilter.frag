#version 460 core


out vec4 fragment;

layout(binding = 10) uniform samplerCube cubemap; 

in vec3 vertex_fragment_fragment_position; 

uniform float roughness; 
uniform uint envmap_resolution;
uniform uint samples_count;  
const float PI = 3.14159265;

float HaltonSequence(uint base , uint index){
    float result = 0.f ; 
    float digit_weight = 1.f ; 
    while(index > 0){
        digit_weight = digit_weight / float(base); 
        uint nominator = index % base ;
        result += float(nominator) *digit_weight;
        index = index / base ;   
    }
    return result ; 
}

// return uniformly spread , low discrepancy pseudo randoms float between 0 and 1 
float radicalInverse(uint bits) 
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; 
}

vec2 Hammersley(uint i , uint N){
    return vec2(float(i) / float(N) , radicalInverse(i)); 
}
vec2 pseudo_rand(uint i , uint N){
    return vec2(float(i) / float(N) , fract(sin(dot(vec2(i , N) ,vec2(12.9898,78.233))) * 43758.5453));
}

float DistributionGGX(vec3 N, vec3 H, float r)
{
    float a = r*r;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float r)
{
    float a = r*r;
	
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
	
    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;
	
    vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent   = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
	
    vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
} 


vec3 averageSample(vec3 L , float step_s , float mipLevel){
    vec3 col = vec3(0.f); 
    for(int i = -1 ; i <= 1 ; i++)
        for(int j = -1 ; j <= 1 ; j++)
            for(int k = -1 ; k <= 1 ; k++)
                col += textureLod(cubemap , L + vec3(i*step_s , j * step_s , k * step_s) , mipLevel).rgb ; 
    return col / 9.f; 
}



void main(){
    vec3 N = normalize(vertex_fragment_fragment_position);    
    vec3 R = N;
    vec3 V = R;

    float totalWeight = 0.0;   
    vec3 prefilteredColor = vec3(0.0);     
    
    if(roughness == 0)
        prefilteredColor = textureLod(cubemap , N , 0).rgb ; 
    else{
        float resolution = float(envmap_resolution); // resolution of source cubemap (per face)
        float saTexel  = 4.0 * PI / (6.0 * resolution * resolution);
        for(uint i = 0; i < samples_count; ++i){
            vec2 Xi = Hammersley(i, samples_count);
            vec3 H  = ImportanceSampleGGX(Xi, N, roughness);
            vec3 L  = reflect(-V , H);

            float NdotL = max(dot(N, L), 0.0);
            if(NdotL > 0.0){
                // sample from the environment's mip level based on roughness/pdf
                float D   = DistributionGGX(N, H, roughness);
                float NdotH = max(dot(N, H), 0.0);
                float HdotV = max(dot(H, V), 0.0);
                float pdf = D * NdotH / (4.0 * HdotV) + 0.0001; 
                
                float saSample = 1.0 / (float(samples_count) * pdf + 0.0001);
                float mipLevel = roughness == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel); 
                prefilteredColor += textureLod(cubemap, L , mipLevel).rgb * NdotL;
                totalWeight      += NdotL;
            }
        }
        prefilteredColor = prefilteredColor / totalWeight;
    }
    fragment=vec4(prefilteredColor , 1.f); 
}
#version 460 core

layout(binding = 1) uniform sampler2D framebuffer_map;

in vec2 vertex_fragment_uv;
out vec4 fragment;

/*Uniforms*/
uniform float gamma;
uniform float exposure;
uniform bool uniform_sharpen;
uniform bool uniform_edge;

const float offset = 1.f / 300.f;
float kernel[9] = float[](-0, -1, -0, -1, 5, -1, -0, -1, -0);

vec4 samplePixel(float x, float y, float factor) { return factor * texture(framebuffer_map, vertex_fragment_uv + vec2(x, y)); }
vec4 edge() {
  return samplePixel(0, offset, -1) + samplePixel(-offset, 0, -1) + samplePixel(0, 0, 4) + samplePixel(offset, 0, -1) + samplePixel(0, -offset, -1);
}

vec4 sharpen() {
  return samplePixel(-offset, offset, kernel[0]) + samplePixel(0, offset, kernel[1]) + samplePixel(offset, offset, kernel[2]) +
         samplePixel(-offset, 0, kernel[3]) + samplePixel(0, 0, kernel[4]) + samplePixel(offset, 0, kernel[5]) +
         samplePixel(-offset, -offset, kernel[6]) + samplePixel(0, -offset, kernel[7]) + samplePixel(offset, -offset, kernel[8]);
}

vec3 computeReinhardToneMapping(vec3 hdr_color) { return hdr_color.rgb / (hdr_color.rgb + vec3(1.f)); }

vec3 computeExposureToneMapping(vec3 hdr_color) { return vec3(1.f) - exp(-hdr_color * exposure); }

void main() {
  vec4 hdr_color;
  if (uniform_sharpen)
    hdr_color = sharpen();
  else if (uniform_edge)
    hdr_color = edge();
  else
    hdr_color = texture(framebuffer_map, vertex_fragment_uv);
  vec3 tone_mapped_color = computeExposureToneMapping(hdr_color.rgb);
  fragment = vec4(pow(tone_mapped_color.rgb, vec3(1.0 / gamma)), hdr_color.a);
}
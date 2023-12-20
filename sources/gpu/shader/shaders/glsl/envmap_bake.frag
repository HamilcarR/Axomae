#version 460 core

layout(binding = 11) uniform sampler2D environment_map;

out vec4 fragment;
in vec3 vertex_fragment_fragment_position;

const float PI = 3.14159265359;

vec3 sphericalToUv(in vec3 coord) {
  float x = coord.x;
  float y = coord.y;
  float z = coord.z;
  float theta = atan(z, x);
  float phi = acos(y);
  float u = theta / (2.0 * PI);
  float v = phi / PI;
  return vec3(u, v, 1.0);
}

void main() {
  vec3 uv = sphericalToUv(normalize(vertex_fragment_fragment_position));
  vec3 color = texture(environment_map, vec2(uv.x, uv.y)).rgb;
  fragment = vec4(color, 1.f);
}
#version 460 core

layout(location = 0) in vec3 positions;
layout(location = 1) in vec3 colors;
layout(location = 2) in vec3 normals;
layout(location = 3) in vec2 uv;
layout(location = 4) in vec3 tangents;

uniform mat4 MAT_VP;
uniform mat4 MAT_MVP;
uniform mat4 MAT_VIEW;
uniform mat4 MAT_MODEL;
uniform mat4 MAT_PROJECTION;
uniform mat3 MAT_NORMAL;

out vec3 cubemap_vector_sample;

void main() {
  cubemap_vector_sample = positions;
  vec4 POS = MAT_MVP * vec4(positions, 1.f);
  gl_Position = POS.xyww;
}

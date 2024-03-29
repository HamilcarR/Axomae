#version 460 core

layout(location = 0) in vec3 positions;
layout(location = 1) in vec3 colors;
layout(location = 2) in vec3 normals;
layout(location = 3) in vec2 uv;
layout(location = 4) in vec3 tangents;

uniform mat4 MAT_VP;
uniform mat4 MAT_MVP;

out vec3 vertex_fragment_fragment_position;
out vec2 vertex_fragment_texCoords;
void main() {
  vertex_fragment_texCoords = uv;
  vertex_fragment_fragment_position = positions;
  gl_Position = MAT_VP * vec4(positions, 1.f);
}
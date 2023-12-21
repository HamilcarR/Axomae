#version 460 core

layout(location = 0) in vec3 positions;
layout(location = 3) in vec2 uv;

out vec2 vertex_fragment_uv;

void main() {
  gl_Position = vec4(positions.x, positions.y, 0.f, 1.f);
  vertex_fragment_uv = uv;
}
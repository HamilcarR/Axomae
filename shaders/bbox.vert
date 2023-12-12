#version 460 core

layout(location = 0) in vec3 positions;
layout(location = 1) in vec3 colors;
layout(location = 2) in vec3 normals;
layout(location = 3) in vec2 uv;
layout(location = 4) in vec3 tangents;

uniform mat4 MAT_MVP;

void main() { gl_Position = MAT_MVP * vec4(positions, 1.f); }
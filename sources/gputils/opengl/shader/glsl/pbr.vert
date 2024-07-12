#version 460 core

/* Vertex Data */
layout(location = 0) in vec3 positions;
layout(location = 1) in vec3 colors;
layout(location = 2) in vec3 normals;
layout(location = 3) in vec2 uv;
layout(location = 4) in vec3 tangents;
/******************************************/

/* Uniforms*/
uniform vec3 camera_position;
uniform mat4 MAT_VP;
uniform mat4 MAT_MVP;
uniform mat4 MAT_VIEW;
uniform mat3 MAT_NORMAL;  // transpose(invert(model))
uniform mat4 MAT_MODEL;
uniform mat4 MAT_MODELVIEW;
uniform mat4 MAT_PROJECTION;
uniform mat4 MAT_INV_MODEL;
uniform mat4 MAT_INV_MODELVIEW;
/******************************************/

/* Interpolated Shader Output */
out vec4 vertex_fragment_colors;
out vec2 vertex_fragment_uv;
out vec3 vertex_fragment_normals;
out vec3 vertex_fragment_positions;
out vec3 vertex_fragment_fragment_position;
out vec3 vertex_fragment_camera_position; //is 0
/******************************************/
/* Flat Shader Output*/
out mat3 MAT_TBN;
/******************************************/

/* Constants */

/******************************************/

void main() {
  vertex_fragment_colors = vec4(colors, 1.f);
  vertex_fragment_uv = uv;
  vertex_fragment_normals = MAT_NORMAL * normals;
  vertex_fragment_fragment_position = vec3(MAT_MODELVIEW * vec4(positions, 1.f));
  vec3 T = normalize(tangents);
  vec3 N = normalize(normals);
  vec3 B = normalize(cross(N, T));
  T = MAT_NORMAL * T;
  N = MAT_NORMAL * N;
  B = MAT_NORMAL * B;
  MAT_TBN = mat3(T, B, N);
  gl_Position = MAT_MVP * vec4(positions, 1.f);
}
#ifndef GEOMETRY_SHAPES_H
#define GEOMETRY_SHAPES_H

namespace CUBE {

  // clang-format off

 const std::vector<float> vertices = {
      -1, -1, -1,  // 0
      1,  -1, -1,  // 1
      -1, 1,  -1,  // 2
      1,  1,  -1,  // 3
      -1, -1, 1,   // 4
      1,  -1, 1,   // 5
      -1, 1,  1,   // 6
      1,  1,  1    // 7
  };


  const std::vector<float> normals = {
    // Front face
    0,  0, -1,
    0,  0, -1,
    0,  0, -1,
    0,  0, -1,
   // Back face
    0,  0,  1,
    0,  0,  1,
    0,  0,  1,
    0,  0,  1,
   // Left face
   -1,  0,  0,
   -1,  0,  0,
   -1,  0,  0,
   -1,  0,  0,
   // Right face
    1,  0,  0,
    1,  0,  0,
    1,  0,  0,
    1,  0,  0,
   // Top face
    0,  1,  0,
    0,  1,  0,
    0,  1,  0,
    0,  1,  0,
   // Bottom face
    0, -1,  0,
    0, -1,  0,
    0, -1,  0,
    0, -1,  0
};



const std::vector<float> colors = {
    // Front face (red)
    1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
    // Back face (green)
    0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
    // Left face (blue)
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
    // Right face (yellow)
    1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0,
    // Top face (magenta)
    1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1,
    // Bottom face (cyan)
    0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1
};


 const std::vector<float> uv = {
  // Front face
  0, 0, 1, 0, 0, 1, 1, 1,
  // Back face
  0, 0, 1, 0, 0, 1, 1, 1,
  // Left face
  0, 0, 1, 0, 0, 1, 1, 1,
  // Right face
  0, 0, 1, 0, 0, 1, 1, 1,
  // Top face
  0, 0, 1, 0, 0, 1, 1, 1,
  // Bottom face
  0, 0, 1, 0, 0, 1, 1, 1
};

 const std::vector<float> tangents = {
  // Front face
  1, 0, 0,  1, 0, 0,  1, 0, 0,  1, 0, 0,
  // Back face
  1, 0, 0,  1, 0, 0,  1, 0, 0,  1, 0, 0,
  // Left face
  0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
  // Right face
  0, 0, 1,  0, 0, 1,  0, 0, 1,  0, 0, 1,
  // Top face
  1, 0, 0,  1, 0, 0,  1, 0, 0,  1, 0, 0,
  // Bottom face
  1, 0, 0,  1, 0, 0,  1, 0, 0,  1, 0, 0
};

 const std::vector<float> bitangents = {
  // Front face
  0, 1, 0,  0, 1, 0,  0, 1, 0,  0, 1, 0,
  // Back face
  0, 1, 0,  0, 1, 0,  0, 1, 0,  0, 1, 0,
  // Left face
  0, 1, 0,  0, 1, 0,  0, 1, 0,  0, 1, 0,
  // Right face
  0, 1, 0,  0, 1, 0,  0, 1, 0,  0, 1, 0,
  // Top face
  0, 0, 1,  0, 0, 1,  0, 0, 1,  0, 0, 1,
  // Bottom face
  0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1
};

  const std::vector<unsigned int> indices = {
    // Front face
    0, 1, 2, 1, 3, 2,
    // Back face
    4, 6, 5, 5, 6, 7,
    // Left face
    0, 2, 4, 4, 2, 6,
    // Right face
    1, 5, 3, 3, 5, 7,
    // Top face
    2, 3, 6, 6, 3, 7,
    // Bottom face
    0, 4, 1, 1, 4, 5
};

  // clang-format on
}  // namespace CUBE
#endif  // GEOMETRY_SHAPES_H

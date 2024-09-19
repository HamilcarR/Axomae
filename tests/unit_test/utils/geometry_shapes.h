#ifndef GEOMETRY_SHAPES_H
#define GEOMETRY_SHAPES_H
#include <vector>
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


 const std::vector<unsigned int> indices = {
    // Front face
    0, 1, 2, 1, 3, 2,
    // Back face
    5, 4, 6, 6, 7, 5,
    // Left face
    0, 2, 6, 0, 6, 4,
    // Right face
    1, 5, 7, 7, 3, 1,
    // Top face
    3, 7, 6, 2, 3, 6,
    // Bottom face
    0, 4, 5, 0, 5, 1
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

 }  // namespace CUBE

namespace QUAD {


  const std::vector<float> vertices = {
      -1, -1, -1,   // 0: Bottom Left
       1, -1, -1,   // 1: Bottom Right
      -1,  1, -1,   // 2: Top Left
       1,  1, -1    // 3: Top Right
  };

  const std::vector<float> normals = {
      0,  0, -1,   // Vertex 0
      0,  0, -1,   // Vertex 1
      0,  0, -1,   // Vertex 2
      0,  0, -1    // Vertex 3
  };

  const std::vector<float> colors = {
      1, 0, 0,   // Vertex 0
      1, 0, 0,   // Vertex 1
      1, 0, 0,   // Vertex 2
      1, 0, 0    // Vertex 3
  };

  const std::vector<float> uv = {
      0, 0,   // Vertex 0
      1, 0,   // Vertex 1
      0, 1,   // Vertex 2
      1, 1    // Vertex 3
  };

  const std::vector<float> tangents = {
      1, 0, 0,   // Vertex 0
      1, 0, 0,   // Vertex 1
      1, 0, 0,   // Vertex 2
      1, 0, 0    // Vertex 3
  };

  const std::vector<float> bitangents = {
      0, 1, 0,   // Vertex 0
      0, 1, 0,   // Vertex 1
      0, 1, 0,   // Vertex 2
      0, 1, 0    // Vertex 3
  };

  const std::vector<unsigned int> indices = {
      0, 1, 2,         
      1, 3, 2    
  };

}  // namespace QUAD

namespace TRIANGLE {


  const std::vector<float> vertices = {
      -1, -1, -1,   // 0: Left-Bottom
       1, -1, -1,   // 1: Right-Bottom
       0,  1, -1    // 2: Top-Center
  };

  const std::vector<float> normals = {
      0,  0, -1,   // Vertex 0
      0,  0, -1,   // Vertex 1
      0,  0, -1    // Vertex 2
  };

  const std::vector<float> colors = {
      0, 0, 1,   // Vertex 0
      0, 0, 1,   // Vertex 1
      0, 0, 1    // Vertex 2
  };

  const std::vector<float> uv = {
      0,   0,   // Vertex 0
      1,   0,   // Vertex 1
      0.5, 1    // Vertex 2
  };

  const std::vector<float> tangents = {
      1, 0, 0,   // Vertex 0
      1, 0, 0,   // Vertex 1
      1, 0, 0    // Vertex 2
  };

  const std::vector<float> bitangents = {
      0, 1, 0,   // Vertex 0
      0, 1, 0,   // Vertex 1
      0, 1, 0    // Vertex 2
  };

  const std::vector<unsigned int> indices = {
      0, 1, 2
  };

  // clang-format on
}  // namespace TRIANGLE

#endif  // GEOMETRY_SHAPES_H

#ifndef API_GEOMETRY_H
#define API_GEOMETRY_H
#include "api_common.h"
#include <cstdint>
namespace nova {
  class NvAbstractMesh {
   public:
    virtual ~NvAbstractMesh() = default;
    /**
     * @brief
     * Registers the transformation matrix of a mesh.
     * Format is in column major.
     */
    virtual ERROR_STATE registerTransform(const float transform[16]);
  };

  class NvAbstractTriMesh : public NvAbstractMesh {
   public:
    virtual ERROR_STATE registerBufferVertices(float *vert, size_t num);
    virtual ERROR_STATE registerBufferNormals(float *normals, size_t num);
    virtual ERROR_STATE registerBufferTangents(float *tangents, size_t num);
    virtual ERROR_STATE registerBufferBitangents(float *bitangents, size_t num);
    virtual ERROR_STATE registerBufferColors(float *colors, size_t num);
    virtual ERROR_STATE registerBufferUVs(float *uv, size_t num);
    virtual ERROR_STATE registerBufferIndices(unsigned *indices, size_t num);
    virtual ERROR_STATE registerInteropVertices(uint32_t vbo_id);
    virtual ERROR_STATE registerInteropNormals(uint32_t vbo_id);
    virtual ERROR_STATE registerInteropTangents(uint32_t vbo_id);
    virtual ERROR_STATE registerInteropBitangents(uint32_t vbo_id);
    virtual ERROR_STATE registerInteropColors(uint32_t vbo_id);
    virtual ERROR_STATE registerInteropUVs(uint32_t vbo_id);
    virtual ERROR_STATE registerInteropIndices(uint32_t vbo_id);
  };

  inline std::unique_ptr<NvAbstractTriMesh> create_trimesh();

}  // namespace nova

#endif

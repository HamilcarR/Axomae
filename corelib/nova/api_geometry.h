#ifndef API_GEOMETRY_H
#define API_GEOMETRY_H
#include "api_common.h"
#include <cstdint>
#include <internal/common/axstd/span.h>
namespace nova {
  class NvAbstractMesh {
   public:
    virtual ~NvAbstractMesh() = default;
    /**
     * @brief
     * Registers the transformation matrix of a mesh.
     * Format is in column major.
     */
    virtual ERROR_STATE registerTransform(const float transform[16]) = 0;
    virtual void getTransform(float transform[16]) const = 0;
  };

  class NvAbstractTriMesh : public NvAbstractMesh {
   public:
    virtual ERROR_STATE registerBufferVertices(float *vert, size_t num) = 0;
    virtual ERROR_STATE registerBufferNormals(float *normals, size_t num) = 0;
    virtual ERROR_STATE registerBufferTangents(float *tangents, size_t num) = 0;
    virtual ERROR_STATE registerBufferBitangents(float *bitangents, size_t num) = 0;
    virtual ERROR_STATE registerBufferColors(float *colors, size_t num) = 0;
    virtual ERROR_STATE registerBufferUVs(float *uv, size_t num) = 0;
    virtual ERROR_STATE registerBufferIndices(unsigned *indices, size_t num) = 0;
    virtual ERROR_STATE registerInteropVertices(uint32_t vbo_id) = 0;
    virtual ERROR_STATE registerInteropNormals(uint32_t vbo_id) = 0;
    virtual ERROR_STATE registerInteropTangents(uint32_t vbo_id) = 0;
    virtual ERROR_STATE registerInteropBitangents(uint32_t vbo_id) = 0;
    virtual ERROR_STATE registerInteropColors(uint32_t vbo_id) = 0;
    virtual ERROR_STATE registerInteropUVs(uint32_t vbo_id) = 0;
    virtual ERROR_STATE registerInteropIndices(uint32_t vbo_id) = 0;

    virtual axstd::span<float> getVertices() const = 0;
    virtual axstd::span<float> getNormals() const = 0;
    virtual axstd::span<float> getTangents() const = 0;
    virtual axstd::span<float> getBitangents() const = 0;
    virtual axstd::span<float> getColors() const = 0;
    virtual axstd::span<float> getUVs() const = 0;
    virtual axstd::span<unsigned> getIndices() const = 0;

    virtual uint32_t getInteropVertices() const = 0;
    virtual uint32_t getInteropNormals() const = 0;
    virtual uint32_t getInteropTangents() const = 0;
    virtual uint32_t getInteropBitangents() const = 0;
    virtual uint32_t getInteropColors() const = 0;
    virtual uint32_t getInteropUVs() const = 0;
    virtual uint32_t getInteropIndices() const = 0;
  };

  std::unique_ptr<NvAbstractTriMesh> create_trimesh();

}  // namespace nova

#endif

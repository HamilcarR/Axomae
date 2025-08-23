#ifndef API_GEOMETRY_H
#define API_GEOMETRY_H
#include "api_common.h"
#include <cstdint>
#include <internal/common/axstd/span.h>
namespace nova {

  class Transform;

  class AbstractMesh {
   public:
    virtual ~AbstractMesh() = default;
    /**
     * @brief
     * Registers the transformation matrix of a mesh.
     * Format is in column major.
     */
    virtual ERROR_STATE registerTransform(const Transform &t_ptr) = 0;
    virtual const Transform &getTransform() const = 0;
  };

  class Trimesh : public AbstractMesh {
   public:
    /**
     * @brief Registers vertex positions (as float3) into the mesh.
     * @param vert Pointer to the vertex data array.
     * @param num Number of vertices to register.
     * @return SUCCESS if valid, INVALID_BUFFER_STATE otherwise.
     */
    virtual ERROR_STATE registerBufferVertices(float *vert, size_t num) = 0;

    /**
     * @brief Registers normal vectors (as float3) into the mesh.
     * @param normals Pointer to the normal data array.
     * @param num Number of normals to register.
     * @return SUCCESS if valid, INVALID_BUFFER_STATE otherwise.
     */
    virtual ERROR_STATE registerBufferNormals(float *normals, size_t num) = 0;

    /**
     * @brief Registers tangent vectors (as float3) into the mesh.
     * @param tangents Pointer to the tangent data array.
     * @param num Number of tangents to register.
     * @return SUCCESS if valid, INVALID_BUFFER_STATE otherwise.
     */
    virtual ERROR_STATE registerBufferTangents(float *tangents, size_t num) = 0;

    /**
     * @brief Registers bitangent vectors (as float3) into the mesh.
     * @param bitangents Pointer to the bitangent data array.
     * @param num Number of bitangents to register.
     * @return SUCCESS if valid, INVALID_BUFFER_STATE otherwise.
     */
    virtual ERROR_STATE registerBufferBitangents(float *bitangents, size_t num) = 0;

    /**
     * @brief Registers color data (as float4) into the mesh.
     * @param colors Pointer to the color data array.
     * @param num Number of colors to register.
     * @return SUCCESS if valid, INVALID_BUFFER_STATE otherwise.
     */
    virtual ERROR_STATE registerBufferColors(float *colors, size_t num) = 0;

    /**
     * @brief Registers UV coordinates (as float2) into the mesh.
     * @param uv Pointer to the UV data array.
     * @param num Number of UVs to register.
     * @return SUCCESS if valid, INVALID_BUFFER_STATE otherwise.
     */
    virtual ERROR_STATE registerBufferUVs(float *uv, size_t num) = 0;

    /**
     * @brief Registers index buffer data (triangle indices) into the mesh.
     * @param indices Pointer to the index data array.
     * @param num Number of indices to register.
     * @return SUCCESS if valid, INVALID_BUFFER_STATE otherwise.
     */
    virtual ERROR_STATE registerBufferIndices(unsigned *indices, size_t num) = 0;

    /**
     * @brief Registers a vertex buffer object (VBO) ID for vertex data.
     * @param vbo_id The ID of the VBO to register.
     * @return SUCCESS if valid.
     */
    virtual ERROR_STATE registerInteropVertices(uint32_t vbo_id) = 0;

    /**
     * @brief Registers a vertex buffer object (VBO) ID for normal data.
     * @param vbo_id The ID of the VBO to register.
     * @return SUCCESS if valid.
     */
    virtual ERROR_STATE registerInteropNormals(uint32_t vbo_id) = 0;

    /**
     * @brief Registers a vertex buffer object (VBO) ID for tangent data.
     * @param vbo_id The ID of the VBO to register.
     * @return SUCCESS if valid.
     */
    virtual ERROR_STATE registerInteropTangents(uint32_t vbo_id) = 0;

    /**
     * @brief Registers a vertex buffer object (VBO) ID for bitangent data.
     * @param vbo_id The ID of the VBO to register.
     * @return SUCCESS if valid.
     */
    virtual ERROR_STATE registerInteropBitangents(uint32_t vbo_id) = 0;

    /**
     * @brief Registers a vertex buffer object (VBO) ID for color data.
     * @param vbo_id The ID of the VBO to register.
     * @return SUCCESS if valid.
     */
    virtual ERROR_STATE registerInteropColors(uint32_t vbo_id) = 0;

    /**
     * @brief Registers a vertex buffer object (VBO) ID for UV data.
     * @param vbo_id The ID of the VBO to register.
     * @return SUCCESS if valid.
     */
    virtual ERROR_STATE registerInteropUVs(uint32_t vbo_id) = 0;

    /**
     * @brief Registers a vertex buffer object (VBO) ID for index data.
     * @param vbo_id The ID of the VBO to register.
     * @return SUCCESS if valid.
     */
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

  TrimeshPtr create_trimesh();

}  // namespace nova

#endif

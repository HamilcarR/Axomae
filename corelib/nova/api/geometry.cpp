#include "private_includes.h"
namespace nova {
  std::unique_ptr<NvAbstractTriMesh> nv_create_trimesh() { return std::make_unique<NvTriMesh>(); }

  ERROR_STATE NvTriMesh::registerBufferVertices(float *vertices, size_t num) {
    if (!vertices || num == 0)
      return INVALID_BUFFER_STATE;
    attributes.vertices = axstd::span<float>(vertices, num);
    return SUCCESS;
  }

  ERROR_STATE NvTriMesh::registerBufferNormals(float *normals, size_t num) {
    if (!normals || num == 0)
      return INVALID_BUFFER_STATE;
    attributes.normals = axstd::span<float>(normals, num);
    return SUCCESS;
  }

  ERROR_STATE NvTriMesh::registerBufferTangents(float *tangents, size_t num) {
    if (!tangents || num == 0)
      return INVALID_BUFFER_STATE;
    attributes.tangents = axstd::span<float>(tangents, num);
    return SUCCESS;
  }

  ERROR_STATE NvTriMesh::registerBufferBitangents(float *bitangents, size_t num) {
    if (!bitangents || num == 0)
      return INVALID_BUFFER_STATE;
    attributes.bitangents = axstd::span<float>(bitangents, num);
    return SUCCESS;
  }

  ERROR_STATE NvTriMesh::registerBufferColors(float *colors, size_t num) {
    if (!colors || num == 0)
      return INVALID_BUFFER_STATE;
    attributes.colors = axstd::span<float>(colors, num);
    return SUCCESS;
  }

  ERROR_STATE NvTriMesh::registerBufferUVs(float *uv, size_t num) {
    if (!uv || num == 0)
      return INVALID_BUFFER_STATE;
    attributes.uv = axstd::span<float>(uv, num);
    return SUCCESS;
  }

  ERROR_STATE NvTriMesh::registerBufferIndices(unsigned *indices, size_t num) {
    if (!indices || num == 0)
      return INVALID_BUFFER_STATE;
    attributes.indices = axstd::span<unsigned>(indices, num);
    return SUCCESS;
  }

  ERROR_STATE NvTriMesh::registerTransform(const float transform[16]) {
    std::memcpy(this->transform, transform, sizeof(float) * 16);
    return SUCCESS;
  }

  ERROR_STATE NvTriMesh::registerInteropVertices(uint32_t vbo_id) {
    vbo_interop.vbo_vertices = vbo_id;
    return SUCCESS;
  }

  ERROR_STATE NvTriMesh::registerInteropNormals(uint32_t vbo_id) {
    vbo_interop.vbo_normals = vbo_id;
    return SUCCESS;
  }

  ERROR_STATE NvTriMesh::registerInteropTangents(uint32_t vbo_id) {
    vbo_interop.vbo_tangents = vbo_id;
    return SUCCESS;
  }

  ERROR_STATE NvTriMesh::registerInteropBitangents(uint32_t vbo_id) {
    vbo_interop.vbo_bitangents = vbo_id;
    return SUCCESS;
  }

  ERROR_STATE NvTriMesh::registerInteropColors(uint32_t vbo_id) {
    vbo_interop.vbo_colors = vbo_id;
    return SUCCESS;
  }

  ERROR_STATE NvTriMesh::registerInteropUVs(uint32_t vbo_id) {
    vbo_interop.vbo_uvs = vbo_id;
    return SUCCESS;
  }

  ERROR_STATE NvTriMesh::registerInteropIndices(uint32_t vbo_id) {
    vbo_interop.vbo_indices = vbo_id;
    return SUCCESS;
  }
}  // namespace nova

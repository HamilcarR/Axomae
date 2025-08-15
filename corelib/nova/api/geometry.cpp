#include "glm/gtc/type_ptr.hpp"
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

  void NvTriMesh::getTransform(float transform_r[16]) const {
    for (int i = 0; i < 16; i++)
      transform_r[i] = transform[i];
  }

  Object3D to_obj3d(const NvAbstractTriMesh &trimesh) {
    Object3D obj3d;
    obj3d.vertices = trimesh.getVertices();
    obj3d.indices = trimesh.getIndices();
    obj3d.normals = trimesh.getNormals();
    obj3d.tangents = trimesh.getTangents();
    obj3d.bitangents = trimesh.getBitangents();
    obj3d.colors = trimesh.getColors();
    obj3d.uv = trimesh.getUVs();
    return obj3d;
  }

  struct triangle_mesh_properties_t {
    std::size_t mesh_index;
    std::size_t triangle_index;
  };
  static void store_primitive(const nova::material::NovaMaterialInterface &mat,
                              nova::NovaResourceManager &manager,
                              const triangle_mesh_properties_t &m_indices) {
    nova::shape::ShapeResourcesHolder &res_holder = manager.getShapeData();
    auto tri = res_holder.addShape<nova::shape::Triangle>(m_indices.mesh_index, m_indices.triangle_index);
    manager.getPrimitiveData().addPrimitive<nova::primitive::NovaGeoPrimitive>(tri, mat);
  }

  static shape::triangle::mesh_vbo_ids create_vbo_pack(const NvAbstractTriMesh &mesh) {
    shape::triangle::mesh_vbo_ids vbo_pack{};
    vbo_pack.vbo_positions = mesh.getInteropVertices();
    vbo_pack.vbo_indices = mesh.getInteropIndices();
    vbo_pack.vbo_normals = mesh.getInteropNormals();
    vbo_pack.vbo_uv = mesh.getInteropUVs();
    vbo_pack.vbo_tangents = mesh.getInteropTangents();
    return vbo_pack;
  }

  static glm::mat4 convert_transform(const float transform[16]) {
    glm::mat4 final_transform;
    for (int i = 0; i < 16; i++) {
      glm::value_ptr(final_transform)[i] = transform[i];
    }
    return final_transform;
  }

  void setup_geometry_data(const NvAbstractTriMesh &mesh,
                           const float final_transform[16],
                           material::NovaMaterialInterface &material,
                           NovaResourceManager &manager,
                           std::size_t mesh_index,
                           bool use_interops) {
    const Object3D &geometry = to_obj3d(mesh);
    for (std::size_t triangle_index = 0; triangle_index < geometry.indices.size(); triangle_index += 3) {
      triangle_mesh_properties_t m_indices{};
      m_indices.mesh_index = mesh_index;
      m_indices.triangle_index = triangle_index;
      store_primitive(material, manager, m_indices);
    }
    nova::shape::ShapeResourcesHolder &res_holder = manager.getShapeData();
    std::size_t stored_mesh_index = res_holder.addTriangleMesh(geometry);

    res_holder.addTransform(convert_transform(final_transform), stored_mesh_index);
    if (core::build::is_gpu_build && use_interops) {
      shape::triangle::mesh_vbo_ids vbo_pack = create_vbo_pack(mesh);
      res_holder.addTriangleMesh(vbo_pack);
    }
  }

}  // namespace nova

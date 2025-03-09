#include "Drawable.h"
#include "Mesh.h"
#include "bake.h"
#include "nova/bake_render_data.h"
#include "primitive/nova_primitive.h"
#include "shape/nova_shape.h"
namespace nova_baker_utils {

  nova::shape::triangle::mesh_vbo_ids vbos_from_drawable(const Drawable &drawable) {
    namespace nst = nova::shape::triangle;
    nst::mesh_vbo_ids vbos{};
    if constexpr (core::build::is_gpu_build) {
      const PackedGLGeometryBuffer &pkd_geo = drawable.getMeshGLBuffers();
      vbos.vbo_positions = pkd_geo.getVertexBufferID().getID();
      vbos.vbo_normals = pkd_geo.getNormalBufferID().getID();
      vbos.vbo_uv = pkd_geo.getUVBufferID().getID();
      vbos.vbo_tangents = pkd_geo.getTangentxBufferID().getID();
      vbos.vbo_indices = pkd_geo.getIndexBufferID().getID();
    }
    return vbos;
  }

  struct triangle_mesh_properties_t {
    std::size_t mesh_index;
    std::size_t triangle_index;
  };
  static void store_primitive(std::size_t alloc_offset_primitives,
                              primitive_buffers_t &p_buffers,
                              const nova::material::NovaMaterialInterface &mat,
                              nova::NovaResourceManager &manager,
                              const triangle_mesh_properties_t &m_indices) {
    nova::shape::ShapeResourcesHolder &res_holder = manager.getShapeData();
    auto tri = res_holder.addShape<nova::shape::Triangle>(m_indices.mesh_index, m_indices.triangle_index);
    manager.getPrimitiveData().add_primitive<nova::primitive::NovaGeoPrimitive>(
        p_buffers.geo_primitive_alloc_buffer.data(), alloc_offset_primitives, tri, mat);
  }

  void setup_geometry_data(primitive_buffers_t &geometry_buffers,
                           const drawable_original_transform &drawable_transform,
                           std::size_t &alloc_offset_primitives,
                           nova::material::NovaMaterialInterface &material,
                           nova::NovaResourceManager &manager,
                           std::size_t mesh_index) {
    glm::mat4 final_transfo = drawable_transform.mesh_original_transformation;
    const Drawable *drawable = drawable_transform.mesh;
    const Mesh *mesh_object = drawable->getMeshPointer();
    const Object3D &geometry = mesh_object->getGeometry();
    for (std::size_t triangle_index = 0; triangle_index < geometry.indices.size(); triangle_index += 3) {
      triangle_mesh_properties_t m_indices{};
      m_indices.mesh_index = mesh_index;
      m_indices.triangle_index = triangle_index;
      store_primitive(alloc_offset_primitives, geometry_buffers, material, manager, m_indices);
      alloc_offset_primitives++;
    }
    nova::shape::ShapeResourcesHolder &res_holder = manager.getShapeData();
    std::size_t stored_mesh_index = res_holder.addTriangleMesh(geometry);
    res_holder.addTransform(final_transfo, stored_mesh_index);
    if constexpr (core::build::is_gpu_build) {
      auto vbo_pack = vbos_from_drawable(*drawable);
      res_holder.addTriangleMesh(vbo_pack);
    }
  }
}  // namespace nova_baker_utils
